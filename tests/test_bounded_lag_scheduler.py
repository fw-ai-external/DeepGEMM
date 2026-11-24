"""CPU contract tests for the exact host/device bounded-lag cursor."""

import os
from pathlib import Path
import shutil
import subprocess

import pytest


def test_bounded_lag_task_stream(tmp_path):
    compiler = os.environ.get("CXX") or shutil.which("g++")
    if compiler is None:
        pytest.skip("a C++17 compiler is required")
    header = Path(__file__).resolve().parents[1] / "deep_gemm/include/deep_gemm/scheduler/bounded_lag_moe.cuh"
    # Only remove CUDA annotations; compile the actual production cursor body.
    code = (
        header.read_text()
        .replace("#pragma once", "")
        .replace("#include <cutlass/cutlass.h>", "#define CUTLASS_HOST_DEVICE")
    )
    code += r"""
#include <algorithm>
#include <array>
#include <cassert>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>
using namespace deep_gemm::sched;
uint64_t checked = 0;
struct Unit { uint32_t expert, first_m, count, pool; };

template<uint32_t BM, uint32_t Lag, uint32_t N1=24, uint32_t N2=14>
void check(const std::vector<uint32_t>& rows) {
    // The reference materializes units, independently of the candidate's
    // cursor/relative-position arithmetic and dynamically assigned workers.
    std::vector<Unit> units;
    uint32_t total_blocks=0;
    for(uint32_t e=0;e<rows.size();++e) {
        uint32_t blocks=(rows[e]+BM-1)/BM;
        for(uint32_t m=0;m<blocks;m+=8)
            units.push_back({e,m,std::min(8u,blocks-m),total_blocks});
        total_blocks+=blocks;
    }
    std::array<BoundedLagCursor<112,BM,N1,N2,Lag>,74> cursors{};
    std::vector<uint32_t> seen1(total_blocks),seen2(total_blocks);
    const uint32_t full1=static_cast<uint32_t>((1ull<<N1)-1), full2=static_cast<uint32_t>((1ull<<N2)-1);
    const uint32_t ring=std::max(1u,std::min(total_blocks,get_bounded_lag_ring_blocks(Lag)));
    uint32_t ticket=0,live=0,max_live=0;
    uint64_t hash=1469598103934665603ull;
    std::mt19937 random(42);
    auto count=[&](uint32_t e) {return rows.at(e);};
    auto emit=[&](uint32_t phase, uint32_t unit_index) {
        const auto& u=units.at(unit_index);
        for(uint32_t m=u.first_m;m<u.first_m+u.count;++m) {
            uint32_t p=u.pool+m, valid=std::min(BM,rows[u.expert]-m*BM);
            for(uint32_t n=0;n<(phase==1 ? N1 : N2);++n) {
                // Randomized, strongly skewed worker assignment tests jumps
                // in each cursor's global ticket. Workers span both domains.
                uint32_t worker=(ticket%5==0) ? 0 : random()%74;
                auto got=cursors[worker].next(ticket,count);
                assert(got.phase==phase && got.expert==u.expert && got.m==m && got.n==n);
                assert(got.pool_block==p && got.valid_m==valid);
                auto& mask=(phase==1 ? seen1[p] : seen2[p]);
                assert(!(mask&(1u<<n)));
                if(phase==1 && mask==0) {
                    ++live;
                    if(p>=ring) assert(seen2[p-ring]==full2);
                }
                if(phase==2) assert(seen1[p]==full1);
                mask|=1u<<n;
                if(phase==2 && mask==full2) --live;
                max_live=std::max(max_live,live);
                for(uint32_t x:{phase,u.expert,m,n}) {hash^=x;hash*=1099511628211ull;}
                ++ticket;
            }
        }
    };
    uint32_t issued1=0,issued2=0;
    while(issued1<units.size() && issued1<Lag+4) emit(1,issued1++);
    while(issued1<units.size()) {
        for(uint32_t i=0;i<4 && issued2<issued1;++i) emit(2,issued2++);
        for(uint32_t i=0;i<4 && issued1<units.size();++i) emit(1,issued1++);
    }
    while(issued2<units.size()) emit(2,issued2++);
    assert(ticket==total_blocks*(N1+N2) && live==0 && max_live<=8*(Lag+4));
    for(auto x:seen1) assert(x==full1);
    for(auto x:seen2) assert(x==full2);
    for(auto& cursor:cursors) assert(cursor.next(ticket,count).phase==0);
    checked+=ticket;
    std::cout<<BM<<' '<<Lag<<' '<<N1<<' '<<N2<<' '<<ticket<<' '<<hash<<'\n';
}

template<uint32_t BM>
void matrix() {
    for(uint32_t tokens:{1024,4096,8192,16384,32768,65536,131072,262144}) {
        std::vector<uint32_t> rows(112,tokens*16/112);
        for(uint32_t e=0;e<tokens*16%112;++e) ++rows[e];
        uint32_t capacity=((tokens+1919)/1920)*1920;
        uint32_t lag=get_bounded_lag_units(capacity,16,112);
        assert(lag==(tokens<=4096 ? 50u : 64u));
        if(lag==50) check<BM,50>(rows); else check<BM,64>(rows);
    }
    std::vector<uint32_t> concentrated(112,0);
    for(uint32_t e=0;e<16;++e) concentrated[e]=262144*8;
    check<BM,64>(concentrated);
    check<BM,50>(std::vector<uint32_t>(112,0));
    std::vector<uint32_t> one(112,0);one.back()=1;check<BM,50>(one);
    std::mt19937 random(20261106+BM);
    for(uint32_t trial=0;trial<32;++trial) {
        std::vector<uint32_t> rows(112);
        for(auto& n:rows) n=(random()%4==0) ? 0 : random()%(32*BM+1);
        if(trial%3==0) check<BM,8>(rows);
        else if(trial%3==1) check<BM,50>(rows);
        else check<BM,64>(rows);
    }
}
int main() {
    matrix<8>();matrix<16>();matrix<32>();matrix<64>();
    matrix<96>();matrix<128>();matrix<192>();matrix<240>();
    std::vector<uint32_t> rows(112,4096);
    check<128,16,14,24>(rows);check<128,16,1,32>(rows);
    check<128,16,32,1>(rows);check<128,16,7,7>(rows);
    std::cout<<"tasks "<<checked<<'\n';
}
"""
    source = tmp_path / "bounded_lag.cpp"
    source.write_text(code)
    executable = tmp_path / "bounded_lag"
    subprocess.run([compiler, "-std=c++17", "-O2", "-Wall", "-Wextra", str(source), "-o", str(executable)], check=True)
    first = subprocess.check_output([str(executable)], text=True, timeout=180)
    second = subprocess.check_output([str(executable)], text=True, timeout=180)
    assert first == second
    assert len(first.splitlines()) == 349
    assert int(first.splitlines()[-1].split()[1]) > 400_000_000
    print(first.splitlines()[-1])
