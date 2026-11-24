// Compile-only coverage for K3.1 virtual route lanes and its unchanged fallbacks.
// nvcc -std=c++20 --expt-relaxed-constexpr -arch=sm_103f --cubin -O3 \
//   -Ideep_gemm/include -Ithird-party/cutlass/include \
//   tests/compile_k31_route_prelude.cu -o /tmp/k31-route-prelude.cubin
#include <deep_gemm/impls/sm100_fp8_fp4_mega_moe_backward.cuh>

using namespace deep_gemm;

template <CombineOrderMode Order, bool RouteDot, bool Dispatch, bool Prepared>
void instantiate_k31_route_prelude() {
    auto ptr = reinterpret_cast<void*>(
        &sm100_bf16_mega_moe_backward_post_down_prelude<
            3584, 8, 32, 140, 2, Order,
            Dispatch, RouteDot, true, false, false, false, false,
            Prepared, 256>);
}

template <CombineOrderMode Order, bool RouteDot>
void instantiate_dispatch_modes() {
    instantiate_k31_route_prelude<Order, RouteDot, false, false>();
    instantiate_k31_route_prelude<Order, RouteDot, false, true>();
    instantiate_k31_route_prelude<Order, RouteDot, true, false>();
    instantiate_k31_route_prelude<Order, RouteDot, true, true>();
}

void instantiate_k31_route_preludes() {
    instantiate_dispatch_modes<CombineOrderMode::DeepEPV1, true>();
    instantiate_dispatch_modes<CombineOrderMode::DeepEPV1, false>();
    instantiate_dispatch_modes<CombineOrderMode::FixedTopK, true>();
    instantiate_dispatch_modes<CombineOrderMode::FixedTopK, false>();
}
