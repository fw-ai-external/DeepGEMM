"""Compile fork/upstream integration variants without a GPU or PyTorch.

Usage: python tests/compile_mega_moe_compat.py --output-dir /tmp/mega-moe-compile
Requires nvcc and the CUTLASS submodule. This checks compilation, not GPU results.
"""

import argparse
from pathlib import Path
import subprocess


def cases():
    for kind in ("bf16", "fp8_fp4"):
        for block_m in (32, 240):
            args = [1920, 512, 256, 4, 2, 2, block_m, 128,
                    64 if kind == "bf16" else 128, 16 if block_m == 32 else 40]
            if kind == "fp8_fp4":
                args += [128 if block_m == 32 else 256, 128]
            args += [3840]
            if kind == "fp8_fp4":
                args += [61440]
            args += [3, 1024 if kind == "bf16" else 512, 128, 128, 256,
                     8, 2, "__builtin_inff()", False, "ActivationType::SwiGLU", True]
            if kind == "bf16":
                args += [True]
            args += ["RouteWeightMode::PreDown"]
            if kind == "bf16":
                args += ["CombineOrderMode::FixedTopK"]
            args += [True]
            if kind == "bf16":
                args += [True]
            args += [128]
            header = f"sm100_{kind}_mega_moe_side_lora_forward"
            yield f"side_{kind}_m{block_m}", header, header + "_impl", args

    for bf16 in (False, True):
        args = [512, 256, 4, 32, 128, 128, 128, 128, 3, 8, 2,
                True, bf16, "ActivationType::SwiGLU", False,
                "RouteWeightMode::PreDown", "CombineOrderMode::FixedTopK",
                False, False, True, False, True, True, False, False, False, True]
        yield (f"side_backward_b{int(bf16)}", "sm100_bf16_mega_moe_side_lora_backward",
               "sm100_bf16_mega_moe_side_lora_backward_wave_impl", args)

    for kind in ("bf16", "fp8_fp4"):
        for block_m in (32, 240):
            for shared, training in ((0, False), (1, False), (0, True)):
                args = [1920, 512, 256, 4, shared, 2, block_m, 128, 64 if kind == "bf16" else 128]
                if kind == "fp8_fp4":
                    # Asymmetric L1/L2 store blocks, as `get_block_config_for_mega_moe` selects them
                    args += [16, 16] if block_m == 32 else [24, 8]
                    args += [128 if block_m == 32 else 256, 128]
                else:
                    args += [16 if block_m == 32 else 40]
                args += [1920]
                if kind == "fp8_fp4":
                    args += [30720]
                args += [3, 1024 if kind == "bf16" else 512, 128, 128, 256, 8, 2,
                         "__builtin_inff()", False]
                if kind == "fp8_fp4":
                    args += ["cutlass::detail::float_e2m1_unpacksmem_t"]
                args += ["ActivationType::GeGLU" if training else "ActivationType::SwiGLU", training]
                if kind == "bf16":
                    args += [training]
                args += ["RouteWeightMode::PostDown" if training else "RouteWeightMode::PreDown"]
                if kind == "bf16":
                    args += ["CombineOrderMode::DeepEPV1" if training else "CombineOrderMode::FixedTopK"]
                args += [training, training]
                header = f"sm100_{kind}_mega_moe"
                yield f"{kind}_m{block_m}_s{shared}_t{int(training)}", header, header + "_impl", args
                if kind == "bf16" and shared == 0:
                    yield (f"separate_fwd_m{block_m}_t{int(training)}", header, header + "_impl",
                           args + ["4.0f", "__builtin_inff()", True])
                if kind == "bf16" and training:
                    for linear_beta in ("__builtin_inff()", "8.0f", "25.0f"):
                        situ_args = args.copy()
                        situ_args[situ_args.index("ActivationType::GeGLU")] = "ActivationType::SiTU"
                        situ_args += ["4.0f", linear_beta]
                        suffix = "unbounded" if linear_beta == "__builtin_inff()" else f"linear{linear_beta[:-3]}"
                        yield f"situ_fwd_m{block_m}_{suffix}", header, header + "_impl", situ_args
                        yield f"situ_fwd_separate_m{block_m}_{suffix}", header, header + "_impl", situ_args + [True]
                if kind == "fp8_fp4" and not training:
                    fp8_args = args.copy()
                    fp8_args[fp8_args.index("cutlass::detail::float_e2m1_unpacksmem_t")] = "cutlass::float_e4m3_t"
                    yield f"fp8_fp8_m{block_m}_s{shared}", header, header + "_impl", fp8_args

    for bf16 in (False, True):
        for block_m in (32, 240):
            args = [512, 256, 4, block_m, 128, 128, 128 if block_m == 32 else 256,
                    128, 3, 8, 2, True, bf16,
                    "ActivationType::GeGLU" if bf16 else "ActivationType::SwiGLU", False,
                    "RouteWeightMode::PostDown", "CombineOrderMode::DeepEPV1",
                    False, False, True, True, True, True]
            yield (f"backward_b{int(bf16)}_m{block_m}", "sm100_fp8_fp4_mega_moe_backward",
                   "sm100_fp8_fp4_mega_moe_backward_wave_impl", args)

            if bf16:
                for activation in ("GeGLU", "SwiGLU"):
                    separate_args = args.copy()
                    separate_args[separate_args.index("ActivationType::GeGLU")] = f"ActivationType::{activation}"
                    separate_args += [False, False, False, "4.0f", "__builtin_inff()", True]
                    yield (f"separate_bwd_m{block_m}_{activation}", "sm100_fp8_fp4_mega_moe_backward",
                           "sm100_fp8_fp4_mega_moe_backward_wave_impl", separate_args)
                for linear_beta in ("__builtin_inff()", "8.0f"):
                    situ_args = args.copy()
                    situ_args[situ_args.index("ActivationType::GeGLU")] = "ActivationType::SiTU"
                    situ_args += [False, False, False, "4.0f", linear_beta]
                    suffix = "linear" if linear_beta == "8.0f" else "unbounded"
                    yield (f"situ_bwd_m{block_m}_{suffix}", "sm100_fp8_fp4_mega_moe_backward",
                           "sm100_fp8_fp4_mega_moe_backward_wave_impl", situ_args)
                    yield (f"situ_bwd_separate_m{block_m}_{suffix}", "sm100_fp8_fp4_mega_moe_backward",
                           "sm100_fp8_fp4_mega_moe_backward_wave_impl", situ_args + [True])

    for combine in (False, True):
        args = ["cute::UMMA::Major::MN", "cute::UMMA::Major::MN", 512, 256, 0,
                128, 256, 16, 4, 32, 32, 128, 4, 128, 128, 1, False, 8, 32,
                False, False, "GemmType::KGroupedContiguous", False,
                "cutlass::bfloat16_t", "epilogue::operators::Identity", 100,
                2, combine, "CombineOrderMode::DeepEPV1", 64 if combine else 0]
        yield f"wgrad_combine{int(combine)}", "sm100_bf16_gemm", "sm100_bf16_gemm_impl", args
        masked_args = args.copy()
        masked_args[7] = 64
        masked_args[9:11] = [128, 128]
        masked_args[18] = 128
        masked_args.append(True)
        yield f"wgrad_masked_tail_combine{int(combine)}", "sm100_bf16_gemm", "sm100_bf16_gemm_impl", masked_args

    for layout in (1, 2):
        yield (f"psum_layout{layout}", "smxx_layout", "transpose_and_pack_strided_fp32_into_ue8m0",
               [512, 48, 16, 4, True, layout])
    yield ("backward_prelude", "sm100_fp8_fp4_mega_moe_backward",
           "sm100_bf16_mega_moe_backward_post_down_prelude",
           [512, 4, 32, 8, 2, "CombineOrderMode::DeepEPV1"])
    yield ("backward_combine", "sm100_mega_moe_backward_combine",
           "sm100_mega_moe_backward_combine_grad_x", [2, 4, "CombineOrderMode::DeepEPV1"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nvcc", default="nvcc")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--case-prefix", default="")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    count = 0
    for name, header, function, template_args in cases():
        if not name.startswith(args.case_prefix):
            continue
        source = args.output_dir / f"{name}.cu"
        parameters = ", ".join(str(value).lower() if isinstance(value, bool) else str(value)
                               for value in template_args)
        source.write_text(
            f"#include <deep_gemm/impls/{header}.cuh>\nusing namespace deep_gemm;\n"
            f"void instantiate() {{ auto p = reinterpret_cast<void*>(&{function}<{parameters}>); }}\n"
        )
        command = [args.nvcc, "-std=c++20", "--expt-relaxed-constexpr", "-arch=sm_100a",
                   "--cubin", "-O3", "-I", str(root / "deep_gemm/include"),
                   "-I", str(root / "third-party/cutlass/include"), str(source),
                   "-o", str(source.with_suffix(".cubin"))]
        with source.with_suffix(".log").open("w") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"{name} failed; see {source.with_suffix('.log')}")
        count += 1
        print(f"Compiled {name}", flush=True)
    print(f"All {count} CUDA variants compiled successfully.")


if __name__ == "__main__":
    main()
