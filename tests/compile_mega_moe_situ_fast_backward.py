"""Compile precise/fast backward variants without launching a CUDA context.

Usage: python tests/compile_mega_moe_situ_fast_backward.py --output-dir /tmp/situ-compile
Checks bounded/unbounded SiTU, packed/separate weights, both M tiles, and the
existing SwiGLU/GeGLU and FP8 branches through the shared compatibility harness.
"""

import compile_mega_moe_compat as compat

original_cases = compat.cases


def backward_math_cases():
    for name, header, function, args in original_cases():
        if function != "sm100_fp8_fp4_mega_moe_backward_wave_impl":
            continue
        assert args[14] is False, (name, args)
        for fast_math in (False, True):
            variant = args.copy()
            variant[14] = fast_math
            yield f"{name}_fast{int(fast_math)}", header, function, variant


if __name__ == "__main__":
    compat.cases = backward_math_cases
    compat.main()
