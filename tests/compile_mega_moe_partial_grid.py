"""Compile BF16 forward variants with locality-domain splitting disabled.

Usage: python tests/compile_mega_moe_partial_grid.py --output-dir /tmp/partial-grid
Uses the existing compatibility harness; requires nvcc/CUTLASS but no GPU.
"""

import compile_mega_moe_compat as compat

original_cases = compat.cases


def partial_grid_cases():
    for name, header, function, args in original_cases():
        if header != "sm100_bf16_mega_moe":
            continue
        # Existing cases omit some of the optional SiTU/separate-L1 arguments.
        # Make them explicit before appending kUseLocalityDomains=false.
        defaults = ["4.0f", "__builtin_inff()", False]
        assert 27 <= len(args) <= 30, (name, args)
        args = args + defaults[len(args) - 27 :] + [False]
        yield f"partial_grid_{name}", header, function, args


if __name__ == "__main__":
    compat.cases = partial_grid_cases
    compat.main()
