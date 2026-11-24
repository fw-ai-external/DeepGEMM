import atexit
import os
import torch

# Set some default environment provided at setup
try:
    # noinspection PyUnresolvedReferences
    from .envs import persistent_envs

    for key, value in persistent_envs.items():
        if key not in os.environ:
            os.environ[key] = value
except ImportError:
    pass

# Configs
from . import _C

BF16_MEGAMOE_SITU = bool(getattr(_C, "BF16_MEGAMOE_SITU", False))
BF16_MEGAMOE_SITU_FAST_BACKWARD = bool(getattr(_C, "BF16_MEGAMOE_SITU_FAST_BACKWARD", False))
BF16_MEGAMOE_SITU_PRECISE_PIPELINE = bool(getattr(_C, "BF16_MEGAMOE_SITU_PRECISE_PIPELINE", False))
BF16_MEGAMOE_SEPARATE_W13 = bool(getattr(_C, "BF16_MEGAMOE_SEPARATE_W13", False))
BF16_MEGAMOE_SPLIT_BACKWARD = bool(getattr(_C, "BF16_MEGAMOE_SPLIT_BACKWARD", False))
BF16_MEGAMOE_COMPACT_SPLIT_BACKWARD = bool(getattr(_C, "BF16_MEGAMOE_COMPACT_SPLIT_BACKWARD", False))
BF16_MEGAMOE_SEPARATE_L1 = bool(getattr(_C, "BF16_MEGAMOE_SEPARATE_L1", False))
from ._C import (
    shutdown,
    set_num_sms,
    get_num_sms,
    set_tc_util,
    get_tc_util,
    set_ignore_compile_dims,
    set_block_size_multiple_of,
    set_pdl,
    get_pdl,
    use_deterministic_algorithms,
)

# cuBLASLt Kernels
from ._C import (
    cublaslt_gemm_nt, cublaslt_gemm_nn,
    cublaslt_gemm_tn, cublaslt_gemm_tt,
    cublaslt_nvfp4_gemm_nt,
    batched_syrk, batched_symm,
)

# DeepGEMM Kernels
from ._C import (
    # FP8-FP4 GEMMs (accept any FP8/FP4 operand combination)
    fp8_fp4_gemm_nt, fp8_fp4_gemm_nn,
    fp8_fp4_gemm_tn, fp8_fp4_gemm_tt,
    m_grouped_fp8_fp4_gemm_nt_contiguous,
    m_grouped_fp8_fp4_gemm_nn_contiguous,
    m_grouped_fp8_fp4_gemm_nt_masked,
    # FP8/FP4 GEMMs (alias)
    fp4_gemm_nt,
    fp8_gemm_nt, fp8_gemm_nn,
    fp8_gemm_tn, fp8_gemm_tt,
    m_grouped_fp4_gemm_nt_contiguous,
    m_grouped_fp8_gemm_nt_contiguous,
    m_grouped_fp8_gemm_nn_contiguous,
    m_grouped_fp4_gemm_nt_masked,
    m_grouped_fp8_gemm_nt_masked,
    k_grouped_fp4_gemm_nt_contiguous,
    k_grouped_fp8_gemm_nt_contiguous,
    k_grouped_fp8_gemm_tn_contiguous,
    # BF16 GEMMs
    bf16_gemm_nt, bf16_gemm_nn,
    bf16_gemm_tn, bf16_gemm_tt,
    m_grouped_bf16_gemm_nt_contiguous,
    m_grouped_bf16_gemm_nn_contiguous,
    m_grouped_bf16_gemm_nt_masked,
    k_grouped_bf16_gemm_tn_contiguous,
    # MegaGate kernels
    bf16_mega_gate, get_bf16_mega_gate_config,
    # Einsum kernels
    einsum,
    fp8_einsum,
    # Attention kernels
    get_mqa_logits_metadata,
    fp8_fp4_mqa_logits,
    get_paged_mqa_logits_metadata,
    fp8_fp4_paged_mqa_logits,
    get_sparse_mqa_logits_metadata,
    get_paged_sparse_mqa_logits_metadata,
    fp8_fp4_sparse_mqa_logits,
    fp8_fp4_paged_sparse_mqa_logits,
    # Legacy attention kernels
    fp8_mqa_logits,
    fp8_paged_mqa_logits,
    # Hyperconnection kernels
    mega_mhc,
    tf32_hc_prenorm_gemm,
    # Layout kernels
    transform_sf_into_required_layout,
    # MegaMoE
    get_block_m_for_mega_moe,
)

# Mega kernels
from .mega import (
    CombineOrderMode,
    RouteWeightMode,
    SymmBuffer,
    get_symm_buffer_for_mega_moe,
    transform_weights_for_mega_moe,
    transform_side_lora_for_mega_moe,
    fp8_fp4_mega_moe,
    fp8_fp4_mega_moe_side_lora,
    bf16_mega_moe,
    bf16_mega_moe_side_lora,
)
from .mega.backward import (
    MegaMoESideLoraBackwardResult,
    bf16_mega_moe_backward_dgrad,
    bf16_mega_moe_side_lora_backward,
    bf16_mega_moe_backward_w13,
    bf16_mega_moe_backward_w13_combine,
    bf16_mega_moe_backward_w2,
    bf16_mega_moe_backward_w2_combine,
    fp8_fp4_mega_moe_side_lora_backward,
    fp8_fp4_mega_moe_backward_dgrad_swiglu,
    mega_moe_backward_combine_grad_x,
)

# Epilogue classes
from ._C import epilogue

# Locality domain helpers
from . import locality_domain
from .locality_domain import (
    get_num_locality_domains,
    is_localization_available,
    localize,
    is_localized,
    destroy_localizer,
)

# Some utils
from . import testing
from . import utils
from .utils import *

# Legacy Triton kernels for A100
try:
    from . import legacy
except Exception as e:
    print(f"Failed to load legacy DeepGEMM A100 Triton kernels: {e}")


# Initialize CPP modules
_C.init(os.path.dirname(os.path.abspath(__file__)))
# Native cuBLASLt exit callbacks can free handle storage before C++ static destructors.
# Release handles and workspaces while Python, PyTorch and CUDA are still available.
atexit.register(shutdown)

__version__ = '2.8.2'

# Optional BF16-logit sigmoid scoring, preserving low-latency router precision.
bf16_mega_gate_supports_sigmoid_bf16 = True

# Mega-mHC split barriers are tagged by an epoch, so CUDA graph replays are safe.
mega_mhc_supports_graph_replay = True
