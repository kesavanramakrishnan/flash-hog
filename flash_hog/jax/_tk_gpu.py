"""ThunderKittens (Hopper/SM90) double-backward for causal attention.

Targeted for Hopper architecture

Default path is Pallas, must opt in to us TK:

  1. build the plugin once:  THUNDERKITTENS_PATH=... flash_hog/csrc/tk_bwdbwd/build.sh
  2. opt in at runtime:

       from flash_hog.jax import _tk_gpu as tk
       tk.enable()    # double-backward now runs with TK kernels
       ...
       tk.disable()   # normal Pallas

``enable()`` swaps flash-hog's double-backward rule for one that uses the TK kernels
when ``supported(...)`` says so and falls back to the original Pallas rule per-call
otherwise (non-causal, head_dim != 64, GQA, T % 128 != 0, non-Hopper).

The kernels live in ``flash_hog/csrc/tk_bwdbwd/`` and are compiled out-of-band into
``libtk_bwdbwd.so`` (see ``build.sh`` there).  This module loads the library lazily
and registers the XLA FFI target on first use.

Environment vars:
  - ``FLASH_HOG_TK_LIB``     -- path to libtk_bwdbwd.so (default: next to the sources)
  - ``FLASH_HOG_DISABLE_TK`` -- treat the library as absent (enable() then errors)
"""

from __future__ import annotations

import ctypes
import functools
import os

import jax
import jax.numpy as jnp
import numpy as np

_LIB_ENV = "FLASH_HOG_TK_LIB"
_DISABLE_ENV = "FLASH_HOG_DISABLE_TK"
_DEFAULT_LIB = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "csrc", "tk_bwdbwd", "libtk_bwdbwd.so",
)
_FFI_TARGET = "tk_bwdbwd"


@functools.cache
def _lib() -> ctypes.CDLL | None:
    """Load the FFI library and register the XLA target. None if unavailable."""
    if os.environ.get(_DISABLE_ENV):
        return None
    path = os.environ.get(_LIB_ENV, _DEFAULT_LIB)
    if not os.path.exists(path):
        return None
    lib = ctypes.CDLL(path)
    jax.ffi.register_ffi_target(_FFI_TARGET, jax.ffi.pycapsule(lib.TkBwdBwd), platform="CUDA")
    return lib


@functools.cache
def _on_hopper() -> bool:
    try:
        devices = jax.devices("gpu")
    except RuntimeError:
        return False
    return all(getattr(d, "compute_capability", "") == "9.0" for d in devices)


def supported(*, is_causal: bool, seq_len: int, head_dim: int,
              num_q_heads: int, num_kv_heads: int) -> bool:
    """True iff the TK kernels can serve this shape on this machine."""
    return (
        is_causal
        and head_dim == 64
        and seq_len % 128 == 0
        and num_q_heads == num_kv_heads   # no GQA
        and _on_hopper()
        and _lib() is not None
    )


def flash_bwdbwd(*, Q, K, V, O, dO, ddQ, ddK, ddV, L, scale: float):
    """Causal-attention double-backward via the TK kernels.

    Arguments are BTNH and output is dQ2, dK2, dV2, ddO
    """
    B, T, N, Hd = Q.shape

    def to_bhtd(x):
        return jnp.transpose(x, (0, 2, 1, 3))

    Qb, Kb, Vb, dOb, ddQb, ddKb, ddVb = (
        to_bhtd(x).astype(jnp.bfloat16) for x in (Q, K, V, dO, ddQ, ddK, ddV)
    )
    # D = rowsum(dO * O)
    D = jnp.sum(to_bhtd(dO).astype(jnp.float32) * to_bhtd(O).astype(jnp.float32), axis=-1)
    Lf = L.reshape(B, N, T).astype(jnp.float32)

    outs = jax.ffi.ffi_call(
        _FFI_TARGET,
        [jax.ShapeDtypeStruct((B, N, T, Hd), jnp.bfloat16)] * 4    # dQ2, ddO, dK2, dV2
        + [jax.ShapeDtypeStruct((B, N, T), jnp.float32)] * 2,      # dD, B (scratch)
    )(Qb, Kb, Vb, dOb, ddQb, ddKb, ddVb, Lf, D, scale=np.float32(scale))

    dQ2, ddO, dK2, dV2 = (to_bhtd(x).astype(Q.dtype) for x in outs[:4])
    return dQ2, dK2, dV2, ddO

# opt in switch
_original_rule = None   # non-None iff the TK rule is currently installed


def enable() -> None:
    """Opt in: route the attention double-backward through the TK kernels.

    Per-call fallback: shapes the kernels can't serve still run the original
    Pallas rule.  Idempotent; undo with ``disable()``.
    """
    global _original_rule
    if _lib() is None:
        raise RuntimeError(
            "libtk_bwdbwd.so not found — build it with flash_hog/csrc/tk_bwdbwd/build.sh "
            f"(or point {_LIB_ENV} at it)."
        )
    if _original_rule is not None:
        return  # already enabled

    from jax._src.cudnn.fused_attention_stablehlo import MaskType

    from flash_hog.jax import _attention_impl as impl

    pallas_rule = impl.dot_product_attention_bwd_rule_bwd_rule

    def tk_rule(mask_type, scale, res, g):
        query, key, value, out, activation, dO = res
        if not supported(
            is_causal=(mask_type == MaskType.CAUSAL),
            seq_len=query.shape[1],
            head_dim=query.shape[3],
            num_q_heads=query.shape[2],
            num_kv_heads=key.shape[2],
        ):
            return pallas_rule(mask_type, scale, res, g)
        ddQ, ddK, ddV = g
        dQ2, dK2, dV2, ddO = flash_bwdbwd(
            Q=query, K=key, V=value, O=out, dO=dO,
            ddQ=ddQ, ddK=ddK, ddV=ddV, L=activation, scale=scale,
        )
        return (dQ2, dK2, dV2, None, None), ddO

    _original_rule = pallas_rule
    impl.dot_product_attention_bwd_rule_bwd_rule = tk_rule
    jax.clear_caches()   # already-traced double-backwards captured the old rule


def disable() -> None:
    """Restore the stock Pallas double-backward rule."""
    global _original_rule
    if _original_rule is None:
        return
    from flash_hog.jax import _attention_impl as impl

    impl.dot_product_attention_bwd_rule_bwd_rule = _original_rule
    _original_rule = None
    jax.clear_caches()


def is_enabled() -> bool:
    return _original_rule is not None
