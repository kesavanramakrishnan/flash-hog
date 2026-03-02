"""
Main interface for Jax attention with support for higher order memory-efficient backward on GPU.
"""

from functools import partial

import jax
import jax.numpy as jnp
from jax import Array
from jax._src.cudnn.fused_attention_stablehlo import MaskType
from jaxtyping import Bool, Float, Int

import flash_hog.jax._attention_impl as attn_impl

# Reimplementation of much of jax._src.cudnn.fused_attention_stablehlo as used in jax.nn.dot_product_attention


@partial(jax.custom_vjp, nondiff_argnames=["mask_type", "scale"])
def dot_product_attention(
    query: Float[Array, "T N H"],
    key: Float[Array, "S N H"],
    value: Float[Array, "S N H"],
    mask_type: MaskType = MaskType.NO_MASK,
    scale: float | None = None,
):
    """
    Dimensions:
        T: Query length
        S: Key/value length
        N: Number of attention heads
        H: Head dimension

    """
    if scale is None:
        scale = query.shape[-1] ** -0.5
    dtype = query.dtype
    assert dtype == key.dtype == value.dtype
    return attn_impl.dot_product_attention_fwd(query, key, value, mask_type=mask_type, scale=scale)


@partial(jax.custom_vjp, nondiff_argnames=["mask_type", "scale"])
def _dot_product_attention_fwd(query, key, value, mask_type: bool, scale: float):
    """
    Forward pass, saving for regular backward.
    """
    print("Running _dot_product_attention_fwd")
    out, res = attn_impl.dot_product_attention_fwd_rule(query, key, value, mask_type=mask_type, scale=scale)
    return out, res


def _dot_product_attention_fwd_fwd(query, key, value, mask_type: bool, scale: float):
    """
    Run the forward pass, saving in expectation of a regular backward pass and a higher order backward pass.
    """
    print("Running _dot_product_attention_fwd_fwd")
    out, res = attn_impl.dot_product_attention_fwd_rule(query, key, value, mask_type=mask_type, scale=scale)
    return (out, res), res


def _dot_product_attention_fwd_bwd(mask_type: bool, scale: float, res, g):
    """
    Backward through the saving forward pass.
    """
    print("Running _dot_product_attention_fwd_bwd")
    # breakpoint()
    dO, dvjp_fun = g
    dQ2, dK2, dV2 = dvjp_fun.args_res
    # *_, stats, out = dvjp_fun.opaque_residuals  # TODO: Do I need dO from here?

    dQ, dK, dV = attn_impl.dot_product_attention_bwd_rule(mask_type=mask_type, scale=scale, res=res, g=dO)
    return dQ + dQ2, dK + dK2, dV + dV2
    # return dQ, dK, dV


_dot_product_attention_fwd.defvjp(_dot_product_attention_fwd_fwd, _dot_product_attention_fwd_bwd)


@partial(jax.custom_vjp, nondiff_argnames=["mask_type", "scale"])
def _dot_product_attention_bwd(mask_type: MaskType, scale: float, res, g):
    """
    Regular backward pass.
    """
    print("Running _dot_product_attention_bwd")
    grads = attn_impl.dot_product_attention_bwd_rule(mask_type=mask_type, scale=scale, res=res, g=g)
    return grads


def _dot_product_attention_bwd_fwd(mask_type: MaskType, scale: float, res, g):
    """
    Backward pass, saving for higher order backward.
    """
    print("Running _dot_product_attention_bwd_fwd")
    out, res = attn_impl.dot_product_attention_bwd_rule_fwd_rule(mask_type=mask_type, scale=scale, res=res, g=g)
    return out, res


def _dot_product_attention_bwd_bwd(mask_type: MaskType, scale: float, res, g):
    """
    Backward pass through the backward pass.
    """
    print("Running _dot_product_attention_bwd_bwd")
    grads = attn_impl.dot_product_attention_bwd_rule_bwd_rule(mask_type=mask_type, scale=scale, res=res, g=g)
    return grads


_dot_product_attention_bwd.defvjp(_dot_product_attention_bwd_fwd, _dot_product_attention_bwd_bwd)

dot_product_attention.defvjp(_dot_product_attention_fwd, _dot_product_attention_bwd)
