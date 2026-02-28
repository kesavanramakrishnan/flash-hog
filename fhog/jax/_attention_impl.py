"""
Implementations for the functions used in fhog.jax.attention.
"""

from functools import partial

import jax
from jax._src.cudnn.fused_attention_stablehlo import MaskType
from jax._src.cudnn.fused_attention_stablehlo import dot_product_attention as cuda_dot_product_attention
from jax.tree_util import Partial

from fhog.pallas_bwdbwd import TuningConfig, flash_bwdbwd0


def dot_product_attention_fwd(query, key, value, mask_type: MaskType, scale: float):
    """
    Forward pass, no saving.
    Only needs to return the output.
    """
    return cuda_dot_product_attention(query, key, value, mask_type=mask_type, scale=scale)


def dot_product_attention_fwd_rule(query, key, value, mask_type: MaskType, scale: float):
    """
    Forward pass, saving stats, Q, K, V and O.
    """
    out, vjp_fun = jax.vjp(partial(cuda_dot_product_attention, mask_type=mask_type, scale=scale), query, key, value)
    residual = vjp_fun
    return out, residual


def dot_product_attention_bwd_rule(mask_type: MaskType, scale: float, res, g):
    """
    Backward pass, no saving
    """
    vjp_fun = res
    # breakpoint()
    dQ, dK, dV = vjp_fun(g)
    return dQ, dK, dV


def dot_product_attention_bwd_rule_fwd_rule(mask_type: MaskType, scale: float, res, g):
    """
    Backward pass, saving for higher order backward
    """
    vjp_fun = res
    dO = g

    # query, key, value = vjp_fun.args_res
    # *_, stats, out = vjp_fun.opaque_residuals
    dQ, dK, dV = vjp_fun(dO)
    residual = (vjp_fun, dO)

    return (dQ, dK, dV), residual


def dot_product_attention_bwd_rule_bwd_rule(mask_type: MaskType, scale: float, res, g):
    """
    Backward pass through the backward pass
    """
    vjp_fun, dO = res
    query, key, value = vjp_fun.args_res
    *_, stats, out = vjp_fun.opaque_residuals
    vjp_fun_structure = jax.tree.structure(vjp_fun)
    # breakpoint()

    ddQ, ddK, ddV = g

    dQ2, dK2, dV2, ddO = flash_bwdbwd0(
        Q=query,
        K=key,
        V=value,
        O=out,
        dO=dO,
        ddQ=ddQ,
        ddK=ddK,
        ddV=ddV,
        L=stats,
        mask_type=mask_type,
        scale=scale,
        config=TuningConfig(tile_q=128, tile_k=32, max_concurrent_steps=4),
    )
    vjp_fun_grad = jax.tree.unflatten(vjp_fun_structure, [dQ2, dK2, dV2, None, None, None])  # TODO: Don't I need new dO in the last argument here?
    return vjp_fun_grad, ddO
