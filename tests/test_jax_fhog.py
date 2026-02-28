from functools import partial
from math import sqrt

import chex
import jax
import jax.numpy as jnp
import jax.random as jrandom

import fhog.jax.attention as attn


def reference_dpa(q, k, v, is_causal: bool, scale: float, implementation: str = "xla"):
    return jax.nn.dot_product_attention(q, k, v, is_causal=is_causal, scale=scale, implementation=implementation)


def make_reference_dpa_bwd(q, k, v, is_causal: bool, scale: float, implementation: str = "xla"):
    # For reference dQ, dK, dV = Jac(Q, K, V) @ dO,
    # which means dQ, dK, dV are linear in dO, but not linear in Q, K, V
    """Produced function takes dO and returns dQ, dK, dV"""
    _out, vjp_fun = jax.vjp(partial(reference_dpa, is_causal=is_causal, scale=scale, implementation=implementation), q, k, v)
    return vjp_fun


def make_reference_dpa_bwd_bwd(q, k, v, do, is_causal: bool, scale: float):
    """Produced function takes ddQ, ddK, ddV and returns dQ2, dK2, dV2, ddO"""

    def f(Q, K, V, dO):  # Returns dQ, dK, dV
        vjp_fun = make_reference_dpa_bwd(Q, K, V, is_causal, scale)
        return vjp_fun(dO)

    _out, vjp_fun = jax.vjp(f, q, k, v, do)
    return lambda *args: vjp_fun(args)


def fhog_dpa(q, k, v, is_causal: bool, scale: float):
    print(locals())
    return attn.dot_product_attention(q, k, v, mask_type=attn.MaskType.CAUSAL if is_causal else attn.MaskType.NO_MASK, scale=scale)


def make_fhog_dpa_bwd(q, k, v, is_causal: bool, scale: float):
    # For reference dQ, dK, dV = Jac(Q, K, V) @ dO,
    # which means dQ, dK, dV are linear in dO, but not linear in Q, K, V
    """Produced function takes dO and returns dQ, dK, dV"""
    _out, vjp_fun = jax.vjp(partial(fhog_dpa, is_causal=is_causal, scale=scale), q, k, v)
    return vjp_fun


def make_fhog_dpa_bwd_bwd(q, k, v, do, is_causal: bool, scale: float):
    def f(Q, K, V, dO):  # Returns dQ2, dK2, dV2, ddO
        vjp_fun = make_fhog_dpa_bwd(Q, K, V, is_causal, scale)
        return vjp_fun(dO)

    _out, vjp_fun = jax.vjp(f, q, k, v, do)
    return lambda *args: vjp_fun(args)


# def test_jax_fhog_backward():
#     q = jnp.ones((1, 128, 32, 64), dtype=jnp.float16)
#     k = jnp.ones((1, 128, 32, 64), dtype=jnp.float16)
#     v = jnp.ones((1, 128, 32, 64), dtype=jnp.float16)
#     out = fhog_dpa(q, k, v, is_causal=False, scale=1.0)
#     out, vjp_fun = jax.vjp(fhog_dpa, q, k, v, is_causal=False, scale=1.0)
#     dQ, dK, dV = vjp_fun(jnp.ones_like(out))
#     print(out)
#     print(dQ)
#     print(dK)
#     print(dV)


def test_jax_fhog_backward():
    keys = jrandom.split(jrandom.PRNGKey(42), 10)
    q = jrandom.normal(keys[0], (1, 128, 32, 64), dtype=jnp.bfloat16)
    k = jrandom.normal(keys[1], (1, 128, 32, 64), dtype=jnp.bfloat16)
    v = jrandom.normal(keys[2], (1, 128, 32, 64), dtype=jnp.bfloat16)
    do = jrandom.normal(keys[3], (1, 128, 32, 64), dtype=jnp.bfloat16)
    # ddq = jrandom.normal(keys[4], (1, 128, 32, 64), dtype=jnp.bfloat16)
    # ddk = jrandom.normal(keys[5], (1, 128, 32, 64), dtype=jnp.bfloat16)
    # ddv = jrandom.normal(keys[6], (1, 128, 32, 64), dtype=jnp.bfloat16)

    is_causal = False
    scale = 1.0 / sqrt(q.shape[-1])

    ref_dpa_bwd = make_reference_dpa_bwd(q, k, v, is_causal, scale, implementation="cudnn")
    fhog_bwd = make_fhog_dpa_bwd(q, k, v, is_causal, scale)
    # fhog_bwd = make_reference_dpa_bwd(q, k, v, is_causal, scale)

    ref_output = ref_dpa_bwd(do)
    fhog_output = fhog_bwd(do)

    print(ref_output)
    print(fhog_output)

    chex.assert_trees_all_close(ref_output, fhog_output)

    # out = attn.dot_product_attention(q, k, v)
    # out, vjp_fun = jax.vjp(attn.dot_product_attention, q, k, v)
    # dQ, dK, dV = vjp_fun(jnp.ones_like(out))
    # (dQ, dK, dV), vjp_fun = jax.vjp(attn.dot_product_attention, q, k, v, dQ, dK, dV)
    # dQ2, dK2, dV2 = vjp_fun(jnp.ones_like(out))

    # print(out)
    # print(dQ2)
    # print(dK2)
    # print(dV2)


def test_jax_fhog_backward_backward():
    keys = jrandom.split(jrandom.PRNGKey(42), 10)
    q = jrandom.normal(keys[0], (1, 512, 32, 64), dtype=jnp.bfloat16)
    k = jrandom.normal(keys[1], (1, 512, 16, 64), dtype=jnp.bfloat16)
    v = jrandom.normal(keys[2], (1, 512, 16, 64), dtype=jnp.bfloat16)
    do = jrandom.normal(keys[3], (1, 512, 32, 64), dtype=jnp.bfloat16)
    ddq = jrandom.normal(keys[4], (1, 512, 32, 64), dtype=jnp.bfloat16)
    ddk = jrandom.normal(keys[5], (1, 512, 16, 64), dtype=jnp.bfloat16)
    ddv = jrandom.normal(keys[6], (1, 512, 16, 64), dtype=jnp.bfloat16)

    is_causal = True
    scale = 1.0 / sqrt(q.shape[-1])

    ref_dpa_bwd = make_reference_dpa_bwd_bwd(q, k, v, do, is_causal, scale)  # Only supported for xla
    fhog_bwd = make_fhog_dpa_bwd_bwd(q, k, v, do, is_causal, scale)

    ref_output = ref_dpa_bwd(ddq, ddk, ddv)
    fhog_output = fhog_bwd(ddq, ddk, ddv)

    print(ref_output)
    print(fhog_output)

    from wadler_lindig import pprint

    pprint(ref_output)
    ref_output_zeros = jax.tree.map(lambda x: jnp.mean(x == 0.0), ref_output)
    fhog_output_zeros = jax.tree.map(lambda x: jnp.mean(x == 0.0), fhog_output)
    print(jnp.mean(ref_output[0][0, 1, :, :] == 0.0))

    print(f"ref_output: {ref_output_zeros=}")
    print(f"fhog_output: {fhog_output_zeros=}")

    chex.assert_trees_all_close(fhog_output, ref_output)
