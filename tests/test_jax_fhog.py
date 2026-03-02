from functools import partial
from math import sqrt

import chex
import jax
import jax.numpy as jnp
import jax.random as jrandom
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

import flash_hog.jax.attention as attn


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
    # print(locals())
    return attn.dot_product_attention(q, k, v, is_causal=is_causal, scale=scale)


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


def test_jax_fhog_backward_single():
    keys = jrandom.split(jrandom.PRNGKey(42), 10)
    q = jrandom.normal(keys[0], (1, 128, 32, 64), dtype=jnp.bfloat16)
    k = jrandom.normal(keys[1], (1, 128, 32, 64), dtype=jnp.bfloat16)
    v = jrandom.normal(keys[2], (1, 128, 32, 64), dtype=jnp.bfloat16)
    do = jrandom.normal(keys[3], (1, 128, 32, 64), dtype=jnp.bfloat16)
    # ddq = jrandom.normal(keys[4], (1, 128, 32, 64), dtype=jnp.bfloat16)
    # ddk = jrandom.normal(keys[5], (1, 128, 32, 64), dtype=jnp.bfloat16)
    # ddv = jrandom.normal(keys[6], (1, 128, 32, 64), dtype=jnp.bfloat16)

    is_causal = True
    scale = 1.0 / sqrt(q.shape[-1])

    ref_dpa_bwd = make_reference_dpa_bwd(q, k, v, is_causal, scale, implementation="cudnn")
    fhog_bwd = make_fhog_dpa_bwd(q, k, v, is_causal, scale)
    # fhog_bwd = make_reference_dpa_bwd(q, k, v, is_causal, scale)

    ref_output = ref_dpa_bwd(do)
    fhog_output = fhog_bwd(do)

    # print(ref_output)
    # print(fhog_output)

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


def test_jax_fhog_backward_backward_single_causal():
    batch_size = 2
    n_queries = 512
    n_keys = 512
    hidden_dim = 64
    q_heads = 32
    kv_heads = 32
    keys = jrandom.split(jrandom.PRNGKey(42), 10)
    q = jrandom.normal(keys[0], (batch_size, n_queries, q_heads, hidden_dim), dtype=jnp.bfloat16)
    k = jrandom.normal(keys[1], (batch_size, n_keys, kv_heads, hidden_dim), dtype=jnp.bfloat16)
    v = jrandom.normal(keys[2], (batch_size, n_keys, kv_heads, hidden_dim), dtype=jnp.bfloat16)
    do = jrandom.normal(keys[3], (batch_size, n_queries, q_heads, hidden_dim), dtype=jnp.bfloat16)
    ddq = jrandom.normal(keys[4], (batch_size, n_queries, q_heads, hidden_dim), dtype=jnp.bfloat16)
    ddk = jrandom.normal(keys[5], (batch_size, n_keys, kv_heads, hidden_dim), dtype=jnp.bfloat16)
    ddv = jrandom.normal(keys[6], (batch_size, n_keys, kv_heads, hidden_dim), dtype=jnp.bfloat16)

    is_causal = True
    scale = 1.0 / sqrt(q.shape[-1])

    ref_dpa_bwd = make_reference_dpa_bwd_bwd(q, k, v, do, is_causal, scale)  # Only supported for xla
    fhog_bwd = make_fhog_dpa_bwd_bwd(q, k, v, do, is_causal, scale)

    ref_output = ref_dpa_bwd(ddq, ddk, ddv)
    fhog_output = fhog_bwd(ddq, ddk, ddv)

    from wadler_lindig import pprint

    for i, (fhog_output_item, ref_output_item) in enumerate(zip(fhog_output, ref_output)):
        try:
            chex.assert_trees_all_close(fhog_output_item, ref_output_item, rtol=100, atol=0.04)
            print(f"Passed at {i=}")
        except AssertionError as e:
            print(f"Failed at {i=}")
            print(e)


def test_jax_fhog_backward_backward_single_noncausal():
    batch_size = 2
    n_queries = 512
    n_keys = 512
    hidden_dim = 64
    q_heads = 32
    kv_heads = 32
    keys = jrandom.split(jrandom.PRNGKey(42), 10)
    q = jrandom.normal(keys[0], (batch_size, n_queries, q_heads, hidden_dim), dtype=jnp.bfloat16)
    k = jrandom.normal(keys[1], (batch_size, n_keys, kv_heads, hidden_dim), dtype=jnp.bfloat16)
    v = jrandom.normal(keys[2], (batch_size, n_keys, kv_heads, hidden_dim), dtype=jnp.bfloat16)
    do = jrandom.normal(keys[3], (batch_size, n_queries, q_heads, hidden_dim), dtype=jnp.bfloat16)
    ddq = jrandom.normal(keys[4], (batch_size, n_queries, q_heads, hidden_dim), dtype=jnp.bfloat16)
    ddk = jrandom.normal(keys[5], (batch_size, n_keys, kv_heads, hidden_dim), dtype=jnp.bfloat16)
    ddv = jrandom.normal(keys[6], (batch_size, n_keys, kv_heads, hidden_dim), dtype=jnp.bfloat16)

    is_causal = False
    scale = 1.0 / sqrt(q.shape[-1])

    ref_dpa_bwd = make_reference_dpa_bwd_bwd(q, k, v, do, is_causal, scale)  # Only supported for xla
    fhog_bwd = make_fhog_dpa_bwd_bwd(q, k, v, do, is_causal, scale)

    ref_output = ref_dpa_bwd(ddq, ddk, ddv)
    fhog_output = jax.jit(fhog_bwd)(ddq, ddk, ddv)

    from wadler_lindig import pprint

    failed_one = False
    for i, (fhog_output_item, ref_output_item) in enumerate(zip(fhog_output, ref_output)):
        try:
            chex.assert_trees_all_close(fhog_output_item, ref_output_item, rtol=100, atol=0.04)
            print(f"Passed at {i=}")
        except AssertionError as e:
            print(f"Failed at {i=}")
            print(e)
            failed_one = True
    if failed_one:
        raise AssertionError("Failed one or more tests")


def test_jax_fhog_backward_backward_sharded2batch():
    jax.config.update("jax_use_shardy_partitioner", True)
    batch_size = 16
    n_queries = 512
    n_keys = 512
    hidden_dim = 64
    q_heads = 32
    kv_heads = 32
    keys = jrandom.split(jrandom.PRNGKey(42), 10)
    q = jrandom.normal(keys[0], (batch_size, n_queries, q_heads, hidden_dim), dtype=jnp.bfloat16)
    k = jrandom.normal(keys[1], (batch_size, n_keys, kv_heads, hidden_dim), dtype=jnp.bfloat16)
    v = jrandom.normal(keys[2], (batch_size, n_keys, kv_heads, hidden_dim), dtype=jnp.bfloat16)
    do = jrandom.normal(keys[3], (batch_size, n_queries, q_heads, hidden_dim), dtype=jnp.bfloat16)
    ddq = jrandom.normal(keys[4], (batch_size, n_queries, q_heads, hidden_dim), dtype=jnp.bfloat16)
    ddk = jrandom.normal(keys[5], (batch_size, n_keys, kv_heads, hidden_dim), dtype=jnp.bfloat16)
    ddv = jrandom.normal(keys[6], (batch_size, n_keys, kv_heads, hidden_dim), dtype=jnp.bfloat16)

    is_causal = True
    scale = 1.0 / sqrt(q.shape[-1])

    mesh2 = Mesh(jax.devices("gpu")[:2], ("x",))

    spec = NamedSharding(mesh2, P("x", None, None, None))

    def complete_function(q, k, v, do, ddq, ddk, ddv):
        fhog_bwd = make_fhog_dpa_bwd_bwd(q, k, v, do, is_causal, scale)
        return fhog_bwd(ddq, ddk, ddv)

    # fhog_bwd = make_fhog_dpa_bwd_bwd(q, k, v, do, is_causal, scale)
    sharded_complete_fn = jax.jit(complete_function, in_shardings=(spec, spec, spec, spec, spec, spec, spec), out_shardings=(spec, spec, spec, spec))

    fhog_output = jax.jit(complete_function)(q, k, v, do, ddq, ddk, ddv)
    # breakpoint()
    sharded_output = sharded_complete_fn(q, k, v, do, ddq, ddk, ddv)

    for sharded_output_item, fhog_output_item in zip(sharded_output, fhog_output):
        try:
            chex.assert_trees_all_close(sharded_output_item, fhog_output_item, rtol=100, atol=0.04)
        except AssertionError as e:
            print(e)
