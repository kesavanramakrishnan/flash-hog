from functools import partial

import chex
import jax
import jax.numpy as jnp
import jax.random as jrandom
from flash_hog._utils.jax_utils import tree_rearrange
from flash_hog.jax._pallas_gpu_kernel import MaskType, TuningConfig, flash_bwdbwd0
from jax.experimental.custom_partitioning import (
    BATCHING,
    SdyShardingRule,
    custom_partitioning,
)
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P


def partition_flash_bwdbwd(
    mesh,
    arg_shapes,
    result_shape,
    mask_type: MaskType,
    scale: float | None,
    config: TuningConfig,
):
    arg_shardings = jax.tree.map(lambda s: s.sharding, arg_shapes)
    result_sharding = jax.tree.map(lambda s: s.sharding, result_shape)

    # breakpoint()

    # print(f"arg_shardings: {arg_shardings}")
    # print(f"result_sharding: {result_sharding}")

    # rank = len(arg_shapes[0].shape)
    # print(f"arg_shardings: {arg_shardings}")
    # print(f"result_sharding: {result_sharding}")
    # print(f"rank: {rank}")
    # breakpoint()

    def lower_fn(*args):
        # breakpoint()
        return flash_bwdbwd0(*args, mask_type=mask_type, scale=scale, config=config)

    return mesh, lower_fn, result_sharding, arg_shardings


BATCHING = "..."


def flash_bwdbwd(
    Q,
    K,
    V,
    O,
    dO,
    ddQ,
    ddK,
    ddV,
    L,
    mask_type: MaskType,
    scale: float | None,
    config: TuningConfig,
):
    if Q.ndim == 3:  # Add a batch dimension to the input arguments if there is none coming in
        batched_args = tree_rearrange((Q, K, V, O, dO, ddQ, ddK, ddV, L), " ... -> 1 ...")
        result_batched = flash_bwdbwd(*batched_args, mask_type=mask_type, scale=scale, config=config)
        return tree_rearrange(result_batched, "1 ... -> ...")

    def flash_fn(
        *args,
    ):  # Partialed to include the kwargs, but compatible with custom_partitioning
        return flash_bwdbwd0(*args, mask_type=mask_type, scale=scale, config=config)

    # f_partitioned = custom_partitioning(flash_fn)
    # f_partitioned.def_partition(
    #     infer_sharding_from_operands=None, propagate_user_sharding=None, # GSPMD options, not needed
    #     partition=partial(partition_flash_bwdbwd, mask_type=mask_type, scale=scale, config=config),
    #     #                             Q                                  K                                 V                               O                                 dO                             ddQ                               ddK                              ddV                                 L      ->              dQ2                                dK2                                 dV2                           ddO
    #     sharding_rule='batch queries q_heads hidden_qk,  batch key_vals kv_heads hidden_qk, batch key_vals kv_heads hidden_v, batch queries q_heads hidden_v, batch queries q_heads hidden_v, batch queries q_heads hidden_qk, batch key_vals kv_heads hidden_qk, batch key_vals kv_heads hidden_v, batch queries q_heads -> batch queries q_heads hidden_qk, batch keys_vals kv_heads hidden_qk, batch keys_vals kv_heads hidden_v, batch queries q_heads hidden_v',
    #     # need_replication_factors=("queries", 'q_heads', 'hidden_qk', 'keys_vals', 'kv_heads', 'hidden_v')
    # )  # fmt: skip
    f_partitioned = flash_fn

    f_batched = jax.custom_batching.custom_vmap(f_partitioned)

    @f_batched.def_vmap
    def flash_bwdbwd_vmap_rule(axis_size, in_batched, *unbatched_args):
        assert all(in_batched), "For now we only support vmapping all inputs to higher order backward attention"

        # Flatten the batch dimensions into the existing one
        batched_args = tree_rearrange(unbatched_args, "new_batch old_batch ... -> (new_batch old_batch) ...")
        result = f_batched(*batched_args)

        # Unroll the batch dimensions back out to get the intended shape
        result_unrolled = tree_rearrange(
            result,
            "(new_batch old_batch) ... -> new_batch old_batch ...",
            new_batch=axis_size,
        )
        out_batched = jax.tree.map(lambda _: True, result_unrolled)
        return result_unrolled, out_batched

    return f_batched(
        Q,
        K,
        V,
        O,
        dO,
        ddQ,
        ddK,
        ddV,
        L,
    )


# Replaced with tests in tests/test_jax_fhog.py
# if __name__ == "__main__":
#     batch_size = 16
#     n_queries = 256
#     n_keys = 256
#     hidden_dim = 64
#     q_heads = 16
#     kv_heads = 8
#     Q = jrandom.normal(jrandom.PRNGKey(0), (batch_size, n_queries, q_heads, hidden_dim), dtype=jnp.bfloat16)
#     K = jrandom.normal(jrandom.PRNGKey(1), (batch_size, n_keys, kv_heads, hidden_dim), dtype=jnp.bfloat16)
#     V = jrandom.normal(jrandom.PRNGKey(2), (batch_size, n_keys, kv_heads, hidden_dim), dtype=jnp.bfloat16)
#     O = jrandom.normal(jrandom.PRNGKey(3), (batch_size, n_queries, q_heads, hidden_dim), dtype=jnp.bfloat16)
#     dO = jrandom.normal(jrandom.PRNGKey(4), (batch_size, n_queries, q_heads, hidden_dim), dtype=jnp.bfloat16)
#     ddQ = jrandom.normal(jrandom.PRNGKey(5), (batch_size, n_queries, q_heads, hidden_dim), dtype=jnp.bfloat16)
#     ddK = jrandom.normal(jrandom.PRNGKey(6), (batch_size, n_keys, kv_heads, hidden_dim), dtype=jnp.bfloat16)
#     ddV = jrandom.normal(jrandom.PRNGKey(7), (batch_size, n_keys, kv_heads, hidden_dim), dtype=jnp.bfloat16)
#     L = jrandom.normal(jrandom.PRNGKey(8), (batch_size, q_heads, n_queries))

#     # @jax.vmap
#     # def use_attn(*unbatched_args):
#     #     # one_batched_args = tree_rearrange(unbatched_args, "... -> 1 ...")
#     #     return flash_bwdbwd(*unbatched_args, mask_type=MaskType.CAUSAL, scale=1.0, config=TuningConfig(tile_q=128, tile_k=32, max_concurrent_steps=4))
#     out = jax.vmap(partial(flash_bwdbwd, mask_type=MaskType.CAUSAL, config=TuningConfig(tile_q=128, tile_k=32, max_concurrent_steps=4)))(Q, K, V, O, dO, ddQ, ddK, ddV, L)
#     # Check if there are any NaNs in the output
#     print(jax.tree.map(lambda o: jnp.any(jnp.isnan(o)), out))
#     print("Regular vmap works!")

#     num_devices = 2
#     devices = jax.devices("gpu")[:num_devices]
#     assert len(devices) == num_devices, f"Expected {num_devices} devices, got {len(devices)}"
#     device_mesh = jax.sharding.Mesh(devices, ("x",))

#     sharding_spec = NamedSharding(device_mesh, P(("x", None)))

#     jitted_fn = jax.jit(
#         partial(flash_bwdbwd, mask_type=MaskType.CAUSAL, config=TuningConfig(tile_q=128, tile_k=32, max_concurrent_steps=4)),
#         in_shardings=(sharding_spec, sharding_spec, sharding_spec, sharding_spec, sharding_spec, sharding_spec, sharding_spec, sharding_spec, sharding_spec),
#         out_shardings=(sharding_spec, sharding_spec, sharding_spec, sharding_spec),
#     )

#     out2 = jitted_fn(Q, K, V, O, dO, ddQ, ddK, ddV, L)
#     print("Custom partitioned vmap works!")
#     chex.assert_trees_all_close(out, out2)
