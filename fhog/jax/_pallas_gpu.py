from functools import partial

import jax
import jax.numpy as jnp
import jax.random as jrandom

from fhog._utils.jax_utils import tree_rearrange
from fhog.jax._pallas_gpu_kernel import MaskType, TuningConfig, _flash_bwdbwd0


def flash_bwdbwd(Q, K, V, O, dO, ddQ, ddK, ddV, L, mask_type: MaskType, scale: float | None, config: TuningConfig):
    if Q.ndim == 3:  # Add a batch dimension to the input arguments if there is none coming in
        batched_args = tree_rearrange((Q, K, V, O, dO, ddQ, ddK, ddV, L), " ... -> 1 ...")
        result_batched = flash_bwdbwd(*batched_args, mask_type=mask_type, scale=scale, config=config)
        breakpoint()
        return tree_rearrange(result_batched, "1 ... -> ...")

    f = jax.custom_batching.custom_vmap(partial(_flash_bwdbwd0, mask_type=mask_type, scale=scale, config=config))

    @f.def_vmap
    def flash_bwdbwd_vmap_rule(axis_size, in_batched, *unbatched_args):
        assert all(in_batched), "For now we only support vmapping all inputs to higher order backward attention"

        # Flatten the batch dimensions into the existing one
        batched_args = tree_rearrange(unbatched_args, "new_batch old_batch ... -> (new_batch old_batch) ...")
        result = f(*batched_args)

        # Unroll the batch dimensions back out to get the intended shape
        result_unrolled = tree_rearrange(result, "(new_batch old_batch) ... -> new_batch old_batch ...", new_batch=axis_size)
        out_batched = jax.tree.map(lambda _: True, result_unrolled)
        return result_unrolled, out_batched

    return f(
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


if __name__ == "__main__":
    batch_size = 16
    n_queries = 256
    hidden_dim = 64
    q_heads = 16
    kv_heads = 8
    n_keys = 256
    Q = jrandom.normal(jrandom.PRNGKey(0), (batch_size, n_queries, q_heads, hidden_dim), dtype=jnp.bfloat16)
    K = jrandom.normal(jrandom.PRNGKey(1), (batch_size, n_keys, kv_heads, hidden_dim), dtype=jnp.bfloat16)
    V = jrandom.normal(jrandom.PRNGKey(2), (batch_size, n_keys, kv_heads, hidden_dim), dtype=jnp.bfloat16)
    O = jrandom.normal(jrandom.PRNGKey(3), (batch_size, n_queries, q_heads, hidden_dim), dtype=jnp.bfloat16)
    dO = jrandom.normal(jrandom.PRNGKey(4), (batch_size, n_queries, q_heads, hidden_dim), dtype=jnp.bfloat16)
    ddQ = jrandom.normal(jrandom.PRNGKey(5), (batch_size, n_queries, q_heads, hidden_dim), dtype=jnp.bfloat16)
    ddK = jrandom.normal(jrandom.PRNGKey(6), (batch_size, n_keys, kv_heads, hidden_dim), dtype=jnp.bfloat16)
    ddV = jrandom.normal(jrandom.PRNGKey(7), (batch_size, n_keys, kv_heads, hidden_dim), dtype=jnp.bfloat16)
    L = jrandom.normal(jrandom.PRNGKey(8), (batch_size, q_heads, n_queries))

    # @jax.vmap
    # def use_attn(*unbatched_args):
    #     # one_batched_args = tree_rearrange(unbatched_args, "... -> 1 ...")
    #     return flash_bwdbwd(*unbatched_args, mask_type=MaskType.CAUSAL, scale=1.0, config=TuningConfig(tile_q=128, tile_k=32, max_concurrent_steps=4))

    out = jax.vmap(partial(flash_bwdbwd, mask_type=MaskType.CAUSAL, scale=1.0, config=TuningConfig(tile_q=128, tile_k=32, max_concurrent_steps=4)))(
        Q, K, V, O, dO, ddQ, ddK, ddV, L
    )
