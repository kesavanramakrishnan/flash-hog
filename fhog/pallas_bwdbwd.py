import dataclasses
from functools import partial

import jax
import jax.experimental.mosaic.gpu  # noqa: F401
import jax.experimental.pallas as pl
import jax.experimental.pallas.mosaic_gpu as plgpu
import jax.experimental.pallas.triton as tlgpu
import jax.numpy as jnp
import jax.random as jrandom
from jax import Array, lax
from jax.extend import backend

# TODO: Implement autotuning using Tokamax


@dataclasses.dataclass(frozen=True)
class TuningConfig:
    tile_q: int
    tile_k: int
    max_concurrent_steps: int
    epilogue_tile_n: int = 64
    grid_minor_dim: int = 0
    grid_tile_width: int = 1


def flash_bwdbwd0(
    Q: Array,
    K: Array,
    V: Array,
    O: Array,
    dO: Array,
    ddQ: Array,
    ddK: Array,
    ddV: Array,
    L: Array,
    scale: float,
    config: TuningConfig,
):
    """
    Flash Higher-Order-Gradients (Flash Hog) backward pass.
    This implementation is the most generic pallas implementation, and should work on both GPU and TPU.

    Args:
        Q: Query tensor of shape (batch_size, N_QUERIES, HIDDEN_DIM)
        K: Key tensor of shape (batch_size, N_KEYS, HIDDEN_DIM)
        V: Value tensor of shape (batch_size, N_KEYS, HIDDEN_DIM)
        O: Output tensor of shape (batch_size, N_QUERIES, HIDDEN_DIM)
        dO: Gradient of the output tensor with respect to the input tensor of shape (batch_size, N_QUERIES, HIDDEN_DIM)
        ddQ: Gradient of the query tensor with respect to the input tensor of shape (batch_size, N_QUERIES, HIDDEN_DIM)
        ddK: Gradient of the key tensor with respect to the input tensor of shape (batch_size, N_KEYS, HIDDEN_DIM)
        ddV: Gradient of the value tensor with respect to the input tensor of shape (batch_size, N_KEYS, HIDDEN_DIM)
    Returns:
        dQ2: Higher order gradients of Q (batch_size, N_QUERIES, HIDDEN_DIM)
        dK2: Higher order gradients of K (batch_size, N_KEYS, HIDDEN_DIM)
        dV2: Higher order gradients of V (batch_size, N_KEYS, HIDDEN_DIM)
        ddO: Higher order gradients of dO (batch_size, N_QUERIES, HIDDEN_DIM)
    """
    dtype = Q.dtype
    batch_size, n_queries, q_heads, hidden_dim = Q.shape
    n_keys, kv_heads = K.shape[1], K.shape[2]
    # Assert the remaining shapes are consistent
    assert K.shape == (batch_size, n_keys, kv_heads, hidden_dim)
    assert V.shape == (batch_size, n_keys, kv_heads, hidden_dim)
    assert O.shape == (batch_size, n_queries, q_heads, hidden_dim)
    assert dO.shape == (batch_size, n_queries, hidden_dim)
    assert ddQ.shape == (batch_size, n_queries, hidden_dim)
    assert ddK.shape == (batch_size, n_keys, kv_heads, hidden_dim)
    assert ddV.shape == (batch_size, n_keys, kv_heads, hidden_dim)
    assert L.shape == (batch_size, n_queries, q_heads)
    assert config.tile_q > 0 and n_queries % config.tile_q == 0, "Haven't tested that tile_q supports non-divisible n_queries"

    # Compute D before the kernel
    D = jnp.einsum("bqd,bqd->bq", dO, O)  # (batch_size, n_queries)

    # Allocate space for stage 1 outputs
    # dQ2 = jnp.empty_like(Q)
    # ddO = jnp.empty_like(O)
    # dD = jnp.empty_like(D)
    # B = jnp.empty_like(D)
    # Instead just define their shapes and dtypes
    dQ2_shape_dtype = jax.ShapeDtypeStruct(Q.shape, dtype)
    ddO_shape_dtype = jax.ShapeDtypeStruct(O.shape, dtype)
    dD_shape_dtype = jax.ShapeDtypeStruct(D.shape, jnp.float32)
    B_shape_dtype = jax.ShapeDtypeStruct(D.shape, jnp.float32)

    dK2_shape_dtype = jax.ShapeDtypeStruct(K.shape, dtype)
    dV2_shape_dtype = jax.ShapeDtypeStruct(V.shape, dtype)

    def bwd_bwd_kernel_stage1(
        Q_ref,  # (tile_q, hidden_dim)
        K_ref,  # (n_keys, hidden_dim)
        V_ref,  # (n_keys, hidden_dim)
        D_ref,  # (tile_q,)
        dO_ref,  # (tile_q, hidden_dim)
        ddQ_ref,  # (tile_q, hidden_dim)
        ddK_ref,  # (n_keys, hidden_dim)
        ddV_ref,  # (n_keys, hidden_dim)
        L_ref,  # (tile_q,)
        # Outputs
        dQ2_ref,  # (tile_q, hidden_dim)
        ddO_ref,  # (tile_q, hidden_dim)
        dD_ref,  # (tile_q,)
        B_ref,  # (tile_q,)
    ):  # fmt: off
        Q_i = Q_ref[:, :]
        L_i = L_ref[:]
        ddQ_i = ddQ_ref[:, :]

        def dd_loop_body(ki, carry):
            dD_i_acc = carry
            kslice = pl.ds(ki * config.tile_k, config.tile_k)

            K_j = K_ref[kslice, :]
            # V_j = V_ref[kslice, :]
            ddK_j = ddK_ref[kslice, :]
            # ddV_j = ddV_ref[kslice, :]

            S_ij = pl.dot(Q_i, K_j, trans_b=True) * scale
            P_ij = jnp.exp(S_ij - L_i[:, None])

            # dP_ij = pl.dot(dO_i, V_j.T)
            ddS_ij = (pl.dot(ddQ_i, K_j, trans_b=True) + pl.dot(Q_i, ddK_j, trans_b=True)) * scale

            dD_i = jnp.sum(ddS_ij * P_ij, axis=1)
            dD_i_acc += dD_i
            return dD_i_acc

        dD_i = jax.lax.fori_loop(0, pl.cdiv(n_keys, config.tile_k), dd_loop_body, jnp.zeros((config.tile_q,), dtype=jnp.float32))
        dD_ref[:] = dD_i

        D_i = D_ref[:]
        dO_i = dO_ref[:, :]

        def b_loop_body(ki, carry):
            B_i_acc = carry
            kslice = pl.ds(ki * config.tile_k, config.tile_k)
            K_j = K_ref[kslice, :]
            V_j = V_ref[kslice, :]
            ddK_j = ddK_ref[kslice, :]
            ddV_j = ddV_ref[kslice, :]

            S_ij = pl.dot(Q_i, K_j, trans_b=True) * scale
            P_ij = jnp.exp(S_ij - L_i[:, None])

            dP_ij = pl.dot(dO_i, V_j, trans_b=True)

            ddS_ij = (pl.dot(ddQ_i, K_j, trans_b=True) + pl.dot(Q_i, ddK_j, trans_b=True)) * scale

            dP2_ij = pl.dot(dO_i, ddV_j, trans_b=True) - dP_ij * dD_i[:, None] - ddS_ij * D_i[:, None] + dP_ij * ddS_ij
            B_i = jnp.sum(dP2_ij * P_ij, axis=1)
            B_i_acc += B_i

            return B_i_acc

        B_i = jax.lax.fori_loop(0, pl.cdiv(n_keys, config.tile_k), b_loop_body, jnp.zeros((config.tile_q,), dtype=jnp.float32))
        B_ref[:] = B_i

        def dQ2_ddO_loop_body(ki, carry):
            dQ2_i_acc, ddO_i_acc = carry
            kslice = pl.ds(ki * config.tile_k, config.tile_k)

            K_j = K_ref[kslice, :]
            V_j = V_ref[kslice, :]
            ddK_j = ddK_ref[kslice, :]
            ddV_j = ddV_ref[kslice, :]

            # Compute attention scores
            S_ij = pl.dot(Q_i, K_j, trans_b=True) * scale
            P_ij = jnp.exp(S_ij - L_i[:, None])

            dP_ij = pl.dot(dO_i, V_j, trans_b=True)

            ddS_ij = (pl.dot(ddQ_i, K_j, trans_b=True) + pl.dot(Q_i, ddK_j, trans_b=True)) * scale

            dP2_ij = pl.dot(dO_i, ddV_j, trans_b=True) - dP_ij * dD_i[:, None] - ddS_ij * D_i[:, None] + dP_ij * ddS_ij
            dS2_ij = P_ij * (dP2_ij - B_i[:, None]) * scale

            dS_ij = scale * P_ij * (dP_ij - D_i[:, None])
            dQ2_i_acc += pl.dot(dS_ij.astype(ddK_j.dtype), ddK_j)
            dQ2_i_acc += pl.dot(dS2_ij.astype(K_j.dtype), K_j)

            ddP_ij = P_ij * (ddS_ij - dD_i[:, None])
            ddO_i_acc += pl.dot(ddP_ij.astype(V_j.dtype), V_j)
            ddO_i_acc += pl.dot(P_ij.astype(ddV_j.dtype), ddV_j)
            return (dQ2_i_acc, ddO_i_acc)

        dQ2_i, ddO_i = jax.lax.fori_loop(
            0,
            pl.cdiv(n_keys, config.tile_k),
            dQ2_ddO_loop_body,
            (jnp.zeros((config.tile_q, hidden_dim), dtype=jnp.float32), jnp.zeros((config.tile_q, hidden_dim), dtype=jnp.float32)),
        )
        dQ2_ref[:] = dQ2_i.astype(dQ2_ref.dtype)
        ddO_ref[:] = ddO_i.astype(ddO_ref.dtype)

    bwd_bwd_stage1 = pl.pallas_call(
        bwd_bwd_kernel_stage1,
        out_shape=(dQ2_shape_dtype, ddO_shape_dtype, dD_shape_dtype, B_shape_dtype),
        grid=(pl.cdiv(n_queries, config.tile_q), batch_size),
        in_specs=[
            pl.BlockSpec((None, config.tile_q, hidden_dim), lambda q, b: (b, q, 0)),  # Q, None indicates to squeeze the index (assuming sliced)
            pl.BlockSpec((None, n_keys, hidden_dim), lambda q, b: (b, 0, 0)),  # K
            pl.BlockSpec((None, n_keys, hidden_dim), lambda q, b: (b, 0, 0)),  # V
            pl.BlockSpec((None, config.tile_q), lambda q, b: (b, q)),  # D
            pl.BlockSpec((None, config.tile_q, hidden_dim), lambda q, b: (b, q, 0)),  # dO
            pl.BlockSpec((None, config.tile_q, hidden_dim), lambda q, b: (b, q, 0)),  # ddQ
            pl.BlockSpec((None, n_keys, hidden_dim), lambda q, b: (b, 0, 0)),  # ddK
            pl.BlockSpec((None, n_keys, hidden_dim), lambda q, b: (b, 0, 0)),  # ddV
            pl.BlockSpec((None, config.tile_q), lambda q, b: (b, q)),  # L
        ],
        out_specs=[
            pl.BlockSpec((None, config.tile_q, hidden_dim), lambda q, b: (b, q, 0)),  # dQ2
            pl.BlockSpec((None, config.tile_q, hidden_dim), lambda q, b: (b, q, 0)),  # ddO
            pl.BlockSpec((None, config.tile_q), lambda q, b: (b, q)),  # dD
            pl.BlockSpec((None, config.tile_q), lambda q, b: (b, q)),  # B
        ],
    )

    dQ2, ddO, dD, B = bwd_bwd_stage1(Q, K, V, D, dO, ddQ, ddK, ddV, L)

    def bwd_bwd_kernel_stage2(
        # Inputs
        Q_ref,
        K_ref,
        V_ref,
        D_ref,
        dO_ref,
        ddQ_ref,
        ddK_ref,
        ddV_ref,
        L_ref,
        dD_ref,
        B_ref,
        # Outputs
        dK2_ref,
        dV2_ref,
    ):
        K_j = K_ref[...]
        V_j = V_ref[...]
        ddK_j = ddK_ref[...]
        ddV_j = ddV_ref[...]

        def dk2_dv2_loop_body(qi, carry):
            dV2_j_acc, dK2_j_acc = carry
            qslice = pl.ds(qi * config.tile_q, config.tile_q)
            Q_i = Q_ref[qslice, :]
            L_i = L_ref[qslice]
            dO_i = dO_ref[qslice, :]
            ddQ_i = ddQ_ref[qslice, :]
            dD_i = dD_ref[qslice]
            B_i = B_ref[qslice]
            D_i = D_ref[qslice]
            dD_i = dD_ref[qslice]

            S_ij = pl.dot(Q_i, K_j, trans_b=True) * scale
            P_ij = jnp.exp(S_ij - L_i[:, None])

            dP_ij = pl.dot(dO_i, V_j, trans_b=True)

            ddS_ij = (pl.dot(ddQ_i, K_j, trans_b=True) + pl.dot(Q_i, ddK_j, trans_b=True)) * scale

            dP2_ij = pl.dot(dO_i, ddV_j, trans_b=True) - dP_ij * dD_i[:, None] - ddS_ij * D_i[:, None] + dP_ij * ddS_ij

            dS2_ij = P_ij * (dP2_ij - B_i[:, None]) * scale

            dS_ij = scale * P_ij * (dP_ij - D_i[:, None])

            ddP_ij = P_ij * (ddS_ij - dD_i[:, None])
            dV2_j_acc += pl.dot(ddP_ij.astype(dO_i.dtype), dO_i, trans_a=True)

            dK2_j_acc += pl.dot(dS_ij.astype(ddQ_i.dtype), ddQ_i, trans_a=True)
            dK2_j_acc += pl.dot(dS2_ij.astype(Q_i.dtype), Q_i, trans_a=True)

            return (dV2_j_acc, dK2_j_acc)

        dV2_j, dK2_j = jax.lax.fori_loop(
            0,
            pl.cdiv(n_queries, config.tile_q),
            dk2_dv2_loop_body,
            (jnp.zeros((config.tile_k, hidden_dim), dtype=jnp.float32), jnp.zeros((config.tile_k, hidden_dim), dtype=jnp.float32)),
        )
        dK2_ref[:] = dK2_j.astype(dK2_ref.dtype)
        dV2_ref[:] = dV2_j.astype(dV2_ref.dtype)

    bwd_bwd_stage2 = pl.pallas_call(
        bwd_bwd_kernel_stage2,
        out_shape=(dK2_shape_dtype, dV2_shape_dtype),
        grid=(pl.cdiv(n_keys, config.tile_k), batch_size),
        in_specs=[
            pl.BlockSpec((None, n_queries, hidden_dim), lambda k, b: (b, 0, 0)),  # Q
            pl.BlockSpec((None, config.tile_k, hidden_dim), lambda k, b: (b, k, 0)),  # K
            pl.BlockSpec((None, config.tile_k, hidden_dim), lambda k, b: (b, k, 0)),  # V
            pl.BlockSpec((None, n_queries), lambda k, b: (b, 0)),  # D
            pl.BlockSpec((None, n_queries, hidden_dim), lambda k, b: (b, 0, 0)),  # dO
            pl.BlockSpec((None, n_queries, hidden_dim), lambda k, b: (b, 0, 0)),  # ddQ
            pl.BlockSpec((None, config.tile_k, hidden_dim), lambda k, b: (b, k, 0)),  # ddK
            pl.BlockSpec((None, config.tile_k, hidden_dim), lambda k, b: (b, k, 0)),  # ddV
            pl.BlockSpec((None, n_queries), lambda k, b: (b, 0)),  # L
            pl.BlockSpec((None, n_queries), lambda k, b: (b, 0)),  # dD
            pl.BlockSpec((None, n_queries), lambda k, b: (b, 0)),  # B
        ],
        out_specs=[
            pl.BlockSpec((None, config.tile_k, hidden_dim), lambda k, b: (b, k, 0)),  # dK2
            pl.BlockSpec((None, config.tile_k, hidden_dim), lambda k, b: (b, k, 0)),  # dV2
        ],
    )

    dK2, dV2 = bwd_bwd_stage2(Q, K, V, D, dO, ddQ, ddK, ddV, L, dD, B)

    return dQ2, dK2, dV2, ddO


if __name__ == "__main__":
    batch_size = 16
    n_queries = 256
    hidden_dim = 64
    n_keys = 256
    Q = jrandom.normal(jrandom.PRNGKey(0), (batch_size, n_queries, hidden_dim), dtype=jnp.bfloat16)
    K = jrandom.normal(jrandom.PRNGKey(1), (batch_size, n_keys, hidden_dim), dtype=jnp.bfloat16)
    V = jrandom.normal(jrandom.PRNGKey(2), (batch_size, n_keys, hidden_dim), dtype=jnp.bfloat16)
    O = jrandom.normal(jrandom.PRNGKey(3), (batch_size, n_queries, hidden_dim), dtype=jnp.bfloat16)
    dO = jrandom.normal(jrandom.PRNGKey(4), (batch_size, n_queries, hidden_dim), dtype=jnp.bfloat16)
    ddQ = jrandom.normal(jrandom.PRNGKey(5), (batch_size, n_queries, hidden_dim), dtype=jnp.bfloat16)
    ddK = jrandom.normal(jrandom.PRNGKey(6), (batch_size, n_keys, hidden_dim), dtype=jnp.bfloat16)
    ddV = jrandom.normal(jrandom.PRNGKey(7), (batch_size, n_keys, hidden_dim), dtype=jnp.bfloat16)
    L = jrandom.normal(jrandom.PRNGKey(8), (batch_size, n_queries))
    scale = 1.0 / hidden_dim**0.5

    config = TuningConfig(
        tile_q=64,
        # tile_m=64,
        # tile_n=64,
        tile_k=32,
        max_concurrent_steps=4,
    )
    out = flash_bwdbwd0(Q, K, V, O, dO, ddQ, ddK, ddV, L, scale, config)
    print(out)
    # out = jax.jit(partial(matmul6, config=config))(a, b)
    # print(out)
