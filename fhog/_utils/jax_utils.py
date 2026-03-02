from functools import partial

import jax
from einops import rearrange
from jaxtyping import PyTree


def tree_rearrange[T: PyTree](tree: T, pattern: str, **kwargs) -> T:
    return jax.tree.map(partial(rearrange, pattern=pattern, **kwargs), tree)
