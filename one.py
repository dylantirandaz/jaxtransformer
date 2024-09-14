import jax
import jax.numpy as jnp
import haiku as hk
import numpy as np
from typing import Optional

class SelfAttention(hk.MultiHeadAttention):
  def __call__(
      self,
      query: jnp.ndarray,
      key: Optional[jnp.narray] = None,
      value: Optional[jnp.ndarray] = None,
      mask: Optional[jnp.ndarray] = None,  
    ) -> jnp.ndarray:
        key = key if key is not None else query
        value = value if value is not None else query

        seq_len = query.shape[1]
        causal_mask = np.tril(np.ones((seq_len, seq_len)))
        mask = mask * causal_mask if mask is not None else causal_mask

        return super().__call__(query, key, value, mask)
  
class DenseBlock(hk.Module): #simple 2 layer MLP
    def __init__(self, init_scale: float, 
        widening_factor: int = 4, name: Optional[str] = None):
        super().__init__(name=name)
        self._init_scale = init_scale
        self._widening_factor = widening_factor

    def _call_(self, x: jnp.ndarray) -> jnp.ndarray:
        hiddens = x.shape[-1]
        initializer = hk.initializers.VarianceScaling(self._init_scale)
        x = hk.initializers.VarianceScaling(self._init_scale)
        x = jax.lnn.gelu(x)
        return hk.linear(hiddens, w_init=initializer)(x)
    
    def layer_norm(x: jnp.ndarray, 
            name: Optional[str] = None) -> jnp.ndarray:
        return hk.LayerNorm(axis=-1, create_scale=True,
                create_offset=True, name=name)(x)


class Transformer(hk.Module):

    def __init__(self, num_heads: int, num_layers: int, 
        dropout_rate: float, name: Optional[str] = None):
        super().__init__(name=name)
        self._num_layers = num_layers
        self._num_heads = num_heads
        self._dropout_rate = dropout_rate

    ### def __call__(self, h: jnp.ndarray, 
        # mask: Optional[jnp.ndarray],
        # is_training: bool) -> jnp.ndarray:
  