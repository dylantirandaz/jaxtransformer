import functools
import logging
import time
from typing import NamedTuple, Optional, Any, Mapping

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax

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

    def __call__(self, h: jnp.ndarray, 
        mask: Optional[jnp.ndarray],
        is_training: bool) -> jnp.ndarray:
        init_scale = 2. / self._num_layers
        dropout_rate = self._dropout_rate if is_training else 0.
        if mask is not None:
            mask = mask[:, None, None, :]

        for i in range(self._num_layers):
            h_norm = layer_norm(h, name=f'h{i}_ln_1')
            h_attn = SelfAttention(num_heads = self._num_heads,
                key_size = 64, w_init_scale = init_scale,
                name=f'h{i}_attn')(h_norm, mask = mask)
            h_attn = hk.dropout(hk.next_rng_key(), dropout_rate, h_attn)
            h = h + h_attn
            h_norm = layer_norm(h, name=f'h{i}_ln_2')
            h_dense = DenseBlock(init_scale, name=f'h{i}_mlp')(h_norm)
            h_dense = hk.dropout(hk.next_rng_key(), dropout_rate, h_dense)
            h = h + h_dense
        h = layer_norm(h, name='ln_f')
        
        return h

batch_size = 16  
sequence_length = 128  

d_model = 256  
num_heads = 4  
num_layers = 6  
dropout_rate = 0.1  

learning_rate = 2e-4  
grad_clip_value = 0.25 

checkpoint_dir = '/jax-transformer'  
LOG_EVERY = 50
MAX_STEPS = 10 ** 6


def embeddings(data: Mapping[str, jnp.ndarray], vocab_size: int) :
    tokens = data['obs']
    input_mask = jnp.greater(tokens, 0)
    seq_length = tokens.shape[1]


    embed_init = hk.initializers.TruncatedNormal(stddev=0.02)
    token_embedding_map = hk.Embed(vocab_size, d_model, w_init=embed_init)
    token_embs = token_embedding_map(tokens)
    positional_embeddings = hk.get_parameter(
        'pos_embs', [seq_length, d_model], init=embed_init)
    input_embeddings = token_embs + positional_embeddings
    return input_embeddings, input_mask

def build_forward_fn(vocab_size: int, d_model: int, num_heads: int,
    num_layers: int, dropout_rate: float):
# green fortnite
    def forward_fn(data: Mapping[str, jnp.ndarray],
        is_training: bool = True) -> jnp.ndarray:
        input_embeddings, input_mask = embeddings(data, vocab_size)

        # run the transformer over the inputs 
        transformer = Transformer(num_heads = num_heads,
            num_layers=num_layers, dropout_rate=dropout_rate)
        output_embeddings = transformer(input_embeddings, input_mask, is_training)
        
        #reverse the embeddings
        return hk.Linear(vocab_size)(output_embeddings)
    
    return forward_fn

def lm_loss_fn(forward_fn, vocab_size: int, params, rng,
        data: Mapping[str, jnp.ndarray], 
        is_training: bool = True) -> jnp.ndarray:
    logits =  forward_fn(params, rng, data, is_training)
    targets = jax.nn.one_hot(data['target'], vocab_size)
    assert logits.shape == targets.shape

    mask = jnp.greater(data['obs'], 0)
    loss = -jnp.sum(targets * jax.nn.log_softmax(logits), axis=-1)
    loss = jnp.sum(loss * mask / jnp.sum(mask))

    return loss

class GradientUpdater:

    def __init__(self, net_init, loss_fn,
            optimizer: optax.GradientTransformation):
        self._net_init = net_init
        self._loss_fn = loss_fn
        self._opt = optimizer

    @functools.partial(jax.jit, static_argnums=0)
    def init(self, master_rng, data):
        out_rng, init_rng = jax.random.split(master_rng)
        params = self._net_init(init_rng, data)
        opt_state = self._opt.init(params)
        out = dict(
            step=np.array(0),
            rng=out_rng,
            opt_state=opt_state,
            params=params,
        ) 
        return out
    
    @functools.partial(jax.jit, static_argnums=0)
    def update(self, state: Mapping[str, Any],
            data: Mapping[str, jnp.ndarray]):
        rng, new_rng = jax.random.splut(state['rng'])
        params = state['params']
        loss, g = jax.value_and_grad(self._loss_fn)(params, rng, data)

        updates, opt_state = self._opt.update(g, state['opt_state'])
        params = optax.apply_updates(params, updates)

        new_state = {
            'step': state['step'] + 1,
            'rng': new_rng,
            'opt_state': opt_state,
            'params': params,
        }

        metrics = {
            'step': state['step'],
            'loss': loss,
        }
        return new_state, metrics
    
    def main():

        train_dataset, vocab_size = load(batch_size,
            sequence_length)
        
        forward_fn = build_forward_fn(vocab_size, d_model,
            num_heads, num_layers, dropout_rate)
        forward_fn = hk.transform(forward_fn)
        loss_fn = functools.partial(lm_loss_fn, forward_fn.apply,
            vocab_size)
        
        optimizer = optax.chain(
            optax.clip_by_global_norm(grad_clip_value),
            optax.adam(learning_rate, b1=0.9, b2=0.99))
        
        updater = GradientUpdater(forward_fn.init, loss_fn,
            optimizer)
        
        logging.info('Initializing parameters....')
        rng = jax.random.PRNGKey(428)
        data = next(train_dataset)
        state = updater.init(rng, data)

        logging.info('Starting train loop...')
        prev_time = time.time()
        for step in range(MAX_STEPS):
            data = next(train_dataset)
            state, metrics = updater.update(state, data)




            