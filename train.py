import os
from pathlib import Path
import time
import pickle
import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state, checkpoints
from flax import serialization
import orbax.checkpoint as orbax
import tiktoken
from flax.training import orbax_utils
from functools import partial
from model import GPTConfig, GPT
from utils import print_compiling

# Configuration
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False
always_save_checkpoint = True
init_from = 'scratch'
dataset = 'shakespeare'
batch_size = 12
block_size = 1024
n_layer, n_head, n_embd = 12, 12, 768
dropout = 0.0
learning_rate = 6e-4
max_iters = 600000
weight_decay = 1e-2
beta1, beta2 = 0.9, 0.95
decay_lr, warmup_iters = True, 2000
lr_decay_iters, min_lr = 600000, 6e-5
device, dtype = 'cuda', 'bfloat16'
compile = True
max_new_tokens, temperature, top_k = 100, 0.8, 200

# Load dataset
data_dir = os.path.join('data', dataset)
train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = np.random.randint(len(data) - block_size, size=(batch_size,))
    x = np.stack([data[i:i+block_size].astype(np.int32) for i in ix])
    y = np.stack([data[i+1:i+1+block_size].astype(np.int32) for i in ix])
    return x, y

# Model Initialization
iter_num, best_val_loss = 0, 1e9
meta_path = os.path.join(data_dir, 'meta.pkl')
vocab_size = 50257
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        vocab_size = pickle.load(f)['vocab_size']
    print(f"vocab_size = {vocab_size}")

model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size, dropout=dropout, vocab_size=vocab_size)
if init_from == 'scratch':
    model = GPT(GPTConfig(**model_args))
    state = model.create_state()
elif init_from == 'resume':
    latest_step = orbax.CheckpointManager(Path(out_dir, 'checkpoint')).latest_step()
    assert latest_step is not None, "No checkpoint found"
    checkpoint = orbax.CheckpointManager.restore(latest_step, items=None)
    model = GPT(GPTConfig(**checkpoint['model_args']))
    state = serialization.from_state_dict(jax.eval_shape(lambda: model.create_state()), checkpoint['state'])
    iter_num, best_val_loss = checkpoint['iter_num'] + 1, checkpoint['val_loss']
else:
    raise RuntimeError(f"init_from={init_from} not supported")

# Training functions
@partial(jax.jit, static_argnames=('train',))
@print_compiling
def forward(state, batch, train: bool):
    inputs, labels = batch
    rngs = {'dropout': jax.random.PRNGKey(0)} if train and dropout > 0.0 else {}
    return state.apply_fn({'params': state.params}, inputs, train=train, targets=labels, rngs=rngs)

@partial(jax.jit, donate_argnums=(0,))
@print_compiling
def train_step(state: train_state.TrainState, batch):
    loss_fn = lambda params: forward(state.replace(params=params), batch, train=True)[1]
    loss, grad = jax.value_and_grad(loss_fn)(state.params)
    return loss, state.apply_gradients(grads=grad)

def estimate_loss():
    return {split: np.mean([float(forward(state, get_batch(split), train=False)[1]) for _ in range(eval_iters)]) for split in ['train', 'val']}

@jax.jit
@print_compiling
def _sample(params, key, tokens):
    return model.generate(key, params, tokens, max_new_tokens=max_new_tokens, top_k=top_k, temperature=temperature)

tokenizer = tiktoken.get_encoding("gpt2")

def sample(params, key, tokens):
    return tokenizer.decode(_sample(params, key, tokens)[0])

# Training loop
t0 = time.time()
while iter_num <= max_iters:
    if iter_num % eval_interval == 0:
        print("Evaluating...")
        print(f"Sample: {sample(state.params, jax.random.PRNGKey(0), val_batch[0][0:1,:5])}")
        losses = estimate_loss()
        print(f"Step {iter_num}: Train Loss {losses['train']:.4f}, Val Loss {losses['val']:.4f}")
        if always_save_checkpoint:
            print(f"Saving checkpoint to {out_dir}")
            orbax.CheckpointManager(Path(out_dir, 'checkpoint')).save(step=iter_num, items={'state': state, 'model_args': model_args, 'iter_num': iter_num, 'val_loss': losses['val']})
    if iter_num == 0 and eval_only:
        break
    loss, state = train_step(state, get_batch('train'))
    if iter_num % log_interval == 0:
        print(f"Iter {iter_num}: Loss {loss:.4f}, Time {(time.time() - t0) * 1000:.2f}ms")
    iter_num += 1