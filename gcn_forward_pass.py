import jax
import jax.numpy as jnp

from gcn import GCN

model = GCN([1,5,5,2], jax.random.PRNGKey(42))

A = jnp.array([
    [0, 1, 1, 0],
    [1, 0, 1, 1],
    [1, 1, 0, 1],
    [0, 1, 1, 0],
    ])

D = jnp.array([2, 3, 3, 3])

x = jnp.array([[0.4],[0.1],[0.3],[0.4]])

z = model(x, A, D)
