import jax
import jax.numpy as jnp

import equinox as eqx

class GCN(eqx.Module):
    num_layers: int
    W_list: list
    B_list: list

    def __init__(self, layers, key):
        """
        Inputs:
            layers is a python list indicating the size of the node embeddings at each layer
            key is used to generate random numbers for initialising the W and B matrices
        """
        
        self.num_layers = len(layers)
        self.W_list = []
        self.B_list = []

        for i in range(self.num_layers-1):
            weights_key, bias_key, key = jax.random.split(key, num=3)
            W = jax.random.normal(weights_key, (layers[i], layers[i+1]))
            B = jax.random.normal(bias_key, (layers[i], layers[i+1]))

            self.W_list.append(W)
            self.B_list.append(B)

    def __call__(self, z, adj_mat, degree):
        """
        Inputs:
            z is a jnp array for which the i-th row is the i-th node embedding
            adj_mat is the adjacency matrix. Ideally it should be a sparse matrix
            degree is a jnp array where the i-th element is the degree of the i-th node

        Output:
            Similar to z. The node embeddings of the output
        """

        for i in range(self.num_layers-1):
            z = jnp.tanh(jnp.diagflat(1.0/degree) @ adj_mat @ z @ self.W_list[i] + z @ self.B_list[i])
        return z


