import jax.numpy as jnp

from typing import *


def find_covariance_eigendata(neighborhoods: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """neighborhoods should have shape [batch, k, 3]
    """
    covariance = jnp.einsum('bkt,bku->btu', neighborhoods, neighborhoods)
    eigvals, eigvecs = jnp.linalg.eigh(covariance)
    return eigvals, eigvecs


def least_squares_coefficients(neighborhoods: jnp.ndarray) -> jnp.ndarray:
    mu = jnp.mean(neighborhoods, axis=1, keepdims=True)
    neighborhoods = neighborhoods - mu
    eigvals, eigvecs = find_covariance_eigendata(neighborhoods)
    # eigvecs might need to be transposed?
    neighborhoods = jnp.einsum('but,bku->bkt', eigvecs, neighborhoods)

    X = jnp.stack([jnp.ones_like(neighborhoods[..., 0]),
                   neighborhoods[..., 1],
                   neighborhoods[..., 2],
                   neighborhoods[..., 1]*neighborhoods[..., 2],
                   neighborhoods[..., 1]**2,
                   neighborhoods[..., 2]**2], axis=-1)
    XTX = jnp.einsum('bks,bkt->bst', X, X)
    coeffs = jnp.einsum('bst,bkt,bk->bs', jnp.linalg.inv(XTX), X, neighborhoods[..., 0])

    return coeffs