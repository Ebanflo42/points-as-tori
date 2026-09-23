import jax.numpy as jnp

from typing import *
from jax import vmap


def find_covariance_eigendata(neighborhoods: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """neighborhoods should have shape [batch, k, 3]
    """
    covariance = jnp.einsum('bkt,bku->btu', neighborhoods, neighborhoods)
    eigvals, eigvecs = jnp.linalg.eigh(covariance)
    return eigvals, eigvecs


def orthonormal_basis(n, rotation):
    # n: [..., 3]
    n_hat = n / jnp.linalg.norm(n, axis=-1, keepdims=True)
    u = jnp.where(jnp.abs(n_hat[..., 0:1]) > 0.9,
                  jnp.array([0., 1., 0.]),
                  jnp.array([1., 0., 0.]))
    # Rodrigues rotation of u about n_hat by `rotation`
    c, s = jnp.cos(rotation), jnp.sin(rotation)
    u_rot = (u * c
             + jnp.cross(n_hat, u) * s
             + n_hat * jnp.sum(n_hat * u, axis=-1, keepdims=True) * (1 - c))
    s_vec = jnp.cross(n_hat, u_rot)
    s_vec = s_vec / jnp.linalg.norm(s_vec, axis=-1, keepdims=True)
    t_vec = jnp.cross(n_hat, s_vec)
    return s_vec, t_vec


"""
def least_squares_coefficients(neighborhoods: jnp.ndarray, normals: jnp.ndarray) -> jnp.ndarray:
    mu = jnp.mean(neighborhoods, axis=1, keepdims=True)
    neighborhoods = neighborhoods - mu
    eigvals, eigvecs = find_covariance_eigendata(neighborhoods)
    neighborhoods = jnp.einsum('but,bku->bkt', eigvecs, neighborhoods)

    # check if we got the right normal orientation
    normals_agree = jnp.where(jnp.sum(eigvecs[:, :, 0] * normals[:, 0], axis=-1) > 0, 1, -1)
    eigvecs_reflect = jnp.where(jnp.linalg.det(eigvecs) > 0, 1, -1)
    # flip neighborhoods where they are facing the wrong direction
    neighborhoods = neighborhoods.at[:, :, 0].mul(normals_agree[:, jnp.newaxis])
    # need to flip a tangent axis if there was a parity flip while transforming to eigenvector frame
    neighborhoods = neighborhoods.at[:, :, 1].mul(normals_agree[:, jnp.newaxis]*eigvecs_reflect[:, jnp.newaxis])

    X = jnp.stack([neighborhoods[..., 1],
                   neighborhoods[..., 2],
                   neighborhoods[..., 1]*neighborhoods[..., 2],
                   neighborhoods[..., 1]**2,
                   neighborhoods[..., 2]**2], axis=-1)
    #XTX = jnp.einsum('bks,bkt->bst', X, X)
    #print(neighborhoods.shape, print(X.shape), print(XTX.shape))
    #coeffs = jnp.einsum('bst,bkt,bk->bs', jnp.linalg.inv(XTX), X, neighborhoods[..., 0])
    coeffs = vmap(lambda A, b: jnp.linalg.lstsq(A, b)[0])(X, neighborhoods[...,0])
    coeffs = jnp.concatenate((jnp.zeros((coeffs.shape[0], 1)), coeffs), axis=-1)

    return coeffs
"""

def least_squares_coefficients(neighborhoods, normals, rotation=0.0):
    """neighborhoods: [b, k, 3], normals: [b, k, 3]. Center at point 0, keep constant term."""
    origin = neighborhoods[:, 0:1]                       # [b, 1, 3]
    local = neighborhoods - origin                       # [b, k, 3]

    n0 = normals[:, 0]                                   # [b, 3]
    s, t = orthonormal_basis(n0, rotation)               # [b, 3] each
    n_hat = n0 / jnp.linalg.norm(n0, axis=-1, keepdims=True)

    h = jnp.sum(local * n_hat[:, None], axis=-1)         # [b, k]
    y = jnp.sum(local * s[:, None], axis=-1)             # [b, k]
    z = jnp.sum(local * t[:, None], axis=-1)             # [b, k]

    X = jnp.stack([jnp.ones_like(y), y, z, y*z, y*y, z*z], axis=-1)   # [b, k, 6]
    coeffs = vmap(lambda A, b: jnp.linalg.lstsq(A, b)[0])(X, h)       # [b, 6]
    return coeffs