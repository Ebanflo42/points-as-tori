import numpy as np
import jax.numpy as jnp

from typing import *
from jax import vmap
from joblib import Parallel, delayed


def find_covariance_eigendata(neighborhoods: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """neighborhoods should have shape [batch, k, 3]
    """
    covariance = jnp.einsum('bkt,bku->btu', neighborhoods, neighborhoods)
    eigvals, eigvecs = jnp.linalg.eigh(covariance)
    return eigvals, eigvecs


def orthonormal_basis(n, rotation):
    # n: [..., 3]
    n_hat = n / np.linalg.norm(n, axis=-1, keepdims=True)
    u = np.where(np.abs(n_hat[..., 0:1]) > 0.9,
                  np.array([0., 1., 0.]),
                  np.array([1., 0., 0.]))
    # Rodrigues rotation of u about n_hat by `rotation`
    c, s = np.cos(rotation), np.sin(rotation)
    u_rot = (u * c
             + np.cross(n_hat, u) * s
             + n_hat * np.sum(n_hat * u, axis=-1, keepdims=True) * (1 - c))
    s_vec = np.cross(n_hat, u_rot)
    s_vec = s_vec / np.linalg.norm(s_vec, axis=-1, keepdims=True)
    t_vec = np.cross(n_hat, s_vec)
    return n_hat[0], s_vec[0], t_vec[0]


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

"""
def least_squares_coefficients(neighborhoods, normals, rotation=0.0):
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
"""

def single_least_squares_coefficients(neighborhoods: np.ndarray, normals: np.ndarray, rotation=0.0):
    """neighborhoods: [k, 3], normals: [k, 3]. Center at point 0, keep constant term."""
    indices = np.where(np.sum(normals*normals[0:1], axis=-1) > 0.5)
    neighborhoods = neighborhoods[indices]
    normals = normals[indices]
    #print(neighborhoods.shape)

    origin = neighborhoods[0:1]
    local = neighborhoods - origin

    n0 = normals[0]
    n_hat, s, t = orthonormal_basis(n0[np.newaxis], rotation)

    h = np.sum(local * n_hat[np.newaxis], axis=-1)
    y = np.sum(local * s[np.newaxis], axis=-1)
    z = np.sum(local * t[np.newaxis], axis=-1)

    X = np.stack([np.ones_like(y), y, z, y*z, y*y, z*z], axis=-1)
    coeffs, _, _, _ = np.linalg.lstsq(X, h)
    return coeffs


def least_squares_coefficients(neighborhoods: np.ndarray, normals: np.ndarray):
    """neighborhoods: [b, k, 3], normals: [b, k, 3]. Center at point 0, keep constant term."""
    all_coeffs = Parallel(n_jobs=6)(delayed(single_least_squares_coefficients)(ne, no) for ne, no in zip(neighborhoods, normals))
    #all_coeffs = [single_least_squares_coefficients(ne, no) for ne, no in zip(neighborhoods, normals)]
    return np.stack(all_coeffs, axis=0)