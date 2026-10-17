import os
import trimesh
import open3d as o3d

import numpy as np
import pointsastori as pat
import matplotlib.pyplot as plt
import orbax.checkpoint as ocp
import jax.numpy as jnp

from pointsastori.network import FundamentalFormPredictorWithUncertainty, torus_distance_gradient_masked_with_clamp_chunked, torus_precompute_masked_sharded
from pointsastori.shape_3d import PointCloud3D, TriangleMesh, get_neighbors

from typing import *
from flax import nnx
from scipy.spatial import KDTree

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

def load_model(path) -> FundamentalFormPredictorWithUncertainty:
    model = FundamentalFormPredictorWithUncertainty(128, 0, 6, 8, 256, 0, rngs=nnx.Rngs(0))
    graphdef, abstract_state = nnx.split(model)
    checkpointer = ocp.StandardCheckpointer()
    state_restored = checkpointer.restore(path, abstract_state)
    model = nnx.merge(graphdef, state_restored)
    return model


def load_snail() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pointcloud = PointCloud3D.read_PLY("data/snail.ply")
    neighbors = get_neighbors(pointcloud.points, 64)
    return pointcloud.points, pointcloud.normals, neighbors


def load_block() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mesh = TriangleMesh.read_OBJ("data/block.obj")
    neighbors = get_neighbors(mesh.vertices, 64)
    return 0.05*mesh.vertices, mesh.vertex_normals, neighbors


def hsv_to_rgb(hsv):
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    i = jnp.floor(h * 6.0)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i = (i % 6).astype(jnp.int32)
    r = jnp.select([i == 0, i == 1, i == 2, i == 3, i == 4, i == 5], [v, q, p, p, t, v])
    g = jnp.select([i == 0, i == 1, i == 2, i == 3, i == 4, i == 5], [t, v, v, q, p, p])
    b = jnp.select([i == 0, i == 1, i == 2, i == 3, i == 4, i == 5], [p, p, t, v, v, q])
    return jnp.stack([r, g, b], axis=-1)


def raymarch_jax(points: jnp.ndarray,
                 centers: jnp.ndarray,
                 axes: jnp.ndarray,
                 major_radii: jnp.ndarray,
                 minor_radii: jnp.ndarray,
                 neighborhood_radii: jnp.ndarray,
                 uncertainties: jnp.ndarray) -> jnp.ndarray:
    ro = np.array([0, 0, -3])[np.newaxis]
    xy = np.stack(np.meshgrid(np.arange(-200, 200),
                              np.arange(-200, 200), indexing='ij'),
                            axis=-1).reshape((-1, 2)).astype(jnp.float32)
    xy /= 200
    xyz = np.stack((xy[:, 0], xy[:, 1], -2*np.ones((400**2,))), axis=-1)
    rd = xyz - ro
    rd /= np.linalg.norm(rd, axis=-1, keepdims=True)
    t = np.zeros((rd.shape[0], 1))
    d = np.zeros_like(t)

    for _ in range(32):
        print(ro.shape, t.shape, rd.shape)
        p = ro + t*rd

        #print(mask.shape, centers.shape, axes.shape, minor_radii.shape, major_radii.shape)
        d, _, _, = torus_distance_gradient_masked_with_clamp_chunked(
            mask, p[np.newaxis], points, centers,
            axes, major_radii, minor_radii,
            neighborhood_radii, uncertainties,
            n_chunks=16
        )
        #print(d[jnp.newaxis, :, 0])
        #print(t.shape, d.shape)
        t += np.clip(-d[0, :, np.newaxis], -1, 1)

    print(f"d shape t shape {d.shape} {t.shape}")
    hit = np.logical_and(d[0, :, jnp.newaxis] < 1e-3, t < 1e2)

    _, grads, weighted_uncertainties = torus_distance_gradient_masked_with_clamp_chunked(
        mask, p[np.newaxis], points, centers,
        axes, major_radii, minor_radii,
        neighborhood_radii, uncertainties
    )

    normalized_uncertainties = (weighted_uncertainties - weighted_uncertainties.min())/(weighted_uncertainties.max() - weighted_uncertainties.min())

    normals = grads / (1e-10 + jnp.linalg.norm(grads, axis=-1, keepdims=True))
    brightness = jnp.maximum(0.2, normals[..., 1])
    print(normalized_uncertainties.shape, normals.shape, brightness.shape)
    img = hsv_to_rgb(jnp.stack((normalized_uncertainties[0, 0], brightness[0, 0], brightness[0, 0]), axis=-1))
    img = jnp.where(hit, img, jnp.zeros_like(img))
    return img.reshape((400, 400, 3))


def cross_section_occupancy(points: jnp.ndarray,
                            centers: jnp.ndarray,
                            axes: jnp.ndarray,
                            major_radii: jnp.ndarray,
                            minor_radii: jnp.ndarray,
                            neighborhood_radii: jnp.ndarray,
                            uncertainties: jnp.ndarray) -> jnp.ndarray:
    xy = np.stack(np.meshgrid(np.arange(-200, 200),
                              np.arange(-200, 200), indexing='ij'),
                            axis=-1).reshape((-1, 2)).astype(jnp.float32)
    xy /= 200
    xyz = np.stack((xy[:, 0], xy[:, 1], np.zeros((400**2,))), axis=-1)
    print(f"x y z min max {xyz[:, 0].min()} {xyz[:, 0].max()} {xyz[:, 1].min()} {xyz[:, 1].max()} {xyz[:, 2].min()} {xyz[:, 2].max()}")
    #rd = xyz - ro
    #rd /= np.linalg.norm(rd, axis=-1, keepdims=True)
    #t = np.zeros((rd.shape[0], 1))
    #d = np.zeros_like(t)

    #for _ in range(32):
    #    print(ro.shape, t.shape, rd.shape)
    #    p = ro + t*rd

    #print(mask.shape, centers.shape, axes.shape, minor_radii.shape, major_radii.shape)
    print(xyz.shape)
    d, _, _, = torus_distance_gradient_masked_with_clamp_chunked(
        mask, xyz[np.newaxis], points, centers,
        axes, major_radii, minor_radii,
        neighborhood_radii, uncertainties
    )
    print(d.shape)
        #print(d[jnp.newaxis, :, 0])
        #print(t.shape, d.shape)
        #t += np.clip(d[0, 0, :, np.newaxis], -1, 1)

    #print(f"d shape t shape {d.shape} {t.shape}")
    #hit = d[0, 0, :, jnp.newaxis] < 1e-3

    return d.reshape((400, 400))


if __name__ == "__main__":
    print("Loading model . . .")
    model = load_model(os.path.abspath("models/sleipner/UncertaintyTest2/UncertaintyTest2_phase2"))

    print("Loading point cloud . . .")
    points, normals, neighbors = load_snail()
    print(f"x y z min max {points[:, 0].min()} {points[:, 0].max()} {points[:, 0].max()} {points[:, 1].min()} {points[:, 1].max()} {points[:, 2].min()} {points[:, 2].max()}")
    n_points = len(points)
    mask = jnp.ones((1, n_points), dtype=jnp.bool)

    jax_points, jax_normals, jax_neighbors = \
        jnp.asarray(points[neighbors]), jnp.asarray(normals[neighbors]), jnp.asarray(neighbors)

    #print("Computing neighborhood radii . . .")
    point_diffs = jnp.asarray(points[:, np.newaxis]) - jax_points
    #print(f"POINT DIFFS SHAPE {point_diffs.shape}")
    point_dists = jnp.sqrt(jnp.sum(point_diffs**2, axis=-1))
    #print(f"POINT DISTS SHAPE {point_dists.shape}")
    neighborhood_radii = jnp.max(point_dists, axis=-1)[jnp.newaxis]
    #print(f"NEIGHBORHOOD RADII SHAPE {neighborhood_radii.shape}")

    print("Evaluating tori . . .")
    model_out = model(jax_points[jnp.newaxis], jax_normals[jnp.newaxis])
    print(model_out.shape)
    coefficients = model_out[..., :-1]
    print(coefficients.shape)
    uncertainties = model_out[..., -1]
    centers, axes, major_radii, minor_radii = torus_precompute_masked_sharded(
        mask, points[np.newaxis], normals[np.newaxis], coefficients
    )
    print(axes.shape)

    print("Raymarching . . .")
    img = cross_section_occupancy(
        points[np.newaxis],
        centers,
        axes,
        major_radii,
        minor_radii,
        neighborhood_radii,
        uncertainties
    )
    #print(f"IMG DTYPE {img.dtype}")

    img = np.array(img)
    print(np.mean(img))
    #print(img.shape)
    #print(f"IMG DTYPE {img.dtype}")
    fig = plt.figure()
    ax = fig.add_subplot()
    img = ax.imshow(img)
    fig.colorbar(img, ax=ax)
    plt.savefig("raymarch_test20.png")