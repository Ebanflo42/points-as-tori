import numpy as np
import jax.numpy as jnp

from typing import *

from pointsastori.shape_3d import PointCloud3D, TriangleMesh, get_neighbors


def load_snail() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pointcloud = PointCloud3D.read_PLY("data/snail.ply")
    neighbors = get_neighbors(pointcloud.points, 64)
    return pointcloud.points, pointcloud.normals, neighbors


def load_block() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mesh = TriangleMesh.read_OBJ("data/block.obj")
    neighbors = get_neighbors(mesh.vertices, 64)
    return 0.05*mesh.vertices, mesh.vertex_normals, neighbors, mesh.faces


def load_femur() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mesh = TriangleMesh.read_OBJ("data/femur.obj")
    neighbors = get_neighbors(mesh.vertices, 64)
    verts = 0.1*mesh.vertices
    verts *= 1.4
    verts -= 0.41
    return verts, mesh.vertex_normals, neighbors, mesh.faces

def load_knot() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mesh = TriangleMesh.read_OBJ("data/knot.obj")
    neighbors = get_neighbors(mesh.vertices, 64)
    return mesh.vertices, mesh.vertex_normals, neighbors, mesh.faces


def load_tooth() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pointcloud = PointCloud3D.read_PLY("data/tooth.ply")
    neighbors = get_neighbors(pointcloud.points, 64)
    return pointcloud.points, pointcloud.normals, neighbors


def load_bunny() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pointcloud = PointCloud3D.read_PLY("data/holey-bunny.ply")
    neighbors = get_neighbors(pointcloud.points, 64)
    return pointcloud.points, pointcloud.normals, neighbors


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

