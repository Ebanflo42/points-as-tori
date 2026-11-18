import os
import trimesh
import open3d as o3d

import jax
from jax import jit
import numpy as np
import pointsastori as pat
import matplotlib.pyplot as plt
import orbax.checkpoint as ocp
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from time import time
import jax.numpy as jnp

from matplotlib.colors import ListedColormap
from matplotlib.cm import ScalarMappable

from pointsastori.network import FundamentalFormPredictorWithUncertainty, torus_distance_gradient_masked_with_clamp, torus_precompute_masked_sharded
from pointsastori.shape_3d import PointCloud3D, TriangleMesh, get_neighbors
from jax_raymarch import *

from typing import *
from tqdm import tqdm
from flax import nnx
from scipy.spatial import KDTree
from skimage.measure import marching_cubes


EPS = 1e-10


@jit
def closest_point_on_triangle(
    query: jax.Array,  # (3,) or (..., 3)
    v0: jax.Array,  # (3,)
    v1: jax.Array,  # (3,)
    v2: jax.Array,  # (3,)
) -> jax.Array:
    """
    Find closest point on triangle to query point.

    Args:
        query: Query point(s)
        v0, v1, v2: Triangle vertices

    Returns:
        Closest point on triangle (same shape as query)
    """
    # Edges
    e0 = v1 - v0
    e1 = v2 - v1
    e2 = v0 - v2

    # Face normal
    n = jnp.cross(e0, -e2)
    n_len = jnp.linalg.norm(n)

    # Handle degenerate triangle
    n_normalized = jnp.where(n_len < EPS, jnp.array([1.0, 0.0, 0.0]), n / n_len)

    # Project query onto plane
    v0_to_query = query - v0
    dist_to_plane = jnp.dot(v0_to_query, n_normalized)
    proj = query - dist_to_plane * n_normalized

    # Check if projection is inside triangle using barycentric sign test
    c0 = jnp.cross(v1 - v0, proj - v0)
    c1 = jnp.cross(v2 - v1, proj - v1)
    c2 = jnp.cross(v0 - v2, proj - v2)

    inside = (
        (jnp.dot(c0, n_normalized) >= 0.0) & (jnp.dot(c1, n_normalized) >= 0.0) & (jnp.dot(c2, n_normalized) >= 0.0)
    )

    # If inside, return projection
    # Otherwise, find closest point on edges

    def closest_on_edge(va: jax.Array, vb: jax.Array) -> Tuple[jax.Array, jax.Array]:
        """Find closest point on edge and its squared distance."""
        edge = vb - va
        edge_len_sq = jnp.dot(edge, edge)
        t = jnp.dot(query - va, edge) / jnp.maximum(edge_len_sq, EPS)
        t = jnp.clip(t, 0.0, 1.0)
        pt = va + t * edge
        dist_sq = jnp.sum((query - pt) ** 2)
        return pt, dist_sq

    # Check all three edges
    pt0, dist_sq0 = closest_on_edge(v0, v1)
    pt1, dist_sq1 = closest_on_edge(v1, v2)
    pt2, dist_sq2 = closest_on_edge(v2, v0)

    # Find minimum
    min_dist_sq = jnp.minimum(jnp.minimum(dist_sq0, dist_sq1), dist_sq2)

    # Select closest edge point
    edge_closest = jnp.where(dist_sq0 == min_dist_sq, pt0, jnp.where(dist_sq1 == min_dist_sq, pt1, pt2))

    # Return projection if inside, otherwise closest edge point
    return jnp.where(inside, proj, edge_closest)


@jit
def unsigned_distance_to_triangle(
    query: jax.Array,  # (..., 3)
    v0: jax.Array,  # (3,)
    v1: jax.Array,  # (3,)
    v2: jax.Array,  # (3,)
) -> jax.Array:
    """
    Compute unsigned distance from query point(s) to triangle.

    Args:
        query: Query point(s) of shape (..., 3)
        v0, v1, v2: Triangle vertices

    Returns:
        Unsigned distance(s) of shape (...)
    """
    closest_pt = closest_point_on_triangle(query, v0, v1, v2)
    return jnp.linalg.norm(query - closest_pt, axis=-1)


@jit
def gwn_single_face(
    query: jax.Array,  # (3,)
    v0: jax.Array,  # (3,)
    v1: jax.Array,  # (3,)
    v2: jax.Array,  # (3,)
) -> jax.Array:
    """
    Compute solid angle contribution of one triangle to generalized winding number.

    This is a direct JAX port of the GWN computation in shape_3d.cpp.

    Args:
        query: Query point
        v0, v1, v2: Triangle vertices

    Returns:
        Solid angle contribution (scalar)
    """
    # Vectors from query to vertices
    a_vec = v0 - query
    b_vec = v1 - query
    c_vec = v2 - query

    # Lengths
    a = jnp.linalg.norm(a_vec)
    b = jnp.linalg.norm(b_vec)
    c = jnp.linalg.norm(c_vec)

    # Solid angle formula from C++:
    # numer = -a_z * b_y * c_x + a_y * b_z * c_x + a_z * b_x * c_y
    # - a_x * b_z * c_y - a_y * b_x * c_z + a_x * b_y * c_z
    # This is the determinant det([a, b, c])
    numer = (
        -a_vec[2] * b_vec[1] * c_vec[0]
        + a_vec[1] * b_vec[2] * c_vec[0]
        + a_vec[2] * b_vec[0] * c_vec[1]
        - a_vec[0] * b_vec[2] * c_vec[1]
        - a_vec[1] * b_vec[0] * c_vec[2]
        + a_vec[0] * b_vec[1] * c_vec[2]
    )

    denom = a * b * c + jnp.dot(a_vec, b_vec) * c + jnp.dot(b_vec, c_vec) * a + jnp.dot(c_vec, a_vec) * b

    return 2.0 * jnp.arctan2(numer, denom)


@jit
def unsigned_distance_to_mesh(
    queries: jax.Array,  # (n_queries, 3)
    vertices: jax.Array,  # (max_vertices, 3)
    faces: jax.Array,  # (max_faces, 3)
    face_mask: jax.Array,  # (max_faces,)
) -> jax.Array:
    """
    Compute unsigned distance from query points to mesh (brute force).

    Args:
        queries: Query points
        vertices: Mesh vertices (padded)
        faces: Mesh faces (padded)
        face_mask: Boolean mask for valid faces

    Returns:
        distances: (n_queries,) unsigned distances to mesh
    """
    n_queries = queries.shape[0]
    max_faces = faces.shape[0]
    print(f"Queries vertices faces shape: {queries.shape} {vertices.shape} {faces.shape}")

    # Triangle vertices
    safe_faces = jnp.maximum(faces, 0)
    v0 = vertices[safe_faces[:, 0]]  # (max_faces, 3)
    v1 = vertices[safe_faces[:, 1]]
    v2 = vertices[safe_faces[:, 2]]

    # Vectorized distance computation: (n_queries, max_faces)
    # Use vmap to compute distance from each query to each triangle
    def dist_to_all_triangles(q):
        # q: (3,) -> returns (max_faces,)
        return jax.vmap(lambda v0, v1, v2: unsigned_distance_to_triangle(q, v0, v1, v2))(v0, v1, v2)

    distances_all = jax.vmap(dist_to_all_triangles)(queries)  # (n_queries, max_faces)

    # Mask invalid faces with inf
    distances_masked = jnp.where(face_mask[None, :], distances_all, jnp.inf)

    # Take minimum over faces
    return jnp.min(distances_masked, axis=1)  # (n_queries,)


@jit
def gwn_mesh(
    queries: jax.Array,  # (n_queries, 3)
    vertices: jax.Array,  # (max_vertices, 3)
    faces: jax.Array,  # (max_faces, 3)
    face_mask: jax.Array,  # (max_faces,)
) -> jax.Array:
    """
    Compute generalized winding number for query points.

    GWN is the sum of solid angles subtended by all faces.

    Args:
        queries: Query points
        vertices: Mesh vertices (padded)
        faces: Mesh faces (padded)
        face_mask: Boolean mask for valid faces

    Returns:
        gwn: (n_queries,) generalized winding numbers (absolute value)
    """
    # Extract triangle vertices
    safe_faces = jnp.maximum(faces, 0)
    v0 = vertices[safe_faces[:, 0]]  # (max_faces, 3)
    v1 = vertices[safe_faces[:, 1]]
    v2 = vertices[safe_faces[:, 2]]

    # Compute solid angle contribution for each query-face pair
    def solid_angles_for_query(q):
        # q: (3,) -> returns (max_faces,)
        angles = jax.vmap(lambda v0, v1, v2: gwn_single_face(q, v0, v1, v2))(v0, v1, v2)
        # Mask invalid faces
        return jnp.where(face_mask, angles, 0.0)

    # Sum over all faces for each query
    solid_angles = jax.vmap(solid_angles_for_query)(queries)  # (n_queries, max_faces)
    gwn = jnp.sum(solid_angles, axis=1)  # (n_queries,)

    # Return absolute value for sign test
    return gwn


@jit
def signed_distance_to_mesh(
    queries: jax.Array,  # (n_queries, 3)
    vertices: jax.Array,  # (max_vertices, 3)
    faces: jax.Array,  # (max_faces, 3)
    face_mask: jax.Array,  # (max_faces,)
) -> jax.Array:
    """
    Compute signed distance from query points to mesh.

    Uses unsigned distance, and GWN for sign determination.
    Negative inside, positive outside.

    Args:
        queries: Query points
        vertices: Mesh vertices (padded)
        faces: Mesh faces (padded)
        face_mask: Boolean mask for valid faces

    Returns:
        signed_distances: (n_queries,)
    """
    # Unsigned distance
    unsigned_dist = unsigned_distance_to_mesh(queries, vertices, faces, face_mask)

    # Sign from GWN
    gwn = gwn_mesh(queries, vertices, faces, face_mask)
    sign = jnp.where(gwn > jnp.pi, -1.0, 1.0)

    return sign * unsigned_dist


def torus_marching_cubes(points: jnp.ndarray,
                         centers: jnp.ndarray,
                         axes: jnp.ndarray,
                         major_radii: jnp.ndarray,
                         minor_radii: jnp.ndarray,
                         neighborhood_radii: jnp.ndarray,
                         uncertainties: jnp.ndarray,
                         faces=None
):

    grid = np.stack(np.meshgrid(
        np.linspace(-1, 1, num=64),
        np.linspace(-1, 1, num=64),
        np.linspace(-1, 1, num=64),
    ), axis=-1).reshape((1, -1, 3)).astype(np.float32)

    d, grads, _ = torus_distance_gradient_masked_with_clamp_chunked(
        jnp.ones(points.shape[:2], dtype=jnp.bool),
        grid,
        points,
        centers,
        axes,
        major_radii,
        minor_radii,
        neighborhood_radii,
        uncertainties,
        n_chunks=16
    )

    norm_grads = jnp.linalg.norm(grads, axis=-1)
    eikonality_error = jnp.mean(jnp.abs(1 - norm_grads))
    print(f"Eikonality error: {eikonality_error}")

    d = d.reshape((64, 64, 64))

    if faces is not None:
        true_ds = []
        for i in range(64):
            subgrid = grid[0, 64*64*i : 64*64*(i + 1)]
            arr = signed_distance_to_mesh(
                subgrid,
                points[0],
                faces,
                jnp.ones((len(faces),), dtype=jnp.bool)
            )
            arr = np.array(arr.block_until_ready())
            true_ds.append(arr)
        true_d = np.stack(true_ds, axis=0).reshape((64, 64, 64))
        print(f"L1 distance error: {np.mean(np.abs(d - true_d))}")

    verts, faces, _, _ = marching_cubes(d, level=0.0, spacing=(2/63, 2/63, 2/63))
    # offset the origin of the voxel grid (marching_cubes assumes positive coordinates)
    verts -= 1
    # for some reason, x and y coordinates of the vertices get permuted at some point, so fix that
    verts = np.stack((verts[:, 1], verts[:, 0], verts[:, 2]), axis=-1)
    print(f"VERTS MIN MAX X Y Z {verts[:, 0].min()} {verts[:, 0].max()} {verts[:, 1].min()} {verts[:, 1].max()} {verts[:, 2].min()} {verts[:, 2].max()}")
    print(verts.shape)

    _, _, uncertainties = torus_distance_gradient_masked_with_clamp_chunked(
        jnp.ones(points.shape[:2], dtype=jnp.bool),
        verts[np.newaxis],
        points,
        centers,
        axes,
        major_radii,
        minor_radii,
        neighborhood_radii,
        uncertainties,
        n_chunks=1
    )

    uncertainties = uncertainties[0]
    uncertainties = (uncertainties - uncertainties.min())/(uncertainties.max() - uncertainties.min())
    hue = 0.7 - 0.7*uncertainties
    colors = hsv_to_rgb(jnp.stack((
        hue, np.ones_like(hue), np.ones_like(hue)
    ), axis=-1))
    colors = (255*colors).astype(np.uint8)

    return verts, faces, colors


def render_to_html(verts, faces, colors, fname):
    # --- Create side-by-side 3D scenes ---
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'scene'}, {'type': 'scene'}]],
        subplot_titles=('Point Cloud', 'Mesh')
    )

    # Point cloud in scene 1
    fig.add_trace(
        go.Scatter3d(
            x=points[:, 0], y=points[:, 1], z=points[:, 2],
            mode='markers',
            marker=dict(size=1, color='blue')
        ),
        row=1, col=1
    )

    # Mesh in scene 2
    fig.add_trace(
        go.Mesh3d(
            x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            vertexcolor=colors, opacity=1.0
        ),
        row=1, col=2
    )

    fig.update_layout(
        height=900,
        width=1800
    )
    fig.write_html(fname)


if __name__ == "__main__":
    print("Loading model . . .")
    model = load_model(os.path.abspath(
        "/media/ncs-tmp/Eben/torus-project/checkpoints/sleipner/UncertaintyTrial2/UncertaintyTrial2_phase3"
    ))
    param_tree = nnx.state(model, nnx.Param)
    param_count = sum(p.size for p in jax.tree_util.tree_leaves(param_tree))
    print(f"Paramter count: {param_count}")

    print("Loading point cloud . . .")
    points, normals, neighbors, faces = load_block()
    print(f"x y z min max {points[:, 0].min()} {points[:, 0].max()} {points[:, 1].min()} {points[:, 1].max()} {points[:, 2].min()} {points[:, 2].max()}")
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
    t0 = time()
    model_out = model(jax_points[jnp.newaxis], jax_normals[jnp.newaxis])
    t1 = time()
    print(f"Time for pre-computing {n_points} neighborhoods: {t1 - t0}")
    coefficients = model_out[..., :-1]
    uncertainties = model_out[..., -1]
    centers, axes, major_radii, minor_radii = torus_precompute_masked_sharded(
        mask, points[np.newaxis], normals[np.newaxis], coefficients
    )
    print(axes.shape)

    block_verts, block_faces, block_colors = torus_marching_cubes(
        points[np.newaxis],
        centers,
        axes,
        major_radii,
        minor_radii,
        neighborhood_radii,
        uncertainties,
        faces
    )
    render_to_html(block_verts, block_faces, block_colors, "sleipner_uncertainty_trial2_block_vis.html")

    #snail_verts, snail_faces, snail_colors = torus_marching_cubes(
    #    points[np.newaxis],
    #    centers,
    #    axes,
    #    major_radii,
    #    minor_radii,
    #    neighborhood_radii,
    #    uncertainties
    #)
    #render_to_html(snail_verts, snail_faces, snail_colors, "sleipner_uncertainty_trial2_snail_vis.html")

    #femur_verts, femur_faces, femur_colors = torus_marching_cubes(
    #    points[np.newaxis],
    #    centers,
    #    axes,
    #    major_radii,
    #    minor_radii,
    #    neighborhood_radii,
    #    uncertainties,
    #    faces
    #)
    #render_to_html(femur_verts, femur_faces, femur_colors, "sleipner_uncertainty_trial2_femur_vis.html")

    #knot_verts, knot_faces, knot_colors = torus_marching_cubes(
    #    points[np.newaxis],
    #    centers,
    #    axes,
    #    major_radii,
    #    minor_radii,
    #    neighborhood_radii,
    #    uncertainties,
    #    faces
    #)
    #render_to_html(knot_verts, knot_faces, knot_colors, "sleipner_uncertainty_trial2_knot_vis.html")

    #tooth_verts, tooth_faces, tooth_colors = torus_marching_cubes(
    #    points[np.newaxis],
    #    centers,
    #    axes,
    #    major_radii,
    #    minor_radii,
    #    neighborhood_radii,
    #    uncertainties
    #)
    #render_to_html(tooth_verts, tooth_faces, tooth_colors, "sleipner_uncertainty_trial2_tooth_vis.html")

    #bunny_verts, bunny_faces, bunny_colors = torus_marching_cubes(
    #    points[np.newaxis],
    #    centers,
    #    axes,
    #    major_radii,
    #    minor_radii,
    #    neighborhood_radii,
    #    uncertainties
    #)
    #render_to_html(bunny_verts, bunny_faces, bunny_colors, "sleipner_uncertainty_trial2_bunny_vis.html")
