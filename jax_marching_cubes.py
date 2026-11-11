import os
import trimesh
import open3d as o3d

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


def torus_marching_cubes(points: jnp.ndarray,
                         centers: jnp.ndarray,
                         axes: jnp.ndarray,
                         major_radii: jnp.ndarray,
                         minor_radii: jnp.ndarray,
                         neighborhood_radii: jnp.ndarray,
                         uncertainties: jnp.ndarray):

    grid = np.stack(np.meshgrid(
        np.linspace(-1, 1, num=64),
        np.linspace(-1, 1, num=64),
        np.linspace(-1, 1, num=64),
    ), axis=-1).reshape((1, -1, 3)).astype(np.float32)

    d, _, _ = torus_distance_gradient_masked_with_clamp_chunked(
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

    d = d.reshape((64, 64, 64))

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
        "/home/kadile/Projects/points-as-tori/models/Better_Blending_Wide_Trial1/Better_Blending_Wide_Trial1_phase2"
    ))

    print("Loading point cloud . . .")
    points, normals, neighbors = load_block()
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
        uncertainties
    )

    render_to_html(block_verts, block_faces, block_colors, "wide_block_vis.html")
    """
    snail_verts, snail_faces, snail_colors = torus_marching_cubes(
        points[np.newaxis],
        centers,
        axes,
        major_radii,
        minor_radii,
        neighborhood_radii,
        uncertainties
    )

    render_to_html(snail_verts, snail_faces, snail_colors, "snail_vis.html")
    """
    """
    femur_verts, femur_faces, femur_colors = torus_marching_cubes(
        points[np.newaxis],
        centers,
        axes,
        major_radii,
        minor_radii,
        neighborhood_radii,
        uncertainties
    )

    render_to_html(femur_verts, femur_faces, femur_colors, "femur_vis.html")
    """
    """
    knot_verts, knot_faces, knot_colors = torus_marching_cubes(
        points[np.newaxis],
        centers,
        axes,
        major_radii,
        minor_radii,
        neighborhood_radii,
        uncertainties
    )

    render_to_html(knot_verts, knot_faces, knot_colors, "knot_vis.html")
    """
    """
    tooth_verts, tooth_faces, tooth_colors = torus_marching_cubes(
        points[np.newaxis],
        centers,
        axes,
        major_radii,
        minor_radii,
        neighborhood_radii,
        uncertainties
    )

    render_to_html(tooth_verts, tooth_faces, tooth_colors, "tooth_vis.html")
    """
    """
    bunny_verts, bunny_faces, bunny_colors = torus_marching_cubes(
        points[np.newaxis],
        centers,
        axes,
        major_radii,
        minor_radii,
        neighborhood_radii,
        uncertainties
    )
    render_to_html(block_verts, block_faces, block_colors, "block_vis.html")
    """
