import os
import numpy as np
from time import time
from argparse import ArgumentParser

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from skimage.measure import marching_cubes

import jax
import jax.numpy as jnp
from flax import nnx

from pointsastori.torus_utils import torus_precompute_masked_sharded, torus_distance_gradient_masked_with_uncertainty_chunked
from pointsastori.mesh_sampling_jax import signed_distance_to_mesh
from pointsastori.network import FundamentalFormPredictorWithUncertainty, ModelConfig
from .vis_utils import *

from typing import *



def torus_marching_cubes(n_chunks: int,
                         points: jnp.ndarray,
                         centers: jnp.ndarray,
                         axes: jnp.ndarray,
                         major_radii: jnp.ndarray,
                         minor_radii: jnp.ndarray,
                         uncertainties: jnp.ndarray,
                         faces=None
):

    grid = np.stack(np.meshgrid(
        np.linspace(-1, 1, num=64),
        np.linspace(-1, 1, num=64),
        np.linspace(-1, 1, num=64),
    ), axis=-1).reshape((1, -1, 3)).astype(np.float32)

    d, grads, _ = torus_distance_gradient_masked_with_uncertainty_chunked(
        jnp.ones(points.shape[:2], dtype=jnp.bool),
        grid,
        points,
        centers,
        axes,
        major_radii,
        minor_radii,
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
    print(f"VERTICES S MIN MAX X Y Z {verts[:, 0].min()} {verts[:, 0].max()} {verts[:, 1].min()} {verts[:, 1].max()} {verts[:, 2].min()} {verts[:, 2].max()}")
    print(f"VERTICES SHAPE: {verts.shape}")

    _, _, uncertainties = torus_distance_gradient_masked_with_uncertainty_chunked(
        jnp.ones(points.shape[:2], dtype=jnp.bool),
        verts[np.newaxis],
        points,
        centers,
        axes,
        major_radii,
        minor_radii,
        uncertainties,
        n_chunks=n_chunks
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

    parser = ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to orbax checkpoint")
    parser.add_argument("--n_layers", type=int, default=8, help="Expected number of transformer layers in the checkpoint")
    parser.add_argument("--mlp_width", type=int, default=256, help="Expected width of MLP layers in the checkpoint")
    parser.add_argument("--mha_width", type=int, default=128, help="Expected width of MHA layers in the checkpoint")
    parser.add_argument("--shape", type=str, default="bunny", help="Which shape to visualize")
    parser.add_argument("--n_vertex_chunks", type=int, default=1, help="How many chunks to evaluate vertex uncertainties in")
    parser.add_argument("--output_file", type=str, required=True, help="Name of the html file to render the visualization to")
    args = parser.parse_args()

    assert args.shape in ["bunny", "block", "snail", "knot", "femur", "tooth"]

    print("Loading model . . .")
    config = ModelConfig(args.n_layers, args.mha_width, args.mlp_width, 8)
    model = FundamentalFormPredictorWithUncertainty(config)
    model.restore_from_checkpoint(os.path.abspath(args.model_path))
    param_tree = nnx.state(model, nnx.Param)
    param_count = sum(p.size for p in jax.tree_util.tree_leaves(param_tree))
    print(f"Parameter count: {param_count}")

    print("Loading shape . . .")
    if args.shape == "bunny":
        points, normals, neighbors = load_bunny()
        faces = None
    elif args.shape == "snail":
        points, normals, neighbors = load_snail()
        faces = None
    elif args.shape == "tooth":
        points, normals, neighbors = load_tooth()
        faces = None
    elif args.shape == "block":
        points, normals, neighbors, faces = load_block()
    elif args.shape == "femur":
        points, normals, neighbors, faces = load_femur()
    elif args.shape == "knot":
        points, normals, neighbors, faces = load_knot()

    points, normals, neighbors, faces = load_block()
    print(f"x y z min max {points[:, 0].min()} {points[:, 0].max()} {points[:, 1].min()} {points[:, 1].max()} {points[:, 2].min()} {points[:, 2].max()}")
    n_points = len(points)
    mask = jnp.ones((1, n_points), dtype=jnp.bool)

    jax_points, jax_normals, jax_neighbors = \
        jnp.asarray(points[neighbors]), jnp.asarray(normals[neighbors]), jnp.asarray(neighbors)

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

    block_verts, block_faces, block_colors = torus_marching_cubes(
        args.n_chunks,
        points[np.newaxis],
        centers,
        axes,
        major_radii,
        minor_radii,
        uncertainties,
        faces
    )
    render_to_html(block_verts, block_faces, block_colors, args.output_file)
