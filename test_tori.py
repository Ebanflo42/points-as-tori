import os
import trimesh
import thingi10k

import numpy as np
import pointsastori as pat
import matplotlib.pyplot as plt

from typing import *
from scipy.spatial import KDTree

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

def raymarch(pat_sdf: pat.PointsAsTori) -> np.ndarray:
    ro = np.array([0, -2, 0])[np.newaxis]
    xz = np.stack(np.meshgrid(np.arange(-500, 500), np.arange(-500, 500), indexing='ij'), axis=-1).reshape((-1, 2)).astype(np.float64)
    xz /= 500
    xyz = np.stack((xz[:, 0], -np.ones((1000**2,)), xz[:, 1]), axis=-1)
    rd = xyz - ro
    rd /= np.linalg.norm(rd, axis=-1, keepdims=True)
    t = np.zeros((rd.shape[0], 1))
    d = np.zeros_like(t)

    for _ in range(32):
        p = ro + t*rd
        #print(p.dtype, p.shape, t.shape, d.shape)
        d = pat_sdf.signed_distance(p)[:, np.newaxis]
        #print(d.min(), d.max(), t.min(), t.max())
        #print(p.dtype, p.shape, t.shape, d.shape)
        t += np.clip(d, 0, 1)

    hit = np.logical_and(d < 1e-3, t < 1e2)
    normals = pat_sdf.sdf_gradient_numeric(p)
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True)

    img = np.where(hit, 0.5 + 0.5*normals, np.zeros_like(normals))
    return img.reshape((1000, 1000, 3))


def process_example(name: str, shape: pat.shape_3d.TriangleMesh):

    # construct points as tori, using both nn and linear least squares
    points = shape.vertices
    normals = shape.vertex_normals
    pat_neural = pat.PointsAsTori(points, normals, chunk_size=5000)
    pat_neural.save_tori(f"tori/{name}/neural/tori.pkl")
    pat_linear = pat.PointsAsTori(points, normals, chunk_size=5000, use_linear_least_squares=True)
    pat_linear.save_tori(f"tori/{name}/linear/tori.pkl")

    print(f"Raymarching {name} . . .")
    neural_raymarch = raymarch(pat_neural)
    linear_raymarch = raymarch(pat_linear)

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.imshow(neural_raymarch)
    plt.savefig(f"tori/{name}/neural/raymarch.png")
    plt.close('all')

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.imshow(linear_raymarch)
    plt.savefig(f"tori/{name}/linear/raymarch.png")
    plt.close('all')


def preprocess_thingi(thing) -> Tuple[str, pat.shape_3d.TriangleMesh]:

    # check directories exist
    name = thing["name"].replace(' ', '')
    print(f"Loading {name} . . .")
    os.makedirs(os.path.join("tori", name, "neural"), exist_ok=True)
    os.makedirs(os.path.join("tori", name, "linear"), exist_ok=True)

    # load mesh
    vertices, faces = thingi10k.load_file(thing['file_path'])

    # try to get normals
    try:
        shape = pat.shape_3d.TriangleMesh(vertices, faces)
        shape.center_and_scale()
        vertices = shape.get_vertices()
        faces = shape.get_faces()

        query = np.array([2, 2, 2])  # outside bounding box and hence mesh
        tree = KDTree(vertices)
        _, closest_vertex_id = tree.query(query)
        cp = vertices[closest_vertex_id]
	    # Get normal from adjacent faces
        mesh_trimesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        vertex_faces = mesh_trimesh.vertex_faces[closest_vertex_id]
        vertex_faces = vertex_faces[vertex_faces >= 0]  # Remove -1 entries
        n = mesh_trimesh.face_normals[vertex_faces].mean(axis=0)
        n = n / np.linalg.norm(n)
        if np.dot(query - cp, n) < 0.0:
            # Flip faces
            faces[:, [1, 2]] = faces[:, [2, 1]]
            shape = pat.shape_3d.TriangleMesh(vertices, faces)
    except:
        print(f"Error {thing['file_path']}")
        return None

    return (name, shape)


def torus_mesh(major_r, minor_r, major_seg=256, minor_seg=256):
    u = np.linspace(0, 2*np.pi, major_seg, endpoint=False)
    v = np.linspace(0, 2*np.pi, minor_seg, endpoint=False)
    U, V = np.meshgrid(u, v, indexing='ij')

    x = (major_r + minor_r*np.cos(V)) * np.cos(U)
    y = (major_r + minor_r*np.cos(V)) * np.sin(U)
    z = minor_r * np.sin(V)

    vertices = np.stack([x, y, z], axis=-1).reshape(-1, 3).astype(np.float64)

    i = np.arange(major_seg)[:, None]
    j = np.arange(minor_seg)[None, :]
    a = (i * minor_seg + j) % (major_seg * minor_seg)
    b = ((i + 1) % major_seg * minor_seg + j) % (major_seg * minor_seg)
    c = ((i + 1) % major_seg * minor_seg + (j + 1) % minor_seg) % (major_seg * minor_seg)
    d = (i * minor_seg + (j + 1) % minor_seg) % (major_seg * minor_seg)

    a, b, c, d = a.ravel(), b.ravel(), c.ravel(), d.ravel()
    faces = np.concatenate([
        np.stack([a, b, c], axis=-1),
        np.stack([a, c, d], axis=-1),
    ], axis=0).astype(np.int64)

    return vertices, faces


def sanity_check():
    mesh = torus_mesh(1, 0.25)
    shape = pat.shape_3d.TriangleMesh(mesh[0], mesh[1])
    process_example("torus", shape)


if __name__ == "__main__":

    os.makedirs("tori", exist_ok=True)

    #"""
    thingi10k.init()
    for entry in thingi10k.dataset(
            closed=True, manifold=True, oriented=True, self_intersecting=False, solid=False, num_components=1
        ):
        preprocessed = preprocess_thingi(entry)
        if preprocessed is not None:
            process_example(preprocessed[0], preprocessed[1])
    #"""
    #sanity_check()