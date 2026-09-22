import os
import trimesh
import thingi10k

import numpy as np
import pointsastori as pat
import matplotlib.pyplot as plt

from typing import *
from scipy.spatial import KDTree

def raymarch(pat_sdf: pat.PointsAsTori) -> np.ndarray:
    ro = np.array([0, 0, -2.5])[np.newaxis]
    xy = np.stack(np.meshgrid(np.arange(-320, 320), np.arange(-240, 240), indexing='xy'), axis=-1).reshape((-1, 2)).astype(np.float64)
    xy /= 480
    xyz = np.concatenate((xy, -1.5*np.ones((480*640, 1))), axis=-1)
    rd = xyz - ro
    rd /= np.linalg.norm(rd)
    t = np.zeros((rd.shape[0], 1))
    d = np.zeros_like(t)

    for _ in range(32):
        p = ro + t*rd
        #print(p.dtype, p.shape, t.shape, d.shape)
        d = pat_sdf.signed_distance(p)[:, np.newaxis]
        #print(p.dtype, p.shape, t.shape, d.shape)
        t += d

    hit = d < 1e-3
    normals = pat_sdf.sdf_gradient(p)

    img = np.where(hit, 0.5 + 0.5*normals, np.zeros_like(normals))
    return img.reshape((640, 480, 3))


def process_example(thing):

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
        return

    # construct points as tori, using both nn and linear least squares
    points = shape.vertices
    normals = shape.vertex_normals
    pat_neural = pat.PointsAsTori(points, normals)
    pat_neural.save_tori(f"tori/{name}/neural/tori.pkl")
    pat_linear = pat.PointsAsTori(points, normals, use_linear_least_squares=True)

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



if __name__ == "__main__":

    os.makedirs("tori", exist_ok=True)

    thingi10k.init()

    for entry in thingi10k.dataset(
            closed=True, manifold=True, oriented=True, self_intersecting=False, solid=False, num_components=1
        ):
        process_example(entry)