import numpy as np

# import meshio
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
from scipy.sparse.linalg import eigsh
import plotly.graph_objects as go

from dfsio import readdfs, writedfs


def laplace_beltrami(mesh):
    vertices = mesh.vertices
    faces = mesh.faces
    num_vertices = len(vertices)

    # Compute adjacency matrix
    adjacency = np.zeros((num_vertices, num_vertices))
    for face in faces:
        i, j, k = face
        adjacency[i, j] = 1
        adjacency[j, i] = 1
        adjacency[j, k] = 1
        adjacency[k, j] = 1
        adjacency[k, i] = 1
        adjacency[i, k] = 1

    # Compute degree matrix
    degree = np.diag(np.sum(adjacency, axis=1))

    # Compute Laplacian matrix
    laplacian = degree - adjacency

    return laplacian


def get_edges(faces):
    """Get the edges of a mesh given its faces
    Args:
        faces: list of faces of the mesh
    Returns:
        edges: list of edges of the mesh
    """
    edges = []

    for face in faces:
        i, j, k = face
        edges.append((i, j))
        edges.append((j, k))
        edges.append((k, i))

    edges = np.array(edges)
    edges = np.sort(edges, axis=1)
    # edges = np.unique(edges, axis=0)

    return edges


def get_boundary_edges(faces):
    """Get the boundary edges of a mesh given its faces
    Args:
        faces: list of faces of the mesh
    Returns:
        boundary_edges: list of boundary edges of the mesh
    """

    if len(faces) == 0:
        return np.array([])

    edges = get_edges(faces)

    unique_edges, edge_counts = np.unique(edges, axis=0, return_counts=True)

    boundary_edges = unique_edges[edge_counts == 1]

    return boundary_edges


# Read mesh file
mesh = readdfs(
    "/deneb_disk/for_akash/diametercalculationforaorta/aeorta_r2.dfs"
)   #"/home/ajoshi/Desktop/tube.dfs")

boundary_edges = get_boundary_edges(mesh.faces)

# Compute Laplace-Beltrami operator
L = laplace_beltrami(mesh)

# Compute eigenvectors and eigenvalues
num_eigenvectors = 3  # Number of eigenvectors to compute
eigenvalues, eigenvectors = eigsh(L, k=num_eigenvectors, which="SM")#, maxiter=1000)

# Plot the first eigenvector
first_eigenvector = eigenvectors[:, 1]  # Extracting the first eigenvector


# Calculate magnitude of the first eigenvector
magnitude = first_eigenvector  # np.linalg.norm(first_eigenvector, axis=1)


fig = plt.figure(figsize=plt.figaspect(0.5))
ax = fig.add_subplot(111, projection="3d")
ax.plot_trisurf(
    mesh.vertices[:, 0],
    mesh.vertices[:, 1],
    mesh.vertices[:, 2],
    triangles=mesh.faces,
    cmap="viridis",
    alpha=0.5,
)


for i, c in enumerate(np.linspace(np.min(magnitude), np.max(magnitude), 10)):

    # delete the faces with magnitude less than c
    face_mag = np.max(magnitude[mesh.faces], axis=1)
    faces = mesh.faces[face_mag > c, :]
    vertices = mesh.vertices.copy()
    # delete the vertices that are not in the faces
    ind = np.unique(faces)
    vertices = vertices[ind, :]
    magnitude_patch = magnitude[ind]

    # reassign the indices
    for j in range(0, len(ind)):
        faces[faces == ind[j]] = j

    boundary_edges = get_boundary_edges(faces)

    if len(boundary_edges) == 0:
        continue

    for e in range(boundary_edges.shape[0]):
        ax.plot3D(
            vertices[boundary_edges[e], 0],
            vertices[boundary_edges[e], 1],
            vertices[boundary_edges[e], 2],
            "r",
        )

    # Use PCA to project the 3D coordinates of the boundary edges to 2D
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2)
    pca.fit(vertices[boundary_edges.flatten(), :])
    vertices_2d = pca.transform(vertices)
    vertices_3d = pca.inverse_transform(vertices_2d)
    for e in range(boundary_edges.shape[0]):
        ax.plot3D(
            vertices_3d[boundary_edges[e], 0],
            vertices_3d[boundary_edges[e], 1],
            vertices_3d[boundary_edges[e], 2],
            "g",
        )

    # calculate circumference of the level set
    circumference = 0
    for e in range(boundary_edges.shape[0]):
        circumference += np.linalg.norm(
            vertices_3d[boundary_edges[e, 0], :] - vertices_3d[boundary_edges[e, 1], :]
        )

    print(f"Level set {i} has a circumference of {circumference}")


ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.title(f"Level sets of 1st eigenvector of Laplace Beltrami operator")
plt.gca().set_aspect("equal")
plt.show()
