import numpy as np
#import meshio
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

# Read mesh file
mesh = readdfs("/home/ajoshi/Desktop/tube.dfs")
#meshio.read("mesh_file.obj")

# Compute Laplace-Beltrami operator
L = laplace_beltrami(mesh)

# Compute eigenvectors and eigenvalues
num_eigenvectors = 10  # Number of eigenvectors to compute
eigenvalues, eigenvectors = eigsh(L, k=num_eigenvectors, which='SM')

# Plot the first eigenvector
first_eigenvector = eigenvectors[:, 1]  # Extracting the first eigenvector




# Calculate magnitude of the first eigenvector
magnitude = first_eigenvector #np.linalg.norm(first_eigenvector, axis=1)

# Plot the mesh with colors based on the magnitude of the first eigenvector
fig = go.Figure()

# Add mesh surface
fig.add_trace(go.Mesh3d(
    x=mesh.vertices[:, 0],
    y=mesh.vertices[:, 1],
    z=mesh.vertices[:, 2],
    i=mesh.faces[:, 0],
    j=mesh.faces[:, 1],
    k=mesh.faces[:, 2],
    intensity=magnitude,
    colorscale='Viridis',
    opacity=0.8
))

# Set layout
fig.update_layout(
    scene=dict(
        xaxis=dict(title='X'),
        yaxis=dict(title='Y'),
        zaxis=dict(title='Z'),
        aspectmode='data'
    ),
    title='Mesh colored by Magnitude of First Eigenvector'
)

# Show the plot
fig.show()




"""
# Plot the mesh with the first eigenvector
#fig = plt.figure()
fig = plt.figure(figsize=plt.figaspect(0.5))

ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.vertices[:, 2], triangles=mesh.faces, cmap='viridis', alpha=0.5)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title('First Eigenvector of Laplace-Beltrami Operator')
plt.gca().set_aspect('equal')

plt.show()
"""