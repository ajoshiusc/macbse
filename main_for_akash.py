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
eigenvalues, eigenvectors = eigsh(L, k=num_eigenvectors, which='SM',maxiter=100000)

# Plot the first eigenvector
first_eigenvector = eigenvectors[:, 1]  # Extracting the first eigenvector




# Calculate magnitude of the first eigenvector
magnitude = first_eigenvector #np.linalg.norm(first_eigenvector, axis=1)


fig = plt.figure(figsize=plt.figaspect(0.5))
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.vertices[:, 2], triangles=mesh.faces, cmap='viridis', alpha=0.5)

for i, c in enumerate(np.linspace(np.min(magnitude), np.max(magnitude), 10)):

    # delete the faces with magnitude less than c
    face_mag = np.max(magnitude[mesh.faces],axis=1)
    faces = mesh.faces[face_mag > c, :]
    vertices = mesh.vertices.copy()
    # delete the vertices that are not in the faces
    ind = np.unique(faces)
    vertices = vertices[ind, :]
    magnitude_patch = magnitude[ind]


    # reassign the indices
    for j in range(0, len(ind)):
        faces[faces == ind[j]] = j
    # write the new dfs file
    class newmesh:
        pass
    newmesh.faces = faces
    newmesh.vertices = vertices
    writedfs(f'/home/ajoshi/Desktop/tube_{i}.dfs', newmesh)


    # find the boundary edges
    boundary_edges = []
    for f in faces:
        edge1 = f[[0,1]]
        edge2 = f[[1,2]]
        edge3 = f[[2,0]]
        
        if sum(magnitude_patch[edge1]>c) == 1:
            boundary_edges.append(edge1)
        if sum(magnitude_patch[edge2]>c) == 1:
            boundary_edges.append(edge2)
        if sum(magnitude_patch[edge3]>c) == 1:
            boundary_edges.append(edge3)
    
    boundary_edges = np.array(boundary_edges)

    print(f'Level set {i} has {boundary_edges.shape[0]} edges')

    if len(boundary_edges) == 0:
        continue

    #ax.plot3D(vertices[boundary_edges[:,0], 0], vertices[boundary_edges[:,0], 1], vertices[boundary_edges[:,0], 2], 'r')
    for e in range(boundary_edges.shape[0]):
        ax.plot3D(vertices[boundary_edges[e], 0], vertices[boundary_edges[e], 1], vertices[boundary_edges[e], 2], 'r')

    #plt.title(f'Level set {i}')
    #plt.gca().set_aspect('equal')
    #plt.show()

    # Use PCA to project the 3D coordinates of the boundary edges to 2D
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    pca.fit(vertices[boundary_edges.flatten(), :])
    vertices_2d = pca.transform(vertices) #.reshape(-1, 2)
    vertices_3d = pca.inverse_transform(vertices_2d)
    # Plot the 2D projection of the boundary
    #plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    #ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], triangles=faces, cmap='viridis', alpha=0.5)
    for e in range(boundary_edges.shape[0]):
        ax.plot3D(vertices_3d[boundary_edges[e], 0], vertices_3d[boundary_edges[e], 1], vertices_3d[boundary_edges[e], 2], 'g')

    #plt.title(f'Level set {i}')
    #plt.gca().set_aspect('equal')
    #plt.show()

    #ax.plot3D(vertices[boundary_edges[:,1], 0], vertices[boundary_edges[:,1], 1], vertices[boundary_edges[:,1], 2], 'r')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title(f'Level set {i}')
plt.gca().set_aspect('equal')
plt.show()

