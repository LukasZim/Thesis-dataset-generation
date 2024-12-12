import potpourri3d as pp3d
import numpy as np
import open3d as o3d

from geometry_tools.mesh_retrieval import create_mesh_from_faces_and_vertices, load_mesh_from_file
from geometry_tools.visualization import visualize_UDF

mesh = load_mesh_from_file("data/bunny_oded.obj")
vertices = np.asarray(mesh.vertices)
faces = np.asarray(mesh.triangles)
# Initialize the geodesic solver
solver = pp3d.MeshHeatMethodDistanceSolver(vertices, faces)
# distances = np.empty((0), float)
# for i in range(0, len(vertices)):
#     # Define the source point (index) and collection of target points (indices)
#     source_point = i  # Example: index of the source point
#     target_points = [333]  # Example: indices of target points
#
#     # Compute distances from source to all vertices
#     distances = solver.compute_distance(source_point)
#
#     # Extract distances for target points
#     shortest_distances = np.min(distances[target_points])
#     print(i, shortest_distances)
#     np.append(distances, shortest_distances)
distances = np.abs(np.array(solver.compute_distance_multisource([1000, 1111, 2902, 800])))
distances[distances > 0.3] = 0.3




visualize_UDF(mesh, distances)


# o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)
