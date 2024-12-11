import potpourri3d as pp3d
import numpy as np
import open3d as o3d

from geometry_tools.mesh_retrieval import create_mesh_from_faces_and_vertices

# Load the mesh (vertices and faces)
vertices, faces = pp3d.read_mesh("data/bunny_oded.obj")
# mesh2 = o3d.io.read_triangle_mesh("data/bunny_oded.obj")
mesh = create_mesh_from_faces_and_vertices(faces, vertices)

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

ratios = np.sqrt(distances / np.max(distances))
colors = 1-(ratios[:, None] * np.array([1.0, 1.0, 1.0]))
print(ratios)
print(colors)
mesh.compute_vertex_normals()
mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

vis = o3d.visualization.Visualizer()
vis.create_window()

# Set camera parameters if needed
ctr = vis.get_view_control()
ctr.set_front([0, 0, -1])
ctr.set_lookat([0, 0, 0])
ctr.set_up([0, -1, 0])
ctr.set_zoom(0.5)

# Add mesh to visualizer
vis.add_geometry(mesh)

# Configure to disable all light effects
opt = vis.get_render_option()
opt.point_size = 1.0
opt.background_color = np.array([1.0, 1.0, 1.0])  # White background
opt.show_coordinate_frame = False
opt.light_on = False
opt.mesh_show_wireframe = True

# Start the visualizer
vis.run()

# Destroy the visualizer window
vis.destroy_window()


# o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)
