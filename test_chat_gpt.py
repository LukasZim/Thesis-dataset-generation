import numpy as np
import open3d as o3d
# import open3d.cuda.pybind.t.geometry
from plyfile import PlyData

input_file = "chair/point_cloud.ply"
ply_data = PlyData.read("./data/" + input_file)
vertex_data = ply_data['vertex'].data
x = vertex_data["x"]
y = vertex_data["y"]
z = vertex_data["z"]

points = np.vstack([x, y, z]).T

r = vertex_data["f_dc_0"]
g = vertex_data["f_dc_1"]
b = vertex_data["f_dc_2"]

colors = np.vstack([r, g, b]).T

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)

print(pcd.has_colors())

angle = np.pi / 2
R = pcd.get_rotation_matrix_from_axis_angle([-angle, 0, 0])
pcd.rotate(R, center=(0, 0, 0))

mesh = o3d.io.read_triangle_mesh("./data/chair/chair.obj", True)
mesh.compute_vertex_normals()
mesh.orient_triangles()

print(mesh)
print(mesh.is_watertight())
mesh.remove_degenerate_triangles()
mesh.remove_duplicated_vertices()
mesh.remove_duplicated_triangles()
mesh.remove_non_manifold_edges()
# mesh.fill_holes()
# mesh = tmesh
print(mesh)
print(mesh.is_watertight())
o3d.visualization.draw_geometries([mesh])

