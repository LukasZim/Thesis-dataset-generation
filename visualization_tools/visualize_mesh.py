import open3d as o3d

from visualization_tools.visualize_pointcloud_ply import get_pcd_ply


def get_mesh_from_file(path):
    mesh = o3d.io.read_triangle_mesh(path)
    # if not mesh.has_vertex_normals():
    mesh.compute_vertex_normals()
    return mesh


if __name__ == "__main__" :
    mesh = get_mesh_from_file("../data/chair/chair3.obj")
    pcd = get_pcd_ply("../data/chair/point_cloud.ply")
    o3d.visualization.draw_geometries([mesh, pcd])