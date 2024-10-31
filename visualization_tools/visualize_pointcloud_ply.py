import numpy as np
import open3d as o3d
from plyfile import PlyData


def get_pcd_ply(filename):
    # filename_labels = str(index) + ".seg"
    ply_data = PlyData.read(filename)
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
    print("Rotating Pointcloud")
    angle = np.pi / 2
    R = pcd.get_rotation_matrix_from_axis_angle([-angle, 0, 0])
    # pcd.rotate(R, center=(0, 0, 0))
    # o3d.visualization.draw_geometries([pcd])

    # estimate normals
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(k=30)
    return pcd


if __name__ == "__main__" :
    filename = "../data/chair/point_cloud.ply"
    pcd = get_pcd_ply(filename)
    o3d.visualization.draw_geometries([pcd])