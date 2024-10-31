import laspy
import open3d as o3d
import numpy as np

def get_pcd_las(filename, downsample):


    # Read the LAS file
    las = laspy.read(filename)

    # Extract point coords
    points = np.vstack((las.x, las.y, las.z)).transpose()
    points = points - np.mean(points, axis=0)

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # estimate normals
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(k=30)

    if downsample:
        pcd = pcd.uniform_down_sample(every_k_points=100)

    # visualize normal estimation
    # o3d.visualization.draw_geometries([pcd], point_show_normal=True)
    return pcd

if __name__ == "__main__" :
    downsample = True
    filename = "../data/stanford_bunny.las"
    pcd = get_pcd_las(filename, downsample)
    o3d.visualization.draw_geometries([pcd])
