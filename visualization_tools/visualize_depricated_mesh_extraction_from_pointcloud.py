from visualize_pointcloud_las import get_pcd_las
import open3d as o3d
import numpy as np

def extract_mesh_from_point_cloud(pointcloud):
    # perform poisson surface reconstruction
    m, d = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pointcloud, depth=15)
    #radii = [0.08, 0.16, .32]
    #m = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
    #   pcd, o3d.utility.DoubleVector(radii))

    m.vertices = o3d.utility.Vector3dVector(np.asarray(m.vertices) - np.mean(np.asarray(m.vertices), axis=0))
    # filter to make it look better
    # vertices_to_remove = densities < np.quantile(densities, 0.05)
    # mesh.remove_vertices_by_mask(vertices_to_remove)
    m.compute_vertex_normals()

    return m, d

def give_mesh_colour(m):

    num_vertices = np.asarray(m.vertices).shape[0]

    m.vertex_colors = o3d.utility.Vector3dVector(np.tile([0.5, 0.5, 0.5], (num_vertices, 1)))
    return m

if __name__ == "__main__" :
    downsample = True
    filename = "../data/stanford_bunny.las"
    print("starting")
    pcd = get_pcd_las(filename=filename, downsample=downsample)
    print("retrieved PCD")
    mesh, _ = extract_mesh_from_point_cloud(pcd)
    print("extracted mesh")
    mesh = give_mesh_colour(mesh)
    print("coloured mesh")

    o3d.visualization.draw_geometries([mesh])
