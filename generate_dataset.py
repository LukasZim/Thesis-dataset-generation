import random

import io
import sys

import laspy
import open3d as o3d
import numpy as np
from scipy.spatial import KDTree
import time

from plyfile import PlyData

import generate_fracture




def downsample(pcd, down_sampling_method, every_k_points=100, voxel_size=0.005, sample_x_points=2000):
    # Uniform downsampling of the point cloud
    if down_sampling_method == "uniform":
        pcd = pcd.uniform_down_sample(every_k_points=every_k_points)
    elif down_sampling_method == "voxel":
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    elif down_sampling_method == "sample_x":
        total_points = len(pcd.points)
        sampling_ratio = sample_x_points / total_points
        pcd = pcd.random_down_sample(sampling_ratio=sampling_ratio)
    elif down_sampling_method.lower() != "none":
        print("sampling method not found, defaulting to using no method")

    return pcd


def extract_point_cloud_from_las_file(filename):
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

    # visualize normal estimation
    # o3d.visualization.draw_geometries([pcd], point_show_normal=True)

    return pcd


def extract_splat_from_ply_file(filename):
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
    print("PCD rotation not needed anymore")
    # o3d.visualization.draw_geometries([pcd])

    # estimate normals
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(k=30)

    return pcd


def extract_mesh_from_obj_file(mesh_filename):
    mesh = o3d.io.read_triangle_mesh(mesh_filename, True)
    mesh.compute_vertex_normals()

    # o3d.visualization.draw_geometries([mesh])
    return mesh


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


def find_mapping(pointcloud, m):
    index = 0
    dict_mapping = {}
    for pointcloud_point in pointcloud.points:
        min_index = np.argmin(np.linalg.norm(m.vertices - pointcloud_point, axis=1))
        dict_mapping[index] = min_index
        index += 1
        if index % 100 == 0:
            print(index / len(pointcloud.points))

    return dict_mapping


# TODO: there is still a scaling issue between the pointcloud and mesh
def find_scipy_mapping(pointcloud, fracture_modes):
    start_time = time.time()
    pointcloud_points = np.asarray(pointcloud.points)
    mesh_points = fracture_modes.fine_vertices
    print(np.max(pointcloud_points, axis=0))
    print(np.min(pointcloud_points, axis=0))
    print(np.max(mesh_points, axis=0))
    print(np.min(mesh_points, axis=0))
    print(np.max(mesh.vertices, axis=0))
    print(np.min(mesh.vertices, axis=0))
    pcd_min = np.min(pointcloud_points, axis=0)
    pcd_max = np.max(pointcloud_points, axis=0)
    mesh_min = np.min(mesh_points, axis=0)
    mesh_max = np.max(mesh_points, axis=0)
    mesh_points = (mesh_points - mesh_min) / (mesh_max - mesh_min) * (pcd_max - pcd_min) + pcd_min
    kd_tree_mesh2 = KDTree(mesh_points)
    _, indices = kd_tree_mesh2.query(pointcloud_points)

    labels = fracture_modes.fine_vertex_labels_after_impact[indices]
    print(np.unique(fracture_modes.fine_vertex_labels_after_impact, return_counts=True))
    print(np.unique(labels, return_counts=True))
    # closest_points = mesh_points[indices]
    print(time.time() - start_time)
    return np.arange(0, len(labels)), labels


def get_colours_list():
    return np.array([
        [1.0, 0.0, 0.0],  # Red
        [0.0, 1.0, 0.0],  # Green
        [0.0, 0.0, 1.0],  # Blue
        [1.0, 1.0, 0.0],  # Yellow
        [1.0, 0.0, 1.0],  # Magenta
        [0.0, 1.0, 1.0],  # Cyan
        [0.5, 0.5, 0.5],  # Gray
        [1.0, 0.5, 0.0],  # Orange
        [0.5, 0.0, 1.0],  # Purple
        [0.0, 0.5, 0.5],  # Teal
        [0.3, 0.7, 0.2],  # Custom color 1
        [0.1, 0.5, 0.9],  # Custom color 2
        [0.9, 0.2, 0.5],  # Custom color 3
        [0.6, 0.4, 0.7],  # Custom color 4
        [0.2, 0.8, 0.3],  # Custom color 5
        [0.8, 0.3, 0.1],  # Custom color 6
        [0.4, 0.1, 0.6],  # Custom color 7
        [0.7, 0.9, 0.2],  # Custom color 8
        [0.5, 0.2, 0.8],  # Custom color 9
        [0.0, 0.8, 0.9]  # Custom color 10
    ])

def extract_point_cloud_from_file(filename):
    if filename.endswith(".las"):
        return extract_point_cloud_from_las_file(filename)

    elif filename.endswith(".ply"):
        return extract_splat_from_ply_file(filename)

    raise Exception("Sorry, file type not supported: ", filename.split(".")[-1])

def scale_mesh_to_pcd(pcd, mesh):
    min_bound_pcd, max_bound_pcd = np.asarray(pcd.get_min_bound()), np.asarray(pcd.get_max_bound())
    min_bound_mesh, max_bound_mesh = np.asarray(mesh.get_min_bound()), np.asarray(mesh.get_max_bound())

    center_pcd = (min_bound_pcd + max_bound_pcd) / 2
    center_mesh = (min_bound_mesh + max_bound_mesh) / 2
    range1 = max_bound_pcd - min_bound_pcd
    range2 = max_bound_mesh - min_bound_mesh

    # Find scale factor to match the ranges (based on the largest relative axis range)
    scale_factor = range1 / range2  # element-wise scale factor

    # To make scaling uniform, use the largest scale factor among axes
    uniform_scale = max(scale_factor)

    mesh.vertices = o3d.utility.Vector3dVector((np.asarray(mesh.vertices) - center_mesh) * uniform_scale + center_pcd)
    return mesh

if __name__ == "__main__":
    pcd_downsample_needed = True
    down_sampling_method = "sample_x"
    filename = "/home/lukasz/Documents/thesis_pointcloud/data/stanford_bunny.las"
    mesh_filename = "./data/bunny_oded.obj"
    output_path = "./results"
    mesh_extraction_needed = False
    mesh_aligning_needed = True
    dataset_name = "dataset"
    category_name = "bunny"
    dataset_size = 100


    print("starting")

    # pcd = extract_point_cloud_from_las_file(filename)
    # pcd = extract_splat_from_ply_file(filename)
    pcd = extract_point_cloud_from_file(filename)
    print("extracted point cloud")

    # pcd.scale(10, center=pcd.get_center())
    if pcd_downsample_needed:
        pcd = downsample(pcd, down_sampling_method, every_k_points=50)
        print("downsampled point cloud")

    if mesh_extraction_needed:
        print("extracting mesh")
        mesh, densities = extract_mesh_from_point_cloud(pcd)
        print("DONE extracted mesh from point cloud")
    else:
        print("reading mesh form file: ", mesh_filename)
        mesh = extract_mesh_from_obj_file(mesh_filename)
        # if mesh_aligning_needed:
        mesh = scale_mesh_to_pcd(pcd, mesh) if mesh_aligning_needed else mesh
        print("reading mesh from file DONE")

    print(mesh.get_center())
    print(pcd.get_center())
    # o3d.visualization.draw_geometries([mesh, pcd])



    # sys.stdout = io.StringIO()
    print("Start creating Fracture modes")
    modes, v, f = generate_fracture.create_modes(mesh.vertices, mesh.triangles)
    print("Done creating Fracture modes")
    # sys.stdout = sys.__stdout__
    print("mesh.vertices = ", len(mesh.vertices))
    # print("mesh.faces = ", len(mesh.triangles))
    # print("pcd.points = ", len(pcd.points))
    print("modes.fine_vertices = ", len(modes.fine_vertices))
    # print("modes.fine_faces = ", len(modes.fine_triangles))
    print("done generating modes")

    print(mesh.get_center())
    print(pcd.get_center())
    # o3d.visualization.draw_geometries([mesh, pcd])

    generate_fracture.generate_multiple_fractures(modes, num_impacts=dataset_size, v=v, f=f, category_name=category_name,
                                                  dataset_name=dataset_name, pcd=pcd, mesh=mesh, ply_file=filename)

    # _, labels = find_scipy_mapping(pcd, modes)
    # print("found mapping")
    #
    # # Visualize Point Cloud
    # # o3d.visualization.draw_geometries([pcd])
    #
    # # Define 20 different RGB colors in the range [0-1]
    # colors_list = get_colours_list()
    # # colours = np.repeat((labels / np.max(labels)), 3).reshape(-1, 3)
    # colours = colors_list[labels]
    # pcd.colors = o3d.utility.Vector3dVector(colours)
    # print("calculated colours")
    #
    # # Visualize mesh and pointcloud
    # o3d.visualization.draw_geometries([pcd, mesh])
    #
    # # visualize mesh
    # # o3d.visualization.draw_geometries([mesh])
    # print("done")
