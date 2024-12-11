import os

import igl
import numpy as np
import open3d as o3d
from gpytoolbox.copyleft import lazy_cage

from visualization_tools.visualize_mesh import get_mesh_from_file
from visualization_tools.visualize_pointcloud_las import get_pcd_las
from visualization_tools.visualize_pointcloud_ply import get_pcd_ply


def sphere_at_location(location, radius = 0.02):
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius = radius)
    sphere.translate(location)
    sphere.paint_uniform_color([1, 0, 0])
    return sphere


def spheres_at_locations(locations, radius = 0.02):
    return [sphere_at_location(location, radius) for location in locations]

def create_mesh_from_vert_and_faces(v, f):
    m = o3d.geometry.TriangleMesh()
    m.vertices = o3d.utility.Vector3dVector(v)
    m.triangles = o3d.utility.Vector3iVector(f)
    m.compute_vertex_normals()
    return m

def create_points_on_mesh(num_impacts, v, f):
    B, FI = igl.random_points_on_mesh(num_impacts, v, f)
    FI[FI >= len(f)] = len(f) - 1
    B = np.vstack((B[:, 0], B[:, 0], B[:, 0], B[:, 1], B[:, 1], B[:, 1], B[:, 2], B[:, 2], B[:, 2])).T
    P = B[:, 0:3] * v[f[FI, 0], :] + B[:, 3:6] * v[f[FI, 1], :] + B[:, 6:9] * v[f[FI, 2], :]
    return P






def load_impact_points(folder_path):
    impact_positions = []
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        with open(filepath, "r") as f:
            impact_positions.append(f.read().split(" ")[:3])

    return impact_positions



if __name__ == "__main__":
    # mesh = get_mesh_from_file("../dataset/Chair/simulation_mesh.obj")
    mesh = get_mesh_from_file("../dataset/Chair/original_mesh.obj")
    # scale_mesh_to_mesh(mesh, mesh2)
    # pcd = get_pcd_las("../data/stanford_bunny.las", downsample=True)

    # pcd = scale_pcd_to_mesh(pcd, mesh)
    # mesh = scale_mesh_to_pcd(pcd, mesh)

    # modes, v_coarse, f_coarse = generate_fracture.create_modes(v, f)
    # v_coarse, f_coarse = lazy_cage(v, f, num_faces=4000, grid_size=256)
    # mesh_coarse = create_mesh_from_vert_and_faces(v_coarse, f_coarse)




    v = np.asarray(mesh.vertices)
    f = np.asarray(mesh.triangles)
    # P = create_points_on_mesh(num_impacts, v, f)

    P = load_impact_points("../dataset/Chair/impulse_info")

    spheres = spheres_at_locations(P, radius = 0.01)
    sphere2 = sphere_at_location([0,0,0])



    o3d.visualization.draw_geometries(spheres + [mesh])
