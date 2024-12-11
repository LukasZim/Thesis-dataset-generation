import numpy as np
import open3d as o3d
from geometry_tools.mesh_retrieval import retrieve_fine_mesh_from_fracture_modes


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





def visualize_labeled_mesh(mesh, labels):
    colors = get_colours_list()[labels]
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([mesh])



def visualize_mesh_and_fine_mesh(mesh, modes):
    fine_mesh = retrieve_fine_mesh_from_fracture_modes(mesh, modes)

    # visualize mesh and fine_mesh
    o3d.visualization.draw_geometries([fine_mesh, mesh])