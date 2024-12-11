import sys
import numpy as np
import pickle
import os
import open3d as o3d
from scipy.spatial import KDTree
from generate_UDF_dataset import Config

from geometry_tools.mesh_retrieval import retrieve_fine_mesh_from_fracture_modes
from geometry_tools.scaling import scale_mesh_to_mesh_inplace
from geometry_tools.visualization import visualize_labeled_mesh


def unload_pickle(file):
    with open(file, 'rb') as f:
        return pickle.load(f)

def map_fine_labels_to_mesh(mesh, modes):
    fine_mesh = retrieve_fine_mesh_from_fracture_modes(mesh, modes)
    scale_mesh_to_mesh_inplace(fine_mesh, mesh)
    kd_tree_mesh2 = KDTree(np.asarray(fine_mesh.vertices))
    _, indices = kd_tree_mesh2.query(np.asarray(mesh.vertices))

    labels = modes.fine_vertex_labels_after_impact[indices]
    return labels



def generate_UDF_dataset(folder):
    # loop over all .pkl files in the folder
    for filename in os.listdir(folder):
        # load the pickle
        config = unload_pickle(os.path.join(folder, filename))

        # extract important variables from config
        modes = config.modes
        mesh = o3d.io.read_triangle_mesh(config.mesh_filename)
        mesh.compute_vertex_normals()

        # visualize the segmentation stored in modes
        visualize_labeled_mesh(retrieve_fine_mesh_from_fracture_modes(mesh,modes), modes.fine_vertex_labels_after_impact) # fine

        # visualize the segmentation mapped to the original mesh
        new_labels = map_fine_labels_to_mesh(mesh, modes)
        visualize_labeled_mesh(mesh, new_labels) # original


        #TODO: find a way of determining the points for which a point at a connecting edge has a different label

        #TODO: visualize these points

        #TODO: load this information somehow into potpourri3d

        #TODO: use potpourri3d to calculate the geodesic distance from any point to this set of points.

        #TODO: visualize the UDF that was calculated

        #TODO: OR SOMEHOW USE THE FRACTURE-MODES MESH IN HERE





if __name__ == '__main__':
    folder = "pickled_modes"
    generate_UDF_dataset(folder)
    Config() # just here such that I dont do cleanup and remove the import
