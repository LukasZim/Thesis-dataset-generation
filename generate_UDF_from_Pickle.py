import shutil

import pandas as pd
import sys
import numpy as np
import pickle
import os
import open3d as o3d
from scipy.spatial import KDTree
from generate_UDF_dataset import Config
import potpourri3d as pp3d
from tqdm import tqdm

from geometry_tools.mesh_retrieval import retrieve_fine_mesh_from_fracture_modes, load_mesh_from_file, \
    create_mesh_from_faces_and_vertices
from geometry_tools.scaling import scale_mesh_to_mesh_inplace
from geometry_tools.visualization import visualize_labeled_mesh, visualize_UDF, visualize_mesh_with_contact_point


def unload_pickle(file):
    with open(file, 'rb') as f:
        return pickle.load(f)

def map_fine_labels_to_mesh(mesh, modes):
    """
    Takes a mesh and a GT fractured mesh and maps the labels of the fractured mesh to the original mesh using scaling of the AABB
    :param mesh: the mesh you want the labelling for
    :param modes: The fracture modes object storing the GT labeling
    :return: a list of labels for the mesh param, size is num(vertices)
    """
    fine_mesh = retrieve_fine_mesh_from_fracture_modes(mesh, modes)
    scale_mesh_to_mesh_inplace(fine_mesh, mesh)
    kd_tree_mesh2 = KDTree(np.asarray(fine_mesh.vertices))
    _, indices = kd_tree_mesh2.query(np.asarray(mesh.vertices))

    labels = modes.fine_vertex_labels_after_impact[indices]
    return labels


def determine_edge_points(mesh, labels):
    """
    Takes a mesh and a labelling and calculates the vertex points on a fracture line.
    :param mesh: mesh
    :param labels: labelling of the mesh
    :return: Tuple of labelling of the fracture line and list of vertex points on the fracture line
    """
    edge_labels = np.zeros(labels.shape).astype(int)
    vert_list = []

    for triangle in mesh.triangles:

        #check if triangle[0] is on an fracture edge
        if labels[triangle[0]] < labels[triangle[1]] or labels[triangle[0]] < labels[triangle[2]]:
            edge_labels[triangle[0]] = 1
            if triangle[0] not in vert_list:
                vert_list.append(triangle[0])

        # check if triangle[1] is on a fracture edge
        if labels[triangle[1]] < labels[triangle[2]] or labels[triangle[1]] < labels[triangle[0]]:
            edge_labels[triangle[1]] = 1
            if triangle[1] not in vert_list:
                vert_list.append(triangle[1])

        # check if triangle[2] i s a fracture edge
        if labels[triangle[2]] < labels[triangle[0]] or labels[triangle[2]] < labels[triangle[1]]:
            edge_labels[triangle[2]] = 1
            if triangle[2] not in vert_list:
                vert_list.append(triangle[2])

    return edge_labels, vert_list

def calculate_UDF(mesh, edge_set, clamping_distance=0.3):
    """
    Takes a mesh and a set of edge points to calculate the UDF using Geodesic distance with potpourri3d
    :param mesh: mesh to use, should be loaded in using mesh_retrieval.load_mesh_from_file or potpourri3d's mesh load method
    :param edge_set: Set of points on the edge of the fracture lines
    :param clamping_distance: The distance for which all values greater than it will be rounded ot it. Nice for visualization and some DL methods
    :return: The Geodesic distances as labels for each vertex in the mesh to the edge_set vertices.
    """
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    # Initialize the geodesic solver
    # solver = pp3d.MeshHeatMethodDistanceSolver(vertices, faces)
    #
    # distances = np.abs(np.array(solver.compute_distance_multisource(edge_set)))
    # distances[distances > clamping_distance] = clamping_distance

    edge_set_positions = np.array(vertices[edge_set])
    distances = np.min(np.linalg.norm(vertices[:, np.newaxis, :] - edge_set_positions[np.newaxis, :, :], axis=2), axis=1)


    return distances


def write_data_to_file(root_dataset_folder, config, distances, index, vertices, mesh, labeling, edge_labels, impact_point, direction):
    # check if the root folder exists, if not create it
    os.makedirs(root_dataset_folder, exist_ok=True)

    # check if the dataset folder exists
    os.makedirs(os.path.join(root_dataset_folder, config.category_name), exist_ok=True)

    if index == 0:
        # write mesh to file in dataset
        shutil.copyfile(config.mesh_filename, os.path.join(root_dataset_folder, config.category_name, config.category_name + ".obj"))

    # open a file to write to and determine a name, maybe look at config?
    distances = np.reshape(distances, (-1, 1))
    DF = pd.DataFrame(distances)
    DF_vertices = pd.DataFrame(vertices)
    DF_labels = pd.DataFrame(labeling)
    DF_edge_labels = pd.DataFrame(edge_labels)
    DF = pd.concat([DF_labels, DF_edge_labels, DF, DF_vertices], axis=1)
    DF.columns = ['label', 'edge_labels', 'distance', 'x', 'y', 'z']
    filename = str(os.path.join(root_dataset_folder, config.category_name, str(index) + ".pkl"))
    DF.to_pickle(filename)

    DF = pd.DataFrame(np.concatenate((impact_point, direction)).reshape(1, -1))
    DF.columns = ['x_point', 'y_point', 'z_point', 'x_direction', 'y_direction', 'z_direction']
    filename = str(os.path.join(root_dataset_folder, config.category_name, str(index) + "_impulse.pkl"))
    DF.to_pickle(filename)

    # o3d.io.write_triangle_mesh(str(os.path.join(root_dataset_folder, config.category_name, config.category_name + ".obj")), mesh)


def rescale_impact_position(position, sim_min, sim_max, mesh_min, mesh_max):
    point = np.array(position)
    src_min = np.array(sim_min)
    src_max = np.array(sim_max)
    tgt_min = np.array(mesh_min)
    tgt_max = np.array(mesh_max)
    # Compute the rescaled point
    rescaled_point = tgt_min + (point - src_min) / (src_max - src_min) * (tgt_max - tgt_min)
    return rescaled_point

def generate_UDF_dataset(pickle_folder, root_dataset_folder, do_visualize = True, clamping_distance=9999999):

    #TODO: write mesh to dataset_folder
    count = 0
    # loop over all .pkl files in the folder
    for filename in tqdm(os.listdir(pickle_folder)):
        count+=1
        if count < 14:
            continue

        # get index from the filename
        print(filename)
        index = int(filename.split('_')[0])

        # load the pickle
        config = unload_pickle(os.path.join(pickle_folder, filename))

        # extract important variables from config
        modes = config.modes
        mesh = load_mesh_from_file(config.mesh_filename)

        contact_point = config.contact_point
        direction = config.direction

        f = config.f
        v = config.v

        if do_visualize:
            visualize_mesh_with_contact_point(create_mesh_from_faces_and_vertices(f, v), contact_point)

        print(contact_point)
        print(np.max(v, axis=0), np.min(v, axis=0))
        print(np.max(mesh.vertices, axis=0), np.min(mesh.vertices, axis=0))
        contact_point = rescale_impact_position(contact_point, np.min(v, axis=0), np.max(v, axis=0), np.min(np.asarray(mesh.vertices), axis=0), np.max(np.asarray(mesh.vertices), axis=0),)

        # visualize the segmentation stored in modes
        if do_visualize:
            # visualize_labeled_mesh(retrieve_fine_mesh_from_fracture_modes(mesh,modes), modes.fine_vertex_labels_after_impact) # fine
            visualize_mesh_with_contact_point(mesh, contact_point)

        # visualize the segmentation mapped to the original mesh
        new_labels = map_fine_labels_to_mesh(mesh, modes)
        # Define 90-degree rotation matrix around Z-axis
        R = mesh.get_rotation_matrix_from_axis_angle([0, -np.pi / 4, 0])

        # Apply rotation
        mesh.rotate(R, center=(0, 0, 0))
        if do_visualize:
            visualize_labeled_mesh(mesh, new_labels, filename="labels.png") # original


        # determine the points for which a point at a connecting edge has a different label
        edge_labels, edge_set = determine_edge_points(mesh, new_labels)

        # visualize these points
        if do_visualize:
            visualize_labeled_mesh(mesh, edge_labels, filename="edge_labels.png")

        # use potpourri3d to calculate the geodesic distance from any point to this set of points.
        distances = calculate_UDF(mesh, edge_set)

        # visualize the UDF that was calculated
        if do_visualize:
            visualize_UDF(mesh, distances, contact_point=contact_point)

        # visualize the UDF without any clamping
        if do_visualize:
            visualize_UDF(mesh, calculate_UDF(mesh, edge_set, clamping_distance=99999))


        distances = calculate_UDF(mesh, edge_set, clamping_distance=clamping_distance)

        # generate the actual dataset
        write_data_to_file(root_dataset_folder, config, distances, index, np.asarray(mesh.vertices), mesh, new_labels, edge_labels, impact_point=contact_point, direction=direction)








if __name__ == '__main__':
    pickle_folder = "pickled_modes"
    root_dataset_folder = "datasets"
    clamping_distance = 9999999
    generate_UDF_dataset(pickle_folder, root_dataset_folder, do_visualize = True, clamping_distance=clamping_distance)
    Config(0) # just here such that I dont do cleanup and remove the import, breaking the pickling
