import pickle

import shutil

import numpy as np
import igl
import tetgen
import open3d as o3d

from fracture_modes import fracture_utility as fracture
# import fracture_utility as fracture
from gpytoolbox.copyleft import lazy_cage
import gpytoolbox
import os



def generate_fracture(v_fine, f_fine, num_faces=100, output_folder="./results"):
    v_fine = np.asarray(v_fine)
    f_fine = np.asarray(f_fine)
    v_fine = gpytoolbox.normalize_points(v_fine)

    # This is the "cage mesh", i.e. the coarser mesh that we will tetrahedralize and use for the physical simulation
    v, f = lazy_cage(v_fine, f_fine, num_faces=num_faces)

    # Tetrahedralize
    tgen = tetgen.TetGen(v, f)
    nodes, elements = tgen.tetrahedralize()

    # Initialize fracture mode class
    modes = fracture.fracture_modes(nodes, elements)

    # Set parameters for call to fracture modes
    params = fracture.fracture_modes_parameters(num_modes=10, verbose=True, d=3)

    # Compute fracture modes
    modes.compute_modes(parameters=params)

    # We need to precompute some stuff that we will only need to do once
    modes.impact_precomputation(v_fine=v_fine, f_fine=f_fine)

    # Optionally, you can use this to save each mode's segmentation in the current directory
    modes.write_segmented_modes(output_folder + "/output_modes", pieces=True)
    contact_point = nodes[1, :]
    direction = np.array([1.0, 0.0, 0.0])

    # First projection, this should be fast
    modes.impact_projection(contact_point=contact_point, direction=direction)

    # Second projection, this should be fast
    new_contact_point = nodes[5, :]
    modes.impact_projection(contact_point=new_contact_point, direction=direction)

    # Write segmented output to obj
    modes.write_segmented_output(output_folder + "/output.obj")
    # modes.write

    print(modes.n_pieces_after_impact)

    return modes


def create_modes(v_fine, f_fine, num_faces=4000, output_folder="./results"):
    v_fine = np.asarray(v_fine).copy()
    f_fine = np.asarray(f_fine).copy()
    print("v_fine", len(v_fine))
    v_fine = gpytoolbox.normalize_points(v_fine)
    print("v_fine", len(v_fine))

    # This is the "cage mesh", i.e. the coarser mesh that we will tetrahedralize and use for the physical simulation
    v, f = lazy_cage(v_fine, f_fine, num_faces=num_faces, grid_size=256)

    print("created lazy_cage")

    # Tetrahedralize
    tgen = tetgen.TetGen(v, f)
    nodes, elements = tgen.tetrahedralize(minratio=1.5)

    print("created tetrahedralize")

    # Initialize fracture mode class
    modes = fracture.fracture_modes(nodes, elements)

    print("created modes")

    # Set parameters for call to fracture modes
    params = fracture.fracture_modes_parameters(num_modes=20, verbose=False, d=3)

    print("set params")

    # Compute fracture modes
    modes.compute_modes(parameters=params)

    print("inserted params")

    # We need to precompute some stuff that we will only need to do once
    modes.impact_precomputation(v_fine=v_fine, f_fine=f_fine)
    print(len(modes.fine_vertices))
    print(len(v))
    print("precomputed impacts")

    # Optionally, you can use this to save each mode's segmentation in the current directory
    # modes.write_segmented_modes(output_folder + "/output_modes", pieces=True)
    return modes, v, f


def calculateUDF(modes, mesh):
    print(modes)

# generates the fracture modes and pickles all necessary data for later computation
def generate_multiple_fracture_modes(modes, v, f, mesh, config,
                                     volume_constraint=(1 / 50)):
    B, FI = igl.random_points_on_mesh(1000 * config.num_impacts, v, f)
    FI[FI >= len(f)] = len(f) - 1  # fixes bug inside pyigl library which also includes indices equal to len(faces)
    B = np.vstack((B[:, 0], B[:, 0], B[:, 0], B[:, 1], B[:, 1], B[:, 1], B[:, 2], B[:, 2], B[:, 2])).T
    P = B[:, 0:3] * v[f[FI, 0], :] + B[:, 3:6] * v[f[FI, 1], :] + B[:, 6:9] * v[f[FI, 2], :]
    sigmas = np.random.rand(1000 * config.num_impacts) * 1000

    vols = igl.volume(modes.vertices, modes.elements)
    total_vol = np.sum(vols)

    # Loop to generate many possible fractures
    all_labels = np.zeros((modes.precomputed_num_pieces, config.num_impacts), dtype=int)
    running_num = 0
    num_tries = 0
    for i in range(P.shape[0]):
        num_tries += 1
        contact_point = P[i, :]
        direction = mesh.triangle_normals[FI[i]]
        # direction =  -np.copy(contact_point) / np.linalg.norm(np.copy(contact_point))
        modes.impact_projection(contact_point=contact_point, direction=direction, threshold=sigmas[i],
                                num_modes_used=20)
        min_volume = volume_constraint * total_vol / modes.n_pieces_after_impact
        current_min_volume = total_vol
        for j in range(modes.n_pieces_after_impact):
            current_min_volume = min(current_min_volume, np.sum(vols[modes.tet_labels_after_impact == j]))
        valid_volume = (current_min_volume >= min_volume)
        # if verbose:
        #     print("Impact simulation: ",round(t401-t400,3),"seconds.")
        new = not (modes.piece_labels_after_impact.tolist() in all_labels.T.tolist())
        # print(modes.piece_labels_after_impact.tolist() in all_labels.T.tolist())
        # print(modes.n_pieces_after_impact > 1, modes.n_pieces_after_impact < 100, new, valid_volume)
        if 1 < modes.n_pieces_after_impact < 100 and new and valid_volume:
            # write_to_file(modes, output_folder=category_name, dataset_name=dataset_name, index=running_num,
            #               contact_point=contact_point, direction=direction, mesh=mesh, v=v, f=f)
            # udf = calculateUDF(modes = modes, mesh=mesh)
            config.v=v
            config.f=f
            # config.mesh=mesh
            print(type(v))
            print(type(f))
            config.contact_point = contact_point
            config.direction = direction
            config.modes = modes
            with open("pickled_modes/" + str(running_num) + "_data.pkl", "wb") as file:
                pickle.dump(config, file)

            running_num += 1
            print("running_num: ", running_num)
            print("after this amount of tries: ", num_tries)
            num_tries = 0

        if running_num >= config.num_impacts:
            break

    o3d.visualization.draw_geometries([mesh])

