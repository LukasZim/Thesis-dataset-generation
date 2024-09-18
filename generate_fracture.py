import numpy as np
import igl
import tetgen

from fracture_modes import fracture_utility as fracture
# import fracture_utility as fracture
from gpytoolbox.copyleft import lazy_cage
import gpytoolbox
import os

from visualize_las import find_scipy_mapping


# # This is the "fine mesh", i.e. the mesh we use for rendering
# v_fine, f_fine = igl.read_triangle_mesh("data/bunny_oded.obj")
# v_fine = gpytoolbox.normalize_points(v_fine)
#
# # This is the "cage mesh", i.e. the coarser mesh that we will tetrahedralize and use for the physical simulation
# v, f = lazy_cage(v_fine, f_fine, num_faces=100)
#
# # Tetrahedralize
# tgen = tetgen.TetGen(v,f)
# nodes, elements = tgen.tetrahedralize()
#
# # Initialize fracture mode class
# modes = fracture.fracture_modes(nodes, elements)
#
# # Set parameters for call to fracture modes
# params = fracture.fracture_modes_parameters(num_modes=10,verbose=True,d=3)
#
# # Compute fracture modes
# modes.compute_modes(parameters=params)
#
# # We need to precompute some stuff that we will only need to do once
# modes.impact_precomputation(v_fine=v_fine,f_fine=f_fine)
#
# # Optionally, you can use this to save each mode's segmentation in the current directory
# modes.write_segmented_modes("output_modes")
# contact_point = nodes[1,:]
# direction = np.array([1.0, 0.0, 0.0])
#
# # First projection, this should be fast
# modes.impact_projection(contact_point=contact_point, direction=direction)
#
# # Second projection, this should be fast
# new_contact_point = nodes[5,:]
# modes.impact_projection(contact_point=new_contact_point, direction=direction)
#
# # Write segmented output to obj
# modes.write_segmented_output("output.obj")

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


def create_modes(v_fine, f_fine, num_faces=400, output_folder="./results"):
    v_fine = np.asarray(v_fine)
    f_fine = np.asarray(f_fine)
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


def generate_multiple_fractures(modes, num_impacts, v, f, category_name, dataset_name, pcd, mesh, volume_constraint=(1 / 50)):
    B, FI = igl.random_points_on_mesh(1000 * num_impacts, v, f)
    FI[FI >= len(f)] = len(f) - 1  # fixes bug inside pyigl library which also includes indices equal to len(faces)
    B = np.vstack((B[:, 0], B[:, 0], B[:, 0], B[:, 1], B[:, 1], B[:, 1], B[:, 2], B[:, 2], B[:, 2])).T
    P = B[:, 0:3] * v[f[FI, 0], :] + B[:, 3:6] * v[f[FI, 1], :] + B[:, 6:9] * v[f[FI, 2], :]
    sigmas = np.random.rand(1000 * num_impacts) * 1000

    vols = igl.volume(modes.vertices, modes.elements)
    total_vol = np.sum(vols)

    # Loop to generate many possible fractures
    all_labels = np.zeros((modes.precomputed_num_pieces, num_impacts), dtype=int)
    running_num = 0
    num_tries = 0
    for i in range(P.shape[0]):
        num_tries+=1
        contact_point = P[i, :]
        direction = mesh.triangle_normals[FI[i]]
        # direction =  -np.copy(contact_point) / np.linalg.norm(np.copy(contact_point))
        modes.impact_projection(contact_point=contact_point, direction=direction, threshold=sigmas[i],
                                num_modes_used=30)
        min_volume = volume_constraint * total_vol / (modes.n_pieces_after_impact)
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
            write_to_file(modes, output_folder=category_name, dataset_name=dataset_name, index=running_num, pcd=pcd,
                          contact_point=contact_point, direction=direction, mesh=mesh)
            running_num += 1
            print("running_num: ", running_num)
            print("after this amount of tries: ", num_tries)
            num_tries = 0
            # all_labels[:, running_num] = modes.piece_labels_after_impact
            # write_output_name = os.path.join(output_dir, "fractured_") + str(running_num)
            # running_num = running_num + 1
            # if not os.path.exists(write_output_name):
            #     os.mkdir(write_output_name)
            # if compressed:
            #     modes.write_segmented_output_compressed(filename=write_output_name)
            # else:
            #     modes.write_segmented_output(filename=write_output_name, pieces=True)
        if running_num >= num_impacts:
            break



def write_to_file(modes, output_folder, dataset_name, index, pcd, contact_point, direction, mesh):
    if not os.path.exists(dataset_name):
        os.makedirs(dataset_name)
    # if not os.path.exists(os.path.join(dataset_name, output_folder)):
    os.makedirs(os.path.join(dataset_name, output_folder), exist_ok=True)
    os.makedirs(os.path.join(dataset_name, output_folder, "points"), exist_ok=True)
    os.makedirs(os.path.join(dataset_name, output_folder, "points_label"), exist_ok=True)
    os.makedirs(os.path.join(dataset_name, output_folder, "impulse_info"), exist_ok=True)
    os.makedirs(os.path.join(dataset_name, output_folder, "mesh"), exist_ok=True)
    os.makedirs(os.path.join(dataset_name, output_folder, "scaled_mesh"), exist_ok=True)

    #retrieve pointcloud
    pointcloud = pcd

    # write pointcloud to .ply file
    pcd_path = os.path.join(dataset_name, output_folder, "points", str(index) + ".pcd")
    with open(pcd_path, "w") as f:
        for point in np.asarray(pcd.points):
            f.write(f"{point[0]} {point[1]} {point[2]}\n")

    #retrieve segmentation
    _, labels, scaled_vertices = find_scipy_mapping(pointcloud, modes, mesh)

    #write segmentation to .seg file
    seg_path = os.path.join(dataset_name, output_folder, "points_label", str(index) + ".seg")
    with open(seg_path, "w") as f:
        for label in labels:
            f.write(f"{label}\n")

    #retrieve impulse info

    impulse_info = np.concatenate((contact_point, direction))

    #write impulse info to file or append it to pointcloud data
    impulse_path = os.path.join(dataset_name, output_folder, "impulse_info", str(index) + ".imp")
    pcd_min = np.min(pcd.points, axis=0)
    pcd_max = np.max(pcd.points, axis=0)
    mesh_min = np.min(modes.fine_vertices, axis=0)
    mesh_max = np.max(modes.fine_vertices, axis=0)
    contact_point = (contact_point - mesh_min) / (mesh_max - mesh_min) * (pcd_max - pcd_min) + pcd_min

    with open(impulse_path, "w") as f:
        # print(impulse_info)
        f.write(f"{contact_point[0]} {contact_point[1]} {contact_point[2]} {impulse_info[3]} {impulse_info[4]} {impulse_info[5]}\n")

    mesh_path = os.path.join(dataset_name, output_folder, "mesh", str(index) + ".mesh")
    with open(mesh_path, "w") as f:
        for vertex in mesh.vertices:
            f.write(f"{vertex[0]} {vertex[1]} {vertex[2]}\n")

        f.write("\n")

        for face in mesh.triangles:
            f.write(f"{face[0]} {face[1]} {face[2]}\n")


    scaled_mesh_path = os.path.join(dataset_name, output_folder, "scaled_mesh", str(index) + ".mesh")
    with open(scaled_mesh_path, "w") as f:
        for vertex in scaled_vertices:
            f.write(f"{vertex[0]} {vertex[1]} {vertex[2]}\n")

        f.write("\n")

        for face in modes.fine_triangles:
            f.write(f"{face[0]} {face[1]} {face[2]}\n")

