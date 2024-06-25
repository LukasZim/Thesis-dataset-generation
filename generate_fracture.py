import numpy as np
import igl
import tetgen

from fracture_modes import fracture_utility as fracture
# import fracture_utility as fracture
from gpytoolbox.copyleft import lazy_cage
import gpytoolbox

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
    modes.write_segmented_modes(output_folder + "/output_modes")
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

    # TODO: figure out what to return at this point
    return modes
