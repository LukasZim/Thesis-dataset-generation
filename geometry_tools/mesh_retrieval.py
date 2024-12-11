import open3d as o3d

from geometry_tools.scaling import scale_mesh_to_mesh_inplace


def retrieve_fine_mesh_from_fracture_modes(mesh, modes):
    """
    Takes a fracture_modes object, retrieves the vertices and faces, and rescales the created mesh to the target.
    :param mesh: The ground truth mesh
    :param modes: The fracture_modes object
    :return: The scaled mesh retrieved from the fracture_modes object
    """
    fine_mesh = create_mesh_from_faces_and_vertices(modes.fine_triangles, modes.fine_vertices)
    scale_mesh_to_mesh_inplace(fine_mesh, mesh)
    return fine_mesh

def create_mesh_from_faces_and_vertices(faces, vertices):
    # create empty variable for "fine_mesh"
    fine_mesh = o3d.geometry.TriangleMesh()
    # retrieve and set fine vertices
    fine_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    # retrieve and set fine faces
    fine_mesh.triangles = o3d.utility.Vector3iVector(faces)
    fine_mesh.compute_vertex_normals()
    return fine_mesh