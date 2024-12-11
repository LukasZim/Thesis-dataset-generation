import numpy as np
import open3d as o3d

def scale_pcd_to_mesh(pcd, mesh):
    """
    Scale the point cloud to a mesh's AABB
    :param pcd: The point cloud to scale
    :param mesh: The mesh to target
    :return: scaled point cloud
    """
    min_bound_pcd, max_bound_pcd = np.asarray(pcd.get_min_bound()), np.asarray(pcd.get_max_bound())
    min_bound_mesh, max_bound_mesh = np.asarray(mesh.get_min_bound()), np.asarray(mesh.get_max_bound())

    center_pcd = (min_bound_pcd + max_bound_pcd) / 2
    center_mesh = (min_bound_mesh + max_bound_mesh) / 2
    range1 = max_bound_pcd - min_bound_pcd
    range2 = max_bound_mesh - min_bound_mesh

    # Find scale factor to match the ranges (based on the largest relative axis range)
    scale_factor = range2 / range1  # element-wise scale factor

    # To make scaling uniform, use the largest scale factor among axes
    uniform_scale = max(scale_factor)

    pcd.points = o3d.utility.Vector3dVector((np.asarray(pcd.points) - center_pcd) * uniform_scale + center_mesh)
    return pcd

def scale_mesh_to_pcd(pcd, mesh):
    """
    Scales a mesh to a point cloud's AABB.
    :param pcd: the point cloud to target
    :param mesh: the mesh to scale
    :return: the scaled mesh
    """
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


def scale_mesh_to_mesh_inplace(changing_mesh, scaling_target_mesh):
    """
    Does mesh rescaling in-place to match the target mesh's AABB.

    :param changing_mesh: the mesh that will be changed
    :param scaling_target_mesh: the mesh that will be targeted
    :return: nothing, this is in-place
    """
    # Step 1: Get the bounding boxes of both meshes
    bbox1 = changing_mesh.get_axis_aligned_bounding_box()
    bbox2 = scaling_target_mesh.get_axis_aligned_bounding_box()

    # Step 2: Calculate the extents (dimensions) of each bounding box
    extent1 = bbox1.get_extent()  # Dimensions of mesh1's bounding box (x, y, z)
    extent2 = bbox2.get_extent()  # Dimensions of mesh2's bounding box (x, y, z)

    # Step 3: Calculate the scaling factors to exactly match each dimension
    scaling_factors = extent2 / extent1  # Element-wise division to match each axis exactly

    # Step 4: Center mesh1 vertices around the origin
    vertices = np.asarray(changing_mesh.vertices)
    centroid1 = vertices.mean(axis=0)
    vertices -= centroid1  # Translate to the origin

    # Step 5: Apply non-uniform scaling
    vertices *= scaling_factors

    # Step 6: Translate vertices back to the original center of mesh1
    vertices += centroid1

    # Update mesh1's vertices with the new scaled vertices
    changing_mesh.vertices = o3d.utility.Vector3dVector(vertices)

    # Step 7: Translate mesh1 to match the position of mesh2's bounding box
    translation = bbox2.get_center() - changing_mesh.get_axis_aligned_bounding_box().get_center()
    changing_mesh.translate(translation)