import time

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





def visualize_labeled_mesh(mesh, labels, filename="screenshot.png"):
    colors = get_colours_list()[labels]
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    # o3d.visualization.draw_geometries([mesh])
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1840, height=849)



    # Add mesh to visualizer
    vis.add_geometry(mesh)
    # if contact_point is not None:
    #     sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02, resolution=5)
    #     sphere.compute_vertex_normals()
    #
    #     sphere.translate(contact_point)
    #     sphere.paint_uniform_color([1.0, 0.0, 0.0])
    #     vis.add_geometry(sphere)

    # Configure to disable all light effects
    opt = vis.get_render_option()
    opt.point_size = 1.0
    opt.background_color = np.array([1.0, 1.0, 1.0])  # White background
    opt.show_coordinate_frame = False
    opt.light_on = False
    opt.mesh_show_wireframe = True

    # Set camera parameters if needed
    ctr = vis.get_view_control()
    parameters = o3d.io.read_pinhole_camera_parameters("ScreenCamera.json")
    ctr.convert_from_pinhole_camera_parameters(parameters)


    # Start the visualizer
    vis.run()

    # Save a screenshot
    vis.capture_screen_image(filename, do_render=True)


    # Destroy the visualizer window
    vis.destroy_window()



def visualize_mesh_and_fine_mesh(mesh, modes):
    fine_mesh = retrieve_fine_mesh_from_fracture_modes(mesh, modes)

    # visualize mesh and fine_mesh
    o3d.visualization.draw_geometries([fine_mesh, mesh])


def visualize_UDF(mesh, distances, use_sqrt_ratios=True, contact_point=None):
    """
    Takes a mesh and a set of distances per vertex and Visualizes these on the mesh.
    :param mesh: mesh to visualize
    :param distances: list of distances per vertex
    :param use_sqrt_ratios: Whether to use square roots to exaggerate differences in distance close to contact points
    :return: Nothing, opens a window with the visualization
    """
    if use_sqrt_ratios:
        ratios = np.sqrt(distances / np.max(distances))
    else:
        ratios = distances / np.max(distances)
    colors = 1 - (ratios[:, None] * np.array([1.0, 1.0, 1.0]))
    print(ratios)
    print(colors)

    mesh.compute_vertex_normals()
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)



    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1840, height=849)



    # Add mesh to visualizer
    vis.add_geometry(mesh)
    # if contact_point is not None:
    #     sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02, resolution=5)
    #     sphere.compute_vertex_normals()
    #
    #     sphere.translate(contact_point)
    #     sphere.paint_uniform_color([1.0, 0.0, 0.0])
    #     vis.add_geometry(sphere)

    # Configure to disable all light effects
    opt = vis.get_render_option()
    opt.point_size = 1.0
    opt.background_color = np.array([1.0, 1.0, 1.0])  # White background
    opt.show_coordinate_frame = False
    opt.light_on = False
    opt.mesh_show_wireframe = True

    # Set camera parameters if needed
    ctr = vis.get_view_control()
    parameters = o3d.io.read_pinhole_camera_parameters("ScreenCamera.json")
    ctr.convert_from_pinhole_camera_parameters(parameters)


    # Start the visualizer
    vis.run()

    # Save a screenshot
    vis.capture_screen_image("screenshot.png", do_render=True)


    # Destroy the visualizer window
    vis.destroy_window()


def visualize_mesh_with_contact_point(mesh, contact_point):
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02, resolution=5)
    sphere.compute_vertex_normals()

    sphere.translate(contact_point)
    sphere.paint_uniform_color([1.0, 0.0, 0.0])
    # o3d.visualization.draw_geometries([sphere, mesh])
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1840, height=849)



    # Add mesh to visualizer
    vis.add_geometry(mesh)
    vis.add_geometry(sphere)

    # Configure to disable all light effects
    opt = vis.get_render_option()
    opt.point_size = 1.0
    opt.background_color = np.array([1.0, 1.0, 1.0])  # White background
    opt.show_coordinate_frame = False
    opt.light_on = False
    opt.mesh_show_wireframe = True

    # Set camera parameters if needed
    ctr = vis.get_view_control()
    parameters = o3d.io.read_pinhole_camera_parameters("ScreenCamera.json")
    ctr.convert_from_pinhole_camera_parameters(parameters)


    # Start the visualizer
    vis.run()

    # Save a screenshot
    vis.capture_screen_image("screenshot.png", do_render=True)


    # Destroy the visualizer window
    vis.destroy_window()