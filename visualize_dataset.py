import open3d
import open3d as o3d
import numpy as np


index = 0

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

def get_pointcloud_from_file(index, datasetName="chair"):
    filename_points = str(index) + ".pcd"
    filename_labels = str(index) + ".seg"

    points = []
    for line in open("dataset/" + datasetName + "/points/" + filename_points):
        points.append(tuple(map(lambda x: float(x), line.split(" "))))
    normalization_direction = points[0] - np.mean(points, axis=0)
    # print("Normalization direction: ", normalization_direction)
    points = points - np.mean(points, axis=0)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    # estimate normals
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(k=30)

    colours = []
    labels = []
    for line in open("dataset/" + datasetName + "/points_label/" + filename_labels):
        labels.append(int(line))

    colors_list = get_colours_list()
    # colours = np.repeat((labels / np.max(labels)), 3).reshape(-1, 3)
    colours = colors_list[labels]
    pcd.colors = o3d.utility.Vector3dVector(colours)

    return pcd, normalization_direction

def get_impulse_from_file(index, datasetName="bunny"):
    filename_impulses = str(index) + ".imp"
    impulse_info = None
    for line in open("dataset/" + datasetName + "/impulse_info/" + filename_impulses):
        impulse_info = list(map(lambda x: float(x), line.split(" ")))
    location = [impulse_info[0], impulse_info[1], impulse_info[2]]
    direction = [impulse_info[3], impulse_info[4], impulse_info[5]]

    print(location)
    print(direction)
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=.02, resolution=20)
    sphere.translate(location)

    return sphere

def get_mesh_from_file(index, scaled_mesh = False, datasetName = "bunny"):
    filename_mesh = str(index) + ".mesh"
    vertices = []
    faces = []
    done_with_vertices = False
    for line in open("dataset/" + datasetName + "/" + ("scaled_" if scaled_mesh else "")+ "mesh/" + filename_mesh):
        if line == "\n":
            done_with_vertices = True
            continue
        if not done_with_vertices:
            vertices.append(list(map(lambda x: float(x), line.split(" "))))
        else:
            faces.append(list(map(lambda x: int(x), line.split(" "))))

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()
    return mesh




def load_new_model(vis):
    global index
    vis.clear_geometries()
    print("Starting the visualization of the dataset")
    pcd, norm_direction = get_pointcloud_from_file(index)
    sphere = get_impulse_from_file(index)
    mesh = get_mesh_from_file(index)
    scaled_mesh = get_mesh_from_file(index, scaled_mesh=True)
    print("Retrieved pointcloud")
    print(index)
    vis.add_geometry(pcd)
    vis.add_geometry(sphere)
    # vis.add_geometry(mesh)
    # vis.add_geometry(scaled_mesh)

    view_ctl = vis.get_view_control()
    view_ctl.set_zoom(2)

def plus_one(vis):
    global index
    index+=1
    print(index)
    load_new_model(vis)

def minus_one(vis):
    global index
    index-=1
    print(index)
    load_new_model(vis)

if __name__ == "__main__":
    print("Starting the visualization of the dataset")
    # global index

    print("Retrieved pointcloud")
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    load_new_model(vis)
    vis.register_key_callback(ord("1"), minus_one)
    vis.register_key_callback(ord("2"), plus_one)
    vis.run()
    vis.destroy_window()

