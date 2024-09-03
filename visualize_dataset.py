import open3d
import open3d as o3d
import numpy as np


index = 1

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

def get_pointcloud_from_file(index, datasetName="bunny"):
    filename_points = str(index) + ".pcd"
    filename_labels = str(index) + ".seg"

    points = []
    for line in open("dataset/" + datasetName + "/points/" + filename_points):
        points.append(tuple(map(lambda x: float(x), line.split(" "))))
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

    return pcd

def get_impulse_from_file(index, datasetName="bunny"):
    filename_impulses = str(index) + ".imp"
    impulse_info = None
    for line in open("dataset/" + datasetName + "/impulse_info/" + filename_impulses):
        impulse_info = list(map(lambda x: float(x), line.split(" ")))
    location = [impulse_info[0], impulse_info[1], impulse_info[2]]
    arrow_direction = [impulse_info[3], impulse_info[4], impulse_info[5]]
    location = np.array(location) - np.array(arrow_direction)
    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=0.005,
        cone_radius=0.01,
        cylinder_height=0.05,
        cone_height=0.02
    )
    # arrow.rotate(arrow.get_rotation_matrix_from_xyz(np.arctan2(arrow_direction[1], arrow_direction[0])))
    arrow.translate(location)
    return arrow


def load_new_model(vis):
    global index
    vis.clear_geometries()
    print("Starting the visualization of the dataset")
    pcd = get_pointcloud_from_file(index)
    impulse_arrow = get_impulse_from_file(index)
    print("Retrieved pointcloud")
    vis.add_geometry(pcd)

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
    pcd = get_pointcloud_from_file(index)
    impulse_arrow = get_impulse_from_file(index)
    print("Retrieved pointcloud")
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.register_key_callback(ord("1"), minus_one)
    vis.register_key_callback(ord("2"), plus_one)
    vis.run()
    vis.destroy_window()

