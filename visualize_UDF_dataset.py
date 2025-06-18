import polyscope as ps
import os
import pandas as pd
import numpy as np

from geometry_tools.mesh_retrieval import load_mesh_from_file

if __name__ == "__main__":
    ps.set_window_size(1920, 1080)
    ps.init()
    ps.set_window_size(1920, 1080)

    root_path = "./datasets/bunny"
    object_path = os.path.join(root_path, "bunny.obj")
    index = 71
    udf_path = os.path.join(root_path, f"{index}.pkl")
    impulse_path = os.path.join(root_path, f"{index}_impulse.pkl")

    # load mesh from file
    mesh = load_mesh_from_file(object_path)
    faces = np.asarray(mesh.triangles)

    # load data from pkl files
    df = pd.read_pickle(udf_path)

    vertices = df.drop(["distance", 'label', 'edge_labels'], axis=1).values
    udf = df["distance"].values

    df = pd.read_pickle(impulse_path)
    impulse = df.values[0]
    print(impulse)

    # load data into polyscope
    ps.register_surface_mesh("mesh", vertices, faces, smooth_shade=True)
    ps.get_surface_mesh("mesh").add_scalar_quantity("UDF", udf, defined_on="vertices", enabled=True)

    impact_point = impulse[:3].reshape(-1, 3)
    ps.register_point_cloud("impact point", impact_point, enabled=True)

    # maybe do some transofmatyions idk

    # try to retrieve the simulation mesh and visualize this

    ps.show()