import open3d as o3d
import numpy as np
import generate_UDF_fracture



def extract_mesh_from_obj_file(mesh_filename):
    mesh = o3d.io.read_triangle_mesh(mesh_filename, True)
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()

    # o3d.visualization.draw_geometries([mesh])
    return mesh



class Config:
    def __init__(self, choice, size=100):
        if True:
            self.dataset_size = None
            self.category_name = None
            self.dataset_name = None
            self.mesh_aligning_needed = None
            self.mesh_extraction_needed = None
            self.output_path = None
            self.mesh_filename = None
            self.filename = None
            self.pcd_downsample_needed = None
            self.down_sampling_method = None
            self.BUNNY = 0
            self.CHAIR = 1
            self.VASE = 2
            self.BUNNY_NON_GEODESIC = 3

        # Initialize configuration based on choice
        if choice == self.BUNNY:
            self.initialize_bunny(size)
        elif choice == self.CHAIR:
            self.initialize_chair(size)
        elif choice == self.VASE:
            self.initialize_vase(size)
        elif choice == self.BUNNY_NON_GEODESIC:
            self.initialize_bunny_non_geodesic(size)
        else:
            raise ValueError("Invalid choice. Must be either BUNNY or CHAIR.")

    def initialize_bunny(self, size):
        self.mesh_filename = "./data/bunny_oded.obj"
        self.output_path = "./results"
        self.dataset_name = "dataset"
        self.category_name = "bunny"
        self.num_impacts = size

    def initialize_bunny_non_geodesic(self, size):
        self.mesh_filename = "./data/bunny_oded.obj"
        self.output_path = "./results"
        self.dataset_name = "dataset"
        self.category_name = "bunny_non_geodesic"
        self.num_impacts = size


    def initialize_chair(self, size):
        self.mesh_filename = "data/chair/chair.obj"
        self.output_path = "./results"
        self.dataset_name = "dataset"
        self.category_name = "chair"
        self.num_impacts = size
        # self.pcd_downsample_needed = False
        # self.mesh_extraction_needed = False
        # self.mesh_aligning_needed = False
        # self.down_sampling_method = "sample_x"
        # self.filename = "/home/lukasz/Documents/thesis_pointcloud/data/chair/point_cloud.ply"

    def initialize_vase(self, size):
        self.mesh_filename = "data/vase.obj"
        self.output_path = "./results"
        self.dataset_name = "dataset"
        self.category_name = "vase"
        self.num_impacts = size

if __name__ == "__main__":
    BUNNY = 0
    CHAIR = 1
    VASE = 2
    BUNNY_NON_GEODESIC = 3
    choice = BUNNY_NON_GEODESIC
    config = Config(choice, size=1000)


    mesh = extract_mesh_from_obj_file(config.mesh_filename)

    print(mesh.get_center())
    # o3d.visualization.draw_geometries([mesh])



    # sys.stdout = io.StringIO()
    print("Start creating Fracture modes")
    modes, v, f = generate_UDF_fracture.create_modes(mesh.vertices, mesh.triangles)
    print("Done creating Fracture modes")
    # sys.stdout = sys.__stdout__
    print("mesh.vertices = ", len(mesh.vertices))
    # print("mesh.faces = ", len(mesh.triangles))
    # print("pcd.points = ", len(pcd.points))
    print("modes.fine_vertices = ", len(modes.fine_vertices))
    # print("modes.fine_faces = ", len(modes.fine_triangles))
    print("done generating modes")

    print(mesh.get_center())
    # o3d.visualization.draw_geometries([mesh, ])

    generate_UDF_fracture.generate_multiple_fracture_modes(modes, v=v, f=f,mesh=mesh, config=config)


