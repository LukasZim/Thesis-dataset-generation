import open3d as o3d
import numpy as np


def read_custom_mesh(file_path):
    print("THIS SHOULD ONLY BE USED FOR THE DEPRICATED .mesh FILES USED IN OLDER VERSIONS OF THE DATASET GENERATION")
    vertices = []
    faces = []

    with open(file_path, 'r') as file:
        # Read vertices
        for line in file:
            line = line.strip()
            if not line:  # Stop reading at the blank line
                break
            vertices.append(list(map(float, line.split())))

        # Read faces after the blank line
        for line in file:
            line = line.strip()
            if line:  # Only process non-empty lines
                faces.append(list(map(int, line.split())))

    # Convert lists to numpy arrays for Open3D
    vertices = np.array(vertices)
    faces = np.array(faces)  # Convert to 0-based index for Open3D

    # Create a TriangleMesh from vertices and faces
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)

    return mesh


# Specify the input file and output file paths
input_file = "../dataset/chair/mesh/0.mesh"
output_file = "./output_mesh.obj"

# Read the custom mesh format and create a TriangleMesh
mesh = read_custom_mesh(input_file)
o3d.visualization.draw_geometries([mesh])
# Save the mesh as an .obj file
o3d.io.write_triangle_mesh(output_file, mesh)
print(f"Mesh successfully saved to {output_file}")