import os
import argparse
import numpy as np
import plyfile
from quaternion_util import quat_from_two_vectors, quat_rotate_vector


# Save 3D mesh to a polygon .ply file
def meshwrite(filename,verts,faces,colors):

    # Write header
    ply_file = open(filename,'w')
    ply_file.write("ply\n")
    ply_file.write("format ascii 1.0\n")
    ply_file.write("element vertex %d\n"%(verts.shape[0]))
    ply_file.write("property float x\n")
    ply_file.write("property float y\n")
    ply_file.write("property float z\n")
    ply_file.write("property uchar red\n")
    ply_file.write("property uchar green\n")
    ply_file.write("property uchar blue\n")
    ply_file.write("element face %d\n"%(faces.shape[0]))
    ply_file.write("property list uchar int vertex_index\n")
    ply_file.write("end_header\n")

    # Write vertex list
    for i in range(verts.shape[0]):
        ply_file.write("%f %f %f %d %d %d\n"%(verts[i,0],verts[i,1],verts[i,2], colors[i,0],colors[i,1],colors[i,2]))
    
    # Write face list
    for i in range(faces.shape[0]):
        ply_file.write("4 %d %d %d %d\n"%(faces[i,0],faces[i,1],faces[i,2], faces[i,3]))

    ply_file.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_path", required=True)
    parser.add_argument("--output_path", required=True)
    return parser.parse_args()


def main():

    args = parse_args()

    mesh = plyfile.PlyData().read(os.path.join(args.scene_path, "mesh.ply"))

    ## load vertex
    v=np.array([list(x) for x in mesh.elements[0]])
    vertex_coords=np.ascontiguousarray(v[:,:3])
    vertex_normals=np.ascontiguousarray(v[:,3:6])
    vertex_colors=np.ascontiguousarray(v[:,6:9])


    num_vertex = vertex_coords.shape[0]
    print("There are #{} vertex in the mesh".format(num_vertex))
    
    ## load face 
    face_vertex_indices = mesh.elements[1]['vertex_indices']
    face_vertex_indices = np.stack(face_vertex_indices, axis = 0)


    ## rotate the scene to have Z-gravity
    negative_unit_z = np.array([0,0,-1], dtype=np.float32)
    negative_unit_y = np.array([0,-1,0], dtype=np.float32)
    rotation = quat_from_two_vectors(negative_unit_z, negative_unit_y)


    rotated_points = []
    for i in range(num_vertex):
        point = vertex_coords[i,:]
        rotated_point=quat_rotate_vector(rotation, point)
        rotated_points.append(rotated_point)

    rotated_points = np.asarray(rotated_points)

    output_name = os.path.join(args.output_path, 'rotated_mesh.ply')
    meshwrite(output_name, rotated_points, face_vertex_indices, vertex_colors)

    # new_x = list(rotated_points[:,0])
    # new_y = list(rotated_points[:,1])
    # new_z = list(rotated_points[:,2])

    # ###



    # ###


    # # Compute the bounding box of the rotated ground truth.
    # minx = min(new_x)
    # maxx=max(new_x)

    # miny = min(new_y)
    # maxy=max(new_y)

    # minz = min(new_z)
    # maxz = max(new_z)

    # diffx = maxx - minx
    # diffy = maxy - miny
    # diffz = maxz - minz



    

if __name__ == "__main__":
    main()
