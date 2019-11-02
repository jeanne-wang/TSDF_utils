import os
import argparse
import numpy as np
from pyntcloud import PyntCloud
from quaternion_util import quat_from_two_vectors, quat_rotate_vector



def mkdir_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_path", required=True)
    parser.add_argument("--output_path", required=True)
    return parser.parse_args()


def main():

    args = parse_args()
    mkdir_if_not_exists(args.output_path)

    # Load the groundtruth mesh and convert it to a point cloud.
    groundtruth = PyntCloud.from_file(os.path.join(args.scene_path, "mesh.ply"))

    vert_x = np.asarray(groundtruth.points.x)
    vert_y = np.asarray(groundtruth.points.y)
    vert_z = np.asarray(groundtruth.points.z)
    verts = np.vstack((vert_x, vert_y, vert_z))
    verts = verts.transpose()

    ## rotate the scene to have Z-gravity
    negative_unit_z = np.array([0,0,-1], dtype=np.float32)
    negative_unit_y = np.array([0,-1,0], dtype=np.float32)
    rotation = quat_from_two_vectors(negative_unit_z, negative_unit_y)



    num_vert = verts.shape[0]
    rotated_points = []
    for i in range(num_vert):
        point = verts[i,:]
        rotated_point=quat_rotate_vector(rotation, point)
        rotated_points.append(rotated_point)

    rotated_points = np.asarray(rotated_points)

    new_x = list(rotated_points[:,0])
    new_y = list(rotated_points[:,1])
    new_z = list(rotated_points[:,2])


    # Compute the bounding box of the rotated ground truth.
    minx = min(new_x)
    maxx=max(new_x)

    miny = min(new_y)
    maxy=max(new_y)

    minz = min(new_z)
    maxz = max(new_z)

    diffx = maxx - minx
    diffy = maxy - miny
    diffz = maxz - minz



    # #### rotate the bounding box to have z-gravity ##########
    # # x direction does not change
    # minx = minx
    # maxx = maxx

    # # y direction
    # maxy = miny
    # miny = maxy-diffz

    # # z direction
    # maxz = maxz
    # minz = maxz-diffy

    # # new size
    # diffx = maxx - minx
    # diffy = maxy - miny
    # diffz = maxz - minz

    # Extend the bounding box by a stretching factor.
    
    minx -= 0.05 * diffx
    maxx += 0.05 * diffx
    miny -= 0.05 * diffy
    maxy += 0.05 * diffy
    minz -= 0.05 * diffz
    maxz += 0.05 * diffz

    # Write the bounding box. 
    bbox = np.array(((minx, maxx), (miny, maxy), (minz, maxz)),
                    dtype=np.float32)
    np.savetxt(os.path.join(args.output_path, "bbox.txt"), bbox)

if __name__ == "__main__":
    main()
