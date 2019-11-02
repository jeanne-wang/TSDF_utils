import os
import glob
import argparse
import numpy as np
import plyfile
from observation_volume_from_2D_cameras import ObservedVolume
from quaternion_util import quat_from_two_vectors, quat_rotate_vector

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames_2d_path", required=True)
    parser.add_argument("--scene_path", required=True)
    parser.add_argument("--viewport_height", type=int, default=480)
    parser.add_argument("--viewport_width", type=int, default=640)
    return parser.parse_args()


def main():

    args = parse_args()

    mesh = plyfile.PlyData().read(args.scene_path)
    ## load vertex
    v=np.array([list(x) for x in mesh.elements[0]])
    vertex_coords=np.ascontiguousarray(v[:,:3])

    num_vertex = vertex_coords.shape[0]
    print("There are #{} vertex in the mesh".format(num_vertex))

    ## rotate the scene to have Z-gravity
    negative_unit_z = np.array([0,0,-1], dtype=np.float32)
    negative_unit_y = np.array([0,-1,0], dtype=np.float32)
    rotation = quat_from_two_vectors(negative_unit_z, negative_unit_y)


    rotated_points = []
    for i in range(num_vertex):
        point = vertex_coords[i,:]
        rotated_point=quat_rotate_vector(rotation, point)
        rotated_points.append(rotated_point)

    rotated_points = np.asarray(rotated_points, dtype=np.float32)

    observed_volume = ObservedVolume(args.viewport_height, args.viewport_width, rotated_points)

    
    camera_paths = sorted(glob.glob(os.path.join(args.frames_2d_path, "frame*.camera.npz")))
    for i, camera_path in enumerate(camera_paths):

        frame_id = int(os.path.basename(camera_path)[6:11])
        print("Processing frame {} [{}/{}]".format(
             frame_id, i + 1, len(camera_paths)))

        depth_map_path = os.path.join(args.frames_2d_path, "frame-{:05d}.depth.npz".format(frame_id))
        
        camera = np.load(camera_path)
        proj_matrix = camera['projection_matrix']
        cam_matrix = camera['camera_matrix']
        transform_matrix =  np.matmul(proj_matrix, cam_matrix)

        depth_map = np.load(depth_map_path)['depth']
 

        
        observed_volume.fuse(transform_matrix, depth_map)



    observed_volume_flag = observed_volume.get_volume()
    num_observed_vertex = np.sum(observed_volume_flag)
    coverage = num_observed_vertex/num_vertex
    print("There are {} covered points among the {} points in the scene".format(num_observed_vertex, num_vertex))
    print("The coverage of the cameras set is {}".format(coverage))

    




if __name__ == "__main__":
    main()
