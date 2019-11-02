import os
import glob
import argparse
import numpy as np
import plyfile
import json
from freespace_volume_from_2D_cameras import FreespaceVolume

selected_class_ids = [93, 40, 18, 7, 20, 76, 80, 37, 97, 71, 98, 12, 47, 67, 29, 31, 94]


remapper=np.ones(150)*(-100)
for i,x in enumerate(selected_class_ids):
    remapper[x]=i

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_path", required=True)
    parser.add_argument("--frames_2d_path", required=True)
    parser.add_argument("--scene_sampled_point_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--truncated_distance", type=float, default=0.1)
    return parser.parse_args()


def main():
    args = parse_args()

    # Load the extrinsic calibration between color and depth sensor.
    info_path = glob.glob(os.path.join(args.scene_path, "*.txt"))
    color_to_depth_T = None
    assert len(info_path) == 1
    with open(info_path[0], "r") as fid:
        for line in fid:
            if line.startswith("colorToDepthExtrinsics"):
                color_to_depth_T = \
                    np.array(list(map(float, line.split("=")[1].split())))
                color_to_depth_T = color_to_depth_T.reshape(4, 4)
    if color_to_depth_T is None:
        color_to_depth_T = np.eye(4)

    # Load the intrinsic calibration parameters.
    with open(os.path.join(args.scene_path, "sensor/_info.txt"), "r") as fid:
        for line in fid:
            if line.startswith("m_calibrationColorIntrinsic"):
                label_K = np.array(list(map(float, line.split("=")[1].split())))
                label_K = label_K.reshape(4, 4)[:3, :3]
            elif line.startswith("m_calibrationDepthIntrinsic"):
                depth_K = np.array(list(map(float, line.split("=")[1].split())))
                depth_K = depth_K.reshape(4, 4)[:3, :3]
            elif line.startswith("m_calibrationColorExtrinsic") or \
                    line.startswith("m_calibrationDepthExtrinsic"):
                assert line.split("=")[1].strip() \
                       == "1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1"

    ############ observation fusion ####################

    d = np.load(args.scene_sampled_point_file, allow_pickle=True).item()
    coords = d['point_coordinate']
    distance_to_mesh = d['distance_to_mesh']
    nearest_instance_index = d['nearest_instance_index']
    print("There are {} uniformly sampled points.".format(coords.shape[0]))

    ##################################


    observed_volume = FreespaceVolume(coords)

    
    camera_paths = sorted(glob.glob(os.path.join(args.frames_2d_path, "frame*.camera.npz")))
    for i, camera_path in enumerate(camera_paths):

        frame_id = int(os.path.basename(camera_path)[6:11])
        print("Processing frame {} [{}/{}]".format(
             frame_id, i + 1, len(camera_paths)))

        
        

        camera = np.load(camera_path)
        proj_matrix = camera['projection_matrix']
        cam_matrix = camera['camera_matrix']
        transform_matrix =  np.matmul(proj_matrix, cam_matrix)

        depth_map_path = os.path.join(args.frames_2d_path, "frame-{:05d}.depth.npz".format(frame_id))
        depth_map = np.load(depth_map_path)['depth']

        
        observed_volume.fuse(transform_matrix, depth_map)



    observed_volume_flag = observed_volume.get_volume()

    valid_coords = []
    valid_distance_to_mesh = []
    valid_tsdf = []
    valid_label = []
    for i in range(coords.shape[0]):

        ## freeespace
        if observed_volume_flag[i] == 1:

            valid_coords.append(coords[i,:])
            valid_distance_to_mesh.append(distance_to_mesh[i])
            valid_tsdf.append(min(distance_to_mesh[i], args.truncated_distance))
            valid_label.append(-100) ## freespace also labeled as -100, will be ignored during semantic loss computation

        elif distance_to_mesh[i] <= args.truncated_distance:
            
            valid_coords.append(coords[i,:])
            valid_distance_to_mesh.append(distance_to_mesh[i])
            valid_tsdf.append(-distance_to_mesh[i])
            label = id_to_label[nearest_instance_index[i]]
            if label < 0:
                label = 149
            valid_label.append(remapper[label])

    valid_coords=np.asarray(valid_coords)
    valid_distance_to_mesh = np.asarray(valid_distance_to_mesh)
    valid_tsdf = np.asarray(valid_tsdf, dtype=np.float32)
    valid_label = np.asarray(valid_label, dtype=np.int32)
    

    np.savez(args.output_file, coords=valid_coords,distance=valid_distance_to_mesh, tsdf=valid_tsdf, label=valid_label)

if __name__ == "__main__":
    main()