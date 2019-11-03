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
    nearest_face_label = d['nearest_face_label']
    print("There are {} uniformly sampled points.".format(coords.shape[0]))

    ##################################


    observed_volume = FreespaceVolume(coords)
    pose_paths = sorted(glob.glob(os.path.join(args.scene_path, "sensor/*.pose.txt")))
    for i, pose_path in enumerate(pose_paths):

        frame_id = int(os.path.basename(pose_path)[6:-9])
        print("Processing frame {} [{}/{}]".format(
             frame_id, i + 1, len(pose_paths)))

        depth_map_path = os.path.join(args.scene_path,
                "sensor/frame-{:06d}.depth.pgm".format(frame_id))
        assert os.path.exists(depth_map_path)

        pose = np.loadtxt(pose_path)

        proj_matrix = np.linalg.inv(pose)
        depth_proj_matrix = \
                np.dot(depth_K, np.dot(color_to_depth_T, proj_matrix)[:3])

        depth_proj_matrix = depth_proj_matrix.astype(np.float32)
        depth_map = skimage.io.imread(depth_map_path)
        depth_map = depth_map.astype(np.float32) / 1000

        observed_volume.fuse(depth_proj_matrix, depth_map)



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
            valid_label.append(remapper[nearest_face_label[i]])
            

    valid_coords=np.asarray(valid_coords)
    valid_distance_to_mesh = np.asarray(valid_distance_to_mesh)
    valid_tsdf = np.asarray(valid_tsdf, dtype=np.float32)
    valid_label = np.asarray(valid_label, dtype=np.int32)
    

    np.savez(args.output_file, coords=valid_coords,distance=valid_distance_to_mesh, tsdf=valid_tsdf, label=valid_label)

if __name__ == "__main__":
    main()