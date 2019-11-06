import os
import glob
import argparse
import numpy as np
import plyfile
import json
import skimage.io
import struct
from sample_along_camera_ray_from_depthmap import Sampling
import sys

def write_to_binary(coords, outfile):

    with open(outfile, "wb") as fid:
        version = 1
        fid.write(struct.pack("B", version))

        # Endianness
        endianness = (sys.byteorder == "big")
        fid.write(struct.pack("B", endianness))

        # store sizes of data types written
        uint_size = np.dtype(np.uint32).itemsize
        fid.write(struct.pack("B", uint_size))

        # Store size of array type
        elem_size = coords.dtype.itemsize
        fid.write(struct.pack("I", elem_size))

        # Store size of the array
        data_shape = np.shape(coords)
        fid.write(struct.pack("I", data_shape[0]))
        fid.write(struct.pack("I", data_shape[1]))
    
        fid.write(coords.tobytes())
        fid.close()

    return
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_path", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--num_sample", type=int, default=10000)
    parser.add_argument("--gaussian_variance", type=float, default=0.1)
    return parser.parse_args()


def main():
    args = parse_args()


    ####### Load the extrinsic calibration between color and depth sensor #############
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

    ############Load the intrinsic calibration parameters. ###################
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
    depth_K_inv = np.linalg.inv(depth_K)

    ############ observation fusion ####################

    pose_paths = sorted(glob.glob(os.path.join(args.scene_path, "sensor/*.pose.txt")))
    num_frames = len(pose_paths)
    num_sample_per_frame = int(args.num_sample/num_frames)
    sampling = Sampling(num_frames, num_sample_per_frame, args.gaussian_variance)
    for i, pose_path in enumerate(pose_paths):

        frame_id = int(os.path.basename(pose_path)[6:-9])
        print("Processing frame {} [{}/{}]".format(
             frame_id, i + 1, len(pose_paths)))

        
        depth_map_path = os.path.join(args.scene_path,
                "sensor/frame-{:06d}.depth.pgm".format(frame_id))
        assert os.path.exists(depth_map_path)
        depth_map = skimage.io.imread(depth_map_path)
        depth_map = depth_map.astype(np.float32) / 1000

        pose = np.loadtxt(pose_path)
        depth_extrinsics_matrix_inv = np.dot(pose, np.linalg.inv(color_to_depth_T))[:3]
        depth_extrinsics_matrix_inv = depth_extrinsics_matrix_inv.astype(np.float32)
        
        depth_K_inv = depth_K_inv.astype(np.float32)

        sampling.sample(frame_id, depth_K_inv, 
                        depth_extrinsics_matrix_inv, depth_map)



    pts_along_camera_rays = sampling.get_sampled_points()
    print("There are {} sampled points along camera rays".format(pts_along_camera_rays.shape[0]))
    minx =np.min(pts_along_camera_rays[:,0])
    maxx =np.max(pts_along_camera_rays[:,0])
    
    miny =np.min(pts_along_camera_rays[:,1])
    maxy =np.max(pts_along_camera_rays[:,1])
     
    minz =np.min(pts_along_camera_rays[:,2])
    maxz =np.max(pts_along_camera_rays[:,2])
    print("minx: {}, maxx {}".format(minx, maxx))
    print("miny: {}, maxy {}".format(miny, maxy))
    print("minz: {}, maxz {}".format(minz, maxz))
    write_to_binary(pts_along_camera_rays, args.output_file)

if __name__ == "__main__":
    main()