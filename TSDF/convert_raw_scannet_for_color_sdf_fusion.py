import os
import glob
import shutil
import argparse
import h5py
import numpy as np
import pandas as pd
import skimage.io
from pyntcloud import PyntCloud
from pyntcloud.samplers.base import Sampler
from pyntcloud.geometry.areas import triangle_area_multi


def mkdir_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--overwrite", type=int, default=False)
    parser.add_argument("--resolution", type=float, default=0.05)
    return parser.parse_args()


def main():
    args = parse_args()

    mkdir_if_not_exists(args.output_path)
    mkdir_if_not_exists(os.path.join(args.output_path, "images"))


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
                color_K = np.array(list(map(float, line.split("=")[1].split())))
                color_K = color_K.reshape(4, 4)[:3, :3]
            elif line.startswith("m_calibrationDepthIntrinsic"):
                depth_K = np.array(list(map(float, line.split("=")[1].split())))
                depth_K = depth_K.reshape(4, 4)[:3, :3]
            elif line.startswith("m_calibrationColorExtrinsic") or \
                    line.startswith("m_calibrationDepthExtrinsic"):
                assert line.split("=")[1].strip() \
                       == "1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1"

    # Load the groundtruth mesh and convert it to a point cloud.
    groundtruth = PyntCloud.from_file(
        glob.glob(os.path.join(args.scene_path, "*_vh_clean_2.ply"))[0])


    # Compute the bounding box of the ground truth.
    minx = groundtruth.points.x.min()
    miny = groundtruth.points.y.min()
    minz = groundtruth.points.z.min()
    maxx = groundtruth.points.x.max()
    maxy = groundtruth.points.y.max()
    maxz = groundtruth.points.z.max()

    # Extend the bounding box by a stretching factor.
    diffx = maxx - minx
    diffy = maxy - miny
    diffz = maxz - minz
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

    # Process the frames in the scene.
    pose_paths = sorted(glob.glob(
        os.path.join(args.scene_path, "sensor/*.pose.txt")))
    for i, pose_path in enumerate(pose_paths):
        image_id = int(os.path.basename(pose_path)[6:-9])

        print("Processing frame {} [{}/{}]".format(
            image_id, i + 1, len(pose_paths)))

        output_path = os.path.join(args.output_path,
                                   "images/{:06d}.npz".format(image_id))

        if args.overwrite or not os.path.exists(output_path):
            depth_map_path = os.path.join(
                args.scene_path,
                "sensor/frame-{:06d}.depth.pgm".format(image_id))
            color_map_path = os.path.join(
                args.scene_path,
                "sensor/frame-{:06d}.color.jpg".format(image_id))

            assert os.path.exists(depth_map_path)
            assert os.path.exists(color_map_path)

            pose = np.loadtxt(pose_path)

            proj_matrix = np.linalg.inv(pose)
            depth_proj_matrix = \
                np.dot(depth_K, np.dot(color_to_depth_T, proj_matrix)[:3])
            color_proj_matrix = np.dot(color_K, proj_matrix[:3])

            depth_proj_matrix = depth_proj_matrix.astype(np.float32)
            color_proj_matrix = color_proj_matrix.astype(np.float32)

            depth_map = skimage.io.imread(depth_map_path)
            depth_map = depth_map.astype(np.float32) / 1000

            color_map = skimage.io.imread(color_map_path)
            color_map = color_map.astype(np.float32)

            # Write the output into one combined NumPy file.
            np.savez_compressed(
                output_path,
                depth_proj_matrix=depth_proj_matrix,
                color_proj_matrix=color_proj_matrix,
                depth_map=depth_map,
                color_map=color_map)


if __name__ == "__main__":
    main()
