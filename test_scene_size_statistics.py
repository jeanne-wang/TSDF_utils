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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_scene_path", required=True)
    parser.add_argument("--resolution", type=float, default=0.05)
    return parser.parse_args()


def main():
    args = parse_args()

    scenes = sorted(glob.glob(os.path.join(args.test_scene_path, "*_vh_clean_2.ply")))
    print(len(scenes))
    heights = []
    widths = []
    depths = []
    for scene in scenes:
        # Load the  mesh and convert it to a point cloud.
        groundtruth = PyntCloud.from_file(scene)
        # Compute the bounding box of the ground truth.
        minx = groundtruth.points.x.min()
        miny = groundtruth.points.y.min()
        minz = groundtruth.points.z.min()
        maxx = groundtruth.points.x.max()
        maxy = groundtruth.points.y.max()
        maxz = groundtruth.points.z.max()

        # # Extend the bounding box by a stretching factor.
        # diffx = maxx - minx
        # diffy = maxy - miny
        # diffz = maxz - minz
        # minx -= 0.05 * diffx
        # maxx += 0.05 * diffx
        # miny -= 0.05 * diffy
        # maxy += 0.05 * diffy
        # minz -= 0.05 * diffz
        # maxz += 0.05 * diffz

        # Write the bounding box.
        bbox = np.array(((minx, maxx), (miny, maxy), (minz, maxz)),
            dtype=np.float32)
        volume_size = np.diff(bbox, axis=1)
        volume_shape = volume_size.ravel() / args.resolution
        volume_shape = np.ceil(volume_shape).astype(np.int32).tolist()
        heights.append(volume_shape[0])
        widths.append(volume_shape[1])
        depths.append(volume_shape[2])
        print(volume_shape)

    heights = np.asarray(heights)
    widths = np.asarray(widths)
    depths = np.asarray(depths)
    print("minimum height {}, maximum height {}\n".format(np.amin(heights), np.amax(heights)))
    print("minimum width {}, maximum width {}\n".format(np.amin(widths), np.amax(widths)))
    print("minimum depth {}, maximum depth {}\n".format(np.amin(depths), np.amax(depths)))


if __name__ == "__main__":
    main()
