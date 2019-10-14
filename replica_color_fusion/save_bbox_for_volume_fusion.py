import os
import argparse
import numpy as np
from pyntcloud import PyntCloud


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

if __name__ == "__main__":
    main()
