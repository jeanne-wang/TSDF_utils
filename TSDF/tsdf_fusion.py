import os
import glob
import argparse
import numpy as np

from tsdf_volume import TSDFVolume


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--frame_rate", type=int, default=1)
    parser.add_argument("--resolution", type=float, default=0.05)
    parser.add_argument("--resolution_factor", type=int, default=2)
    return parser.parse_args()


def write_ply(path, points, color):
    with open(path, "w") as fid:
        fid.write("ply\n")
        fid.write("format ascii 1.0\n")
        fid.write("element vertex {}\n".format(points.shape[0]))
        fid.write("property float x\n")
        fid.write("property float y\n")
        fid.write("property float z\n")
        fid.write("property uchar diffuse_red\n")
        fid.write("property uchar diffuse_green\n")
        fid.write("property uchar diffuse_blue\n")
        fid.write("end_header\n")
        for i in range(points.shape[0]):
            fid.write("{} {} {} {} {} {}\n".format(points[i, 0], points[i, 1],
                                                   points[i, 2], *color))


def main():
    args = parse_args()

    bbox = np.loadtxt(os.path.join(args.input_path, "bbox.txt"))

    num_labels = 0
    with open(os.path.join(args.input_path, "labels.txt"), "r") as fid:
        for line in fid:
            if line.strip() and not "freespace" in line:
                num_labels += 1

    tsdf_volume = TSDFVolume(num_labels, bbox, args.resolution,
                             args.resolution_factor)

    image_paths = sorted(glob.glob(os.path.join(args.input_path,
                                                "images/*.npz")))

    for i, image_path in enumerate(image_paths):
        if i % args.frame_rate != 0:
            continue

        print("Processing {} [{}/{}]".format(
              os.path.basename(image_path), i + 1, len(image_paths)))
        image = np.load(image_path)

        if image["label_map"].dtype == np.int32:
            label_map = np.zeros((image["label_map"].shape[0],
                                  image["label_map"].shape[1],
                                  num_labels), dtype=np.float32)
            rr, cc = np.mgrid[:label_map.shape[0], :label_map.shape[1]]
            label_map[(rr.ravel(), cc.ravel(), image["label_map"].ravel())] = 1
        else:
            label_map = image["label_map"]

        tsdf_volume.fuse(image["depth_proj_matrix"], image["label_proj_matrix"],
                         image["depth_map"], label_map)

    # occupied_volume_idxs = np.column_stack(
    #     np.where(np.sum(tsdf_volume.get_volume()[..., :-1], axis=-1) < 0))
    # write_ply(args.output_path + ".ply",
    #           occupied_volume_idxs * args.resolution, color=[255, 0, 0])

    np.savez(args.output_path + ".npz",
             volume=tsdf_volume.get_volume(),
             resolution=args.resolution)


if __name__ == "__main__":
    main()
