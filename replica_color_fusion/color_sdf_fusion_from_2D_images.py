import os
import glob
import argparse
import numpy as np
from PIL import Image
from color_sdf_fusion_volume import ColorSDFVolume


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--viewport_height", type=int, default=480)
    parser.add_argument("--viewport_width", type=int, default=640)
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

    color_sdf_volume = ColorSDFVolume(bbox, args.viewport_height, args.viewport_width, args.resolution, args.resolution_factor)

    camera_paths = sorted(glob.glob(os.path.join(args.input_path,
                                                "frames/frame*.npz")))

    for i, camera_path in enumerate(camera_paths):
        if i % args.frame_rate != 0:
            continue

        frame_id = int(os.path.basename(camera_path)[6:11])
        print("Processing frame {} [{}/{}]".format(
             frame_id, i + 1, len(camera_paths)))
        depth_map_path = os.path.join(args.input_path,
                                   "frames/frame-{:05d}.depth.png".format(frame_id))

        color_map_path = os.path.join(args.input_path,
                                   "frames/frame-{:05d}.rgba.png".format(frame_id))
        
        camera = np.load(camera_path)
        proj_matrix = camera['projection_matrix']
        cam_matrix = camera['camera_matrix']
        transform_matrix =  np.matmul(proj_matrix, cam_matrix)

        depth_map = Image.open(depth_map_path)
        depth_map = np.ascontiguousarray(depth_map, dtype=np.float32)
        depth_map = depth_map*10/255 ## here depth map stores unprojected depth value

        color_image = Image.open(color_map_path)
        color_image = np.ascontiguousarray(color_image, dtype=np.float32)
        color_image = color_image[:,:,:-1].copy(order='C')

        
        color_sdf_volume.fuse(transform_matrix,
                            depth_map, color_image)


    np.savez(args.output_path + "color_sdf.npz",
             volume=color_sdf_volume.get_volume(),
             resolution=args.resolution)


if __name__ == "__main__":
    main()
