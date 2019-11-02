import os
import argparse
import numpy as np, json
import trimesh
import math
import sys
import struct

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
    parser.add_argument("--habitat_mesh_file", type=str, required=True)
    parser.add_argument("--num_sample", type=int, default=200000)
    parser.add_argument("--variance", type=float, default=0.05)
    parser.add_argument("--output_path", type=str, default='./')
    parser.add_argument("--density", type=float, default=1000)
    return parser.parse_args()

def main():
	
    args = parse_args()
    variance = args.variance
    second_variance = variance/10

    mesh = trimesh.load(args.habitat_mesh_file)

    if args.density != 0:
        area = mesh.area_faces
        # total area (float)
        area_sum = np.sum(area)
        num_sample = int(area_sum*args.density)
    else:
        num_sample = int(args.num_sample/2)

    print("Sample {} points on the space evenly".format(num_sample))




    coords_surf, face_indices = trimesh.sample.sample_surface_even(mesh, num_sample)
    print("Sampled {} points after reject close points".format(coords_surf.shape[0]))
    
    coords = []
    for i in range(coords_surf.shape[0]):
        samp1_x = coords_surf[i,0] + np.random.normal(0, variance)
        samp1_y = coords_surf[i,1] + np.random.normal(0, variance)
        samp1_z = coords_surf[i,2] + np.random.normal(0, variance)
        samp1 = np.array([samp1_x, samp1_y,  samp1_z])

        samp2_x = coords_surf[i,0] + np.random.normal(0, second_variance)
        samp2_y = coords_surf[i,1] + np.random.normal(0, second_variance)
        samp2_z = coords_surf[i,2] + np.random.normal(0, second_variance)
        samp2 = np.array([samp2_x, samp2_y,  samp2_z])
        
        coords.append(samp1)
        coords.append(samp2)

    coords = np.asarray(coords, dtype=np.float32, order='C')

    write_to_binary(coords, args.output_path+'sample_along_surf_points.dat')


if __name__ == '__main__':
	main()

