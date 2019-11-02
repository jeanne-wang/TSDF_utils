import os
import glob
import argparse
import numpy as np
from skimage import measure


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--resolution", type=float, default=0.05)
    return parser.parse_args()


def get_mesh(color_sdf_volume, bbox, resolution):
    tsdf_vol = color_sdf_volume[:,:,:,-1]
    color_vol = color_sdf_volume[:,:,:, :-1]


    
    # Marching cubes
    verts,faces,norms,vals = measure.marching_cubes_lewiner(tsdf_vol,level=0)
    verts_ind = np.round(verts).astype(int)
    verts = verts* resolution + bbox[:,0] # voxel grid coordinates to world coordinates

    # Get vertex colors
    colors = color_vol[verts_ind[:,0],verts_ind[:,1],verts_ind[:,2]]
    colors = colors.astype(np.uint8)
    return verts,faces,norms,colors

# Save 3D mesh to a polygon .ply file
def meshwrite(filename,verts,faces,norms,colors):

    # Write header
    ply_file = open(filename,'w')
    ply_file.write("ply\n")
    ply_file.write("format ascii 1.0\n")
    ply_file.write("element vertex %d\n"%(verts.shape[0]))
    ply_file.write("property float x\n")
    ply_file.write("property float y\n")
    ply_file.write("property float z\n")
    ply_file.write("property float nx\n")
    ply_file.write("property float ny\n")
    ply_file.write("property float nz\n")
    ply_file.write("property uchar red\n")
    ply_file.write("property uchar green\n")
    ply_file.write("property uchar blue\n")
    ply_file.write("element face %d\n"%(faces.shape[0]))
    ply_file.write("property list uchar int vertex_index\n")
    ply_file.write("end_header\n")

    # Write vertex list
    for i in range(verts.shape[0]):
        ply_file.write("%f %f %f %f %f %f %d %d %d\n"%(verts[i,0],verts[i,1],verts[i,2],norms[i,0],norms[i,1],norms[i,2],colors[i,0],colors[i,1],colors[i,2]))
    
    # Write face list
    for i in range(faces.shape[0]):
        ply_file.write("3 %d %d %d\n"%(faces[i,0],faces[i,1],faces[i,2]))

    ply_file.close()

def main():
    args = parse_args()

    bbox = np.loadtxt(os.path.join(args.input_path, "bbox.txt"))
    bbox = bbox.astype(np.float32)

    color_sdf_volume = np.load(os.path.join(args.input_path, "color_sdf.npz"))['volume']
    
    verts,faces,norms,colors = get_mesh(color_sdf_volume, bbox, args.resolution)
    meshwrite(os.path.join(args.input_path, "mesh_from_2D.ply"),verts,faces,norms,colors)



if __name__ == "__main__":
    main()
