import os
import glob
import argparse
import numpy as np
import plyfile
import json
from observation_volume_from_2D_cameras import ObservationVolume

def create_color_palette():
    return [
       (0, 0, 0),
       (174, 199, 232),     # wall
       (152, 223, 138),     # floor
       (31, 119, 180),      # cabinet
       (255, 187, 120),     # bed
       (188, 189, 34),      # chair
       (140, 86, 75),       # sofa
       (255, 152, 150),     # table
       (214, 39, 40),       # door
       (197, 176, 213),     # window
       (148, 103, 189),     # bookshelf
       (196, 156, 148),     # picture
       (23, 190, 207),      # counter
       (178, 76, 76),  
       (247, 182, 210),     # desk
       (66, 188, 102), 
       (219, 219, 141),     # curtain
       (140, 57, 197), 
       (202, 185, 52), 
       (51, 176, 203), 
       (200, 54, 131), 
       (92, 193, 61),  
       (78, 71, 183),  
       (172, 114, 82), 
       (255, 127, 14),      # refrigerator
       (91, 163, 138), 
       (153, 98, 156), 
       (140, 153, 101),
       (158, 218, 229),     # shower curtain
       (100, 125, 154),
       (178, 127, 135),
       (120, 185, 128),
       (146, 111, 194),
       (44, 160, 44),       # toilet
       (112, 128, 144),     # sink
       (96, 207, 209), 
       (227, 119, 194),     # bathtub
       (213, 92, 176), 
       (94, 106, 211), 
       (82, 84, 163),       # otherfurn
       (100, 85, 144),
       (255, 0, 0)          # freespace
    ]

# Save 3D mesh to a polygon .ply file
def meshwrite(filename,verts,faces,colors):

    # Write header
    ply_file = open(filename,'w')
    ply_file.write("ply\n")
    ply_file.write("format ascii 1.0\n")
    ply_file.write("element vertex %d\n"%(verts.shape[0]))
    ply_file.write("property float x\n")
    ply_file.write("property float y\n")
    ply_file.write("property float z\n")
    ply_file.write("property uchar red\n")
    ply_file.write("property uchar green\n")
    ply_file.write("property uchar blue\n")
    ply_file.write("element face %d\n"%(faces.shape[0]))
    ply_file.write("property list uchar int vertex_index\n")
    ply_file.write("end_header\n")

    # Write vertex list
    for i in range(verts.shape[0]):
        ply_file.write("%f %f %f %d %d %d\n"%(verts[i,0],verts[i,1],verts[i,2], colors[i,0],colors[i,1],colors[i,2]))
    
    # Write face list
    for i in range(faces.shape[0]):
        ply_file.write("3 %d %d %d\n"%(faces[i,0],faces[i,1],faces[i,2]))

    ply_file.close()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_path", required=True)
    parser.add_argument("--scene_sampled_point_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--visualization", type=bool, default=False)
    parser.add_argument("--visual_output_file", type=string, default='mesh_vis.ply')
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
    nearest_point_in_mesh = d['nearest_point_in_mesh']
    print("There are {} sampled points along surface.".format(coords.shape[0]))

    ##################################


    observed_volume = ObservationVolume(coords)
    pose_paths = sorted(glob.glob(os.path.join(args.scene_path, "sensor/*.pose.txt")))
    num_frames = len(pose_paths)
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



    front_of_camera, behind_of_camera = observed_volume.get_volume()

    valid_coords = []
    valid_nearest_point_in_mesh = []
    valid_sdf = []
    valid_label = []
    for i in range(coords.shape[0]):

        ## freeespace
        if front_of_camera[i] == 1:
            valid_coords.append(coords[i,:])
            valid_nearest_point_in_mesh.append(nearest_point_in_mesh[i])
            valid_sdf.append(distance_to_mesh[i])
            valid_label.append(41) ## freespace 
            

        elif ((behind_of_camera[i] == num_frames) and (distance_to_mesh[i] <= args.truncated_distance)):
            
            valid_coords.append(coords[i,:])
            valid_nearest_point_in_mesh.append(nearest_point_in_mesh[i])
            valid_sdf.append(-distance_to_mesh[i])
            valid_label.append(nearest_face_label[i])
            


    valid_coords=np.asarray(valid_coords)
    valid_nearest_point_in_mesh = np.asarray(valid_nearest_point_in_mesh)
    valid_sdf = np.asarray(valid_sdf, dtype=np.float32)
    valid_label = np.asarray(valid_label, dtype=np.int32)
    
    num_points_along_camera_rays = valid_coords.shape[0]
    print("There are {} sampled near surface points along camera rays.".format(num_points_along_camera_rays))

    np.savez(args.output_file, coords=valid_coords, sdf=valid_sdf, 
        label=valid_label, nearest_point=valid_nearest_point_in_mesh)

    if args.visualization:
        ## create new vertex colors
        vertex_sem_colors = np.empty(shape=[num_points_along_camera_rays,3], dtype=np.ubyte)
        color_palette = create_color_palette()
        for idx, color in enumerate(color_palette):
            vertex_sem_colors[valid_label==idx] = color

        vis_coords = []
        vis_color = []
        for i in range(num_points_along_camera_rays):
            vis_coords.append(valid_coords[i,:])
            vis_coords.append(valid_nearest_point_in_mesh[i,:])
            vis_coords.append(valid_nearest_point_in_mesh[i,:])
            vis_color.append(vertex_sem_colors[i])
            vis_color.append(vertex_sem_colors[i])
            vis_color.append(vertex_sem_colors[i])

        vis_coords = np.asarray(vis_coords)
        vis_color = np.asarray(vis_color)
        fff = np.zeros((vis,3), dtype=np.int32)
        for i in range(vis):
            fff[i, 0] = i *3
            fff[i, 1] = i*3+1
            fff[i, 2] = i*3+2
     
        meshwrite(args.visual_output_file, vis_coords, fff, vis_color)

if __name__ == "__main__":
    main()