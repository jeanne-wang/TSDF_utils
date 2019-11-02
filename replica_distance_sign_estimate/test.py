import os
import glob
import argparse
import numpy as np
import plyfile
import json
from freespace_volume_from_2D_cameras import FreespaceVolume


selected_classes = ['wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'shelf',
    'picture', 'rug', 'blinds', 'lamp', 'refrigerator', 'cushion', 'ceiling', 'wall-cabinet']
selected_class_ids = [93, 40, 18, 7, 20, 76, 80, 37, 97, 71, 59, 98, 12, 47, 67, 29, 31, 94]


remapper=np.zeros(150)*(-100)
for i,x in enumerate(selected_class_ids):
    remapper[x]=i+1

## pillow class id 61,map pillow to cushsion
remapper[61] = 16

# color palette for nyu40 labels
def create_color_palette():
    return [
       (255, 0, 0),           # freespace
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
       (23, 190, 207),      # rug
       (178, 76, 76),       # blinds
       (247, 182, 210),     # lamp
       (255, 127, 14),      # refrigerator
       (158, 218, 229),     # cusion
       (44, 160, 44),       # ceiling
       (112, 128, 144)     # wall-cabinet
       
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
    parser.add_argument("--frames_2d_path", required=True)
    parser.add_argument("--scene_sampled_point_file", required=True)
    parser.add_argument("--scene_semantic_json_file", required=True)
    #parser.add_argument("--output_file", required=True)
    parser.add_argument("--viewport_height", type=int, default=480)
    parser.add_argument("--viewport_width", type=int, default=640)
    parser.add_argument("--truncated_distance", type=float, default=0.1)
    return parser.parse_args()


def main():
    args = parse_args()


    #########load face semantic info

    with open(args.scene_semantic_json_file) as info_sem_json_file:
        info_sem = json.load(info_sem_json_file)

    id_to_label = info_sem['id_to_label']
    id_to_label = np.asarray(id_to_label)

    ############ observation fusion ####################

    d = np.load(args.scene_sampled_point_file, allow_pickle=True).item()
    coords = d['point_coordinate']
    distance_to_mesh = d['distance_to_mesh']
    nearest_instance_index = d['nearest_instance_index']
    nearest_point_in_mesh = d['nearest_point_in_mesh']

    print("There are {} uniformly sampled points.".format(coords.shape[0]))

    ##################################


    observed_volume = FreespaceVolume(args.viewport_height, args.viewport_width, coords)

    
    camera_paths = sorted(glob.glob(os.path.join(args.frames_2d_path, "frame*.camera.npz")))
    for i, camera_path in enumerate(camera_paths):
        if i >= 1:
            break

        frame_id = int(os.path.basename(camera_path)[6:11])
        print("Processing frame {} [{}/{}]".format(
             frame_id, i + 1, len(camera_paths)))

        
        

        camera = np.load(camera_path)
        proj_matrix = camera['projection_matrix']
        cam_matrix = camera['camera_matrix']
        transform_matrix =  np.matmul(proj_matrix, cam_matrix)

        depth_map_path = os.path.join(args.frames_2d_path, "frame-{:05d}.depth.npz".format(frame_id))
        depth_map = np.load(depth_map_path)['depth']

        
        observed_volume.fuse(transform_matrix, depth_map)



    observed_volume_flag = observed_volume.get_volume()

    valid_coords = []
    valid_distance_to_mesh = []
    valid_tsdf = []
    valid_label = []
    valid_nearest_point_in_mesh = []
    count_free = 0
    count_occu = 0
    count_occu_know = 0
    for i in range(coords.shape[0]):

        ## freeespace
        if observed_volume_flag[i] == 1:

            valid_coords.append(coords[i,:])
            valid_distance_to_mesh.append(distance_to_mesh[i])
            valid_nearest_point_in_mesh.append(nearest_point_in_mesh[i])
            valid_tsdf.append(min(distance_to_mesh[i], args.truncated_distance))
            valid_label.append(0) ## freespace also labeled as -100, will be ignored during semantic loss computation
            count_free = count_free+1
            

        elif distance_to_mesh[i] <= args.truncated_distance:
            count_occu = count_occu+1
            label = id_to_label[nearest_instance_index[i]]
            if label < 0:
                label = 149
            if (remapper[label] > 0):
                count_occu_know = count_occu_know+1
                valid_coords.append(coords[i,:])
                valid_distance_to_mesh.append(distance_to_mesh[i])
                valid_nearest_point_in_mesh.append(nearest_point_in_mesh[i])
                valid_tsdf.append(-distance_to_mesh[i])
                valid_label.append(remapper[label])



    valid_coords=np.asarray(valid_coords)
    valid_distance_to_mesh = np.asarray(valid_distance_to_mesh)
    valid_tsdf = np.asarray(valid_tsdf, dtype=np.float32)
    valid_label = np.asarray(valid_label, dtype=np.int32)
    valid_nearest_point_in_mesh = np.asarray(valid_nearest_point_in_mesh)

    print("There are {} points along camera ray".format(count_free+count_occu))
    print("There are {} freespace points along camera rays".format(count_free))
    print("There are {} occupied points along rays, of which {} points are not ignored".format(count_occu, count_occu_know))
    ## create new vertex colors
    vertex_sem_colors = np.empty(shape=[valid_coords.shape[0],3], dtype=np.ubyte)
    color_palette = create_color_palette()
    for idx, color in enumerate(color_palette):
        vertex_sem_colors[valid_label==idx] = color


    vis = 50000
    vis_coords = []
    vis_color = []
    for i in range(vis):
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
     
    meshwrite('mesh_vis_1.ply', vis_coords, fff, vis_color)

if __name__ == "__main__":
    main()
