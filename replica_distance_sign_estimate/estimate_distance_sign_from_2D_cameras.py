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


remapper=np.ones(150)*(-100)
for i,x in enumerate(selected_class_ids):
    remapper[x]=i+1

## pillow class id 61,map pillow to cushsion
remapper[61] = 16

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames_2d_path", required=True)
    parser.add_argument("--scene_sampled_point_file", required=True)
    parser.add_argument("--scene_semantic_mesh", required=True)
    parser.add_argument("--scene_semantic_json_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--viewport_height", type=int, default=480)
    parser.add_argument("--viewport_width", type=int, default=640)
    parser.add_argument("--truncated_distance", type=float, default=0.1)
    return parser.parse_args()


def main():
    args = parse_args()


    #########load face semantic info
    mesh_semantic = plyfile.PlyData().read(args.scene_semantic_mesh)
    face_object_ids = np.asarray(mesh_semantic.elements[1]['object_id'])

    with open(args.scene_semantic_json_file) as info_sem_json_file:
        info_sem = json.load(info_sem_json_file)

    id_to_label = info_sem['id_to_label']
    id_to_label = np.asarray(id_to_label)
    face_labels = id_to_label[face_object_ids]


    ############ observation fusion ####################
    d = np.load(args.scene_sampled_point_file, allow_pickle=True).item()
    coords = d['point_coordinate']
    nearest_point = d['nearest_point_in_mesh']
    distance_to_mesh = d['distance_to_mesh']
    nearest_face_index = d['nearest_face_index']
    nearest_instance_index = d['nearest_instance_index']
    print("There are {} uniformly sampled points.".format(coords.shape[0]))

    ########
    for i in range(nearest_face_index.shape[0]):
        assert(face_object_ids[nearest_face_index[i]] == nearest_instance_index[i] )

    print(np.min(coords[:,0]))
    print(np.max(coords[:,0]))
    print(np.min(coords[:,1]))
    print(np.max(coords[:,1]))
    print(np.min(coords[:,2]))
    print(np.max(coords[:,2]))

    #######

    observed_volume = FreespaceVolume(args.viewport_height, args.viewport_width, coords)

    
    camera_paths = sorted(glob.glob(os.path.join(args.frames_2d_path, "frame*.camera.npz")))
    for i, camera_path in enumerate(camera_paths):

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
    for i in range(coords.shape[0]):

        ## freeespace
        if observed_volume_flag[i] == 1:

            valid_coords.append(coords[i,:])
            valid_distance_to_mesh.append(distance_to_mesh[i])
            valid_tsdf.append(min(distance_to_mesh[i], args.truncated_distance))
            valid_label.append(0) ## freespace

        elif distance_to_mesh[i] <= args.truncated_distance:
            
            valid_coords.append(coords[i,:])
            valid_distance_to_mesh.append(distance_to_mesh[i])
            valid_tsdf.append(-distance_to_mesh[i])
            label = id_to_label[nearest_instance_index[i]]
            if label < 0:
                label = 149
            valid_label.append(remapper[label])

    valid_coords=np.asarray(valid_coords)
    valid_distance_to_mesh = np.asarray(valid_distance_to_mesh)
    valid_tsdf = np.asarray(valid_tsdf, dtype=np.float32)
    valid_label = np.asarray(valid_label, dtype=np.int32)
    

    np.savez(args.output_file, coords=valid_coords,distance=valid_distance_to_mesh, tsdf=valid_tsdf, label=valid_label)

if __name__ == "__main__":
    main()
