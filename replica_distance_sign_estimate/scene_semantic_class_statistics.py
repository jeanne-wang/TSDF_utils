import os
import glob
import argparse
import numpy as np
import plyfile
import json
from freespace_volume_from_2D_cameras import FreespaceVolume

selected_classes = ['wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'shelf', 
    'rug', 'blinds', 'lamp', 'refrigerator', 'cushion', 'ceiling', 'wall-cabinet']
selected_class_ids = [93, 40, 18, 7, 20, 76, 80, 37, 97, 71, 98, 12, 47, 67, 29, 31, 94]


remapper=np.ones(150)*(-100)
for i,x in enumerate(selected_class_ids):
    remapper[x]=i

## pillow class id 61,map pillow to cushsion
remapper[61] = 14

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tsdf_path", required=True)
    return parser.parse_args()


def main():
    args = parse_args()


    #########load face semantic info

    
    tsdf_paths = sorted(glob.glob(os.path.join(args.tsdf_path, "*.npz")))
    point_coords = []
    point_scene_idxs=[]
    point_tsdfs=[]
    point_labels=[]
    point_distances = []
    for i, tsdf_file in enumerate(tsdf_paths):

        scene_name=os.path.basename(tsdf_file)[:-4]
        print("Processing scene "+scene_name)
        info = np.load(tsdf_file)
        coords = info['coords']
        distance = info['distance']
        tsdf = info['tsdf']
        label = info['label']

        point_coords.append(coords)
        point_tsdfs.extend(list(tsdf))
        point_labels.extend(list(label))
        point_scene_idxs.extend(coords.shape[0]*[i])
        point_distances.extend(list(distance))


    point_coords = np.vstack(point_coords)
    point_tsdfs = np.asarray(point_tsdfs)
    point_labels = np.asarray(point_labels)
    point_distances = np.asarray(point_distances)
    point_scene_idxs = np.asarray(point_scene_idxs)
    assert(point_coords.shape[0] == point_tsdfs.shape[0])
    assert(point_tsdfs.shape[0] == point_labels.shape[0])
    assert(point_labels.shape[0] == point_scene_idxs.shape[0])
    assert(point_scene_idxs.shape[0] == point_distances.shape[0])

    print("There are {} sampled point together".format(point_coords.shape[0]))
    label_set, label_count = np.unique(point_labels, return_counts=True)
    print(np.sum(point_tsdfs >= 0))
    print(np.max(point_distances))

    # label_set = np.ascontiguousarray(label_set[1:]) ## ignore -100 class
    # label_count = np.ascontiguousarray(label_count[1:])

    statistic_info = list(zip(label_set, label_count))
    for count in statistic_info:
        print (count)
    
    mins = np.min(label_count)
    weight = mins/label_count
    print(weight)



if __name__ == "__main__":
    main()
