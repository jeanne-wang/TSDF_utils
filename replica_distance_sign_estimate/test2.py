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
    parser.add_argument("--point_path", required=True)
    return parser.parse_args()


def main():
    args = parse_args()


    #########load face semantic info

    
    point_paths = sorted(glob.glob(os.path.join(args.point_path, "*.npy")))
    count = 0
    for i, point_file in enumerate(point_paths):
        d = np.load(point_file, allow_pickle=True).item()
        coords = d['point_coordinate']
        distance_to_mesh = d['distance_to_mesh']
        nearest_instance_index = d['nearest_instance_index']
        count = count + coords.shape[0]

    print(count)



if __name__ == "__main__":
    main()
