import os
import glob
import numpy as np
import tensorflow as tf
import save
import read

def main():
    d = '/media/root/data/ScanNet_DAT/'
    scenes = [os.path.join(d, o) for o in os.listdir(d) 
                    if os.path.isdir(os.path.join(d,o))]

    nclasses = 42
    print(len(scenes))

    for scene in scenes:

        datacost_path = os.path.join(scene, "pred_datacost.dat")
        groundtruth_path = os.path.join(scene, "groundtruth.dat")
        
        if not os.path.exists(datacost_path) or not os.path.exists(groundtruth_path):
            print(scene)
            print("  Warning: datacost or groundtruth does not exist")
            continue

        datacost = read.read_gvr_datacost(datacost_path, nclasses)
        groundtruth = read.read_gvr_groundtruth(groundtruth_path)

        # Make sure the data is compatible with the parameters.
        assert datacost.shape[3] == nclasses
        assert datacost.shape == groundtruth.shape


if __name__ == '__main__':
    main()
