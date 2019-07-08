import os
import glob
import numpy as np
import tensorflow as tf
import save
import read

def main():
    d = '/media/root/data/scans_test/'
    scenes = [os.path.join(d, o) for o in os.listdir(d) 
                    if os.path.isdir(os.path.join(d,o))]

    nclasses = 42
    print(len(scenes))
    count = 0
    for scene in scenes:

        datacost_path = os.path.join(scene, "pred_datacost.npz")
        
        if not os.path.exists(datacost_path):
            print(scene)
            print("  Warning: datacost or groundtruth does not exist")
            continue

        datacost = np.load(datacost_path)["volume"]

        # Make sure the data is compatible with the parameters.
        assert datacost.shape[3] == nclasses

        datacost_outfile = os.path.join(scene, "pred_datacost.dat")
        if(os.path.exists(datacost_outfile)):
            os.remove(datacost_outfile)


        save.write_dat_datacost(datacost, datacost_outfile)
        count =  count+1


    print("There are totally {} valid scenes.".format(count))


if __name__ == '__main__':
    main()
