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
    count = 0
    for scene in scenes:

        datacost_path = os.path.join(scene, "datacost.npz")
        groundtruth_path = os.path.join(scene, "groundtruth_model/probs.npz")
        
        if not os.path.exists(datacost_path) or not os.path.exists(groundtruth_path):
            print(scene)
            print("  Warning: datacost or groundtruth does not exist")
            continue

        datacost = np.load(datacost_path)["volume"]
        groundtruth = np.load(groundtruth_path)["probs"]

        # Make sure the data is compatible with the parameters.
        assert datacost.shape[3] == nclasses
        assert datacost.shape == groundtruth.shape

        ## softmax
        softmax_scale = 10
        x = tf.placeholder(tf.float32, shape=groundtruth.shape)
        y = tf.nn.softmax(softmax_scale*x)
        init = tf.initialize_all_variables()
        
        with tf.Session() as sess:
            sotfmax_gt = sess.run(y, feed_dict={x: groundtruth})


        datacost_outfile = os.path.join(scene, "datacost.dat")
        gt_outfile = os.path.join(scene, "groundtruth.dat")
        if(os.path.exists(datacost_outfile)):
            os.remove(datacost_outfile)

        if(os.path.exists(gt_outfile)):
            os.remove(gt_outfile)

        save.write_dat_groundtruth(sotfmax_gt, gt_outfile)
        save.write_dat_datacost(datacost, datacost_outfile)
        count =  count+1


    print("There are totally {} valid scenes.".format(count))


if __name__ == '__main__':
    main()
