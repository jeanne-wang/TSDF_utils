SCANNET_PATH=/cluster/scratch/xiaojwan/ScanNet

declare -i count=0
for scene_path in $(ls -d $SCANNET_PATH/scans/scene*)
do
	scene_name=$(basename $scene_path)

	if [ $count -lt 500 ]; then
        count=count+1
		continue
    fi
    
    
    if [ $count -eq 800 ]; then
        echo $scene_name
        break
    fi
    

    ## delete existed process result to avoid overwritten
    if [ -d ${scene_path}/label ]; then
        rm -r ${scene_path}/label
    fi

    if [ -d ${scene_path}/sensor ]; then
    	rm -r ${scene_path}/sensor    
    fi

    if [ -d ${scene_path}/converted ]; then
        rm -r ${scene_path}/converted
    fi



    unzip $scene_path/${scene_name}_2d-label.zip -d ${scene_path}

    $SCANNET_PATH/SensReader/c++/sens \
     	$scene_path/${scene_name}.sens $scene_path/sensor

   
    

    mkdir -p $scene_path/converted/
    mkdir -p $scene_path/converted/images
    mkdir -p $scene_path/converted/groundtruth_model

    # Convert scannet data into a format read by our code
    python3 $SCANNET_PATH/TSDF/convert_scannet.py \
        --scene_path $scene_path \
        --output_path $scene_path/converted \
        --label_map_path $SCANNET_PATH/scannetv2-labels.combined.tsv \
        --resolution 0.05

    rm -r ${scene_path}/label
    rm -r ${scene_path}/sensor

    # Fuse all depth maps and segmentation to create the ground-truth
    python3 $SCANNET_PATH/TSDF/tsdf_fusion.py \
        --input_path $scene_path/converted/ \
        --output_path $scene_path/converted/groundtruth_datacost \
        --resolution 0.05

    # # Run total variation on the datacost obtained from all depth
    # # to generate the ground truth voxel grid
    # python $SCANNET_PATH/eval_tv_l1.py \
    #     --datacost_path $scene_path/converted/groundtruth_datacost.npz \
    #     --output_path $scene_path/converted/groundtruth_model \
    #     --label_map_path $scene_path/converted/labels.txt \
    #     --niter_steps 10 --lam 10 --nclasses 42

    # Fuse every 50 frame to generate incomplete input data
    python3 $SCANNET_PATH/TSDF/tsdf_fusion.py \
        --input_path $scene_path/converted/ \
        --output_path $scene_path/converted/datacost \
        --frame_rate 50 \
        --resolution 0.05

    rm -r $scene_path/converted/images

    count=count+1

done



