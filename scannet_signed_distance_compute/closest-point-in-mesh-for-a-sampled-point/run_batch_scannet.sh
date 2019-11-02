g++ -std=c++11 sample_points_scannet.cpp -o sample_points_scannet -lpthread
MESH_ROOT="/media/root/data/ScanNet_v2_data/*/scene*2.ply"
for MESH_FILE in $(ls ${MESH_ROOT})
do
	SCENE_NAME=$(basename ${MESH_FILE})
	SCENE_NAME=${SCENE_NAME//.ply/}
	mkdir -p "./scannet/"${SCENE_NAME}
	echo "==========================="
	echo ${SCENE_NAME}
	./sample_points_scannet -f ${MESH_FILE} -d 1000 -o "./scannet/"${SCENE_NAME} -p 0.05 -v 10
	echo "==========================="
done
