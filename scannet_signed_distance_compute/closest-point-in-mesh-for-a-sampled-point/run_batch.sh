g++ -std=c++11 sample_points.cpp -o sample_points -lpthread
MESH_ROOT="../replica_v1/"
for SCENE_NAME in $(ls ${MESH_ROOT})
do
	MESH_FILE=${MESH_ROOT}${SCENE_NAME}"/mesh.ply"
	mkdir -p "scene_"${SCENE_NAME}
	echo "==========================="
	echo ${SCENE_NAME}
	./sample_points -f ${MESH_FILE} -d 3000 -o "scene_"${SCENE_NAME} -p 0.05 -v 10
	echo "==========================="
done
