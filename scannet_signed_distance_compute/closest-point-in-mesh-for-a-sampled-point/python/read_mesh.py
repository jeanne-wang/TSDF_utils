from plyfile import PlyData, PlyElement
plydata = PlyData.read('my_output_file.ply')

# vertex
for item in plydata.elements[0].data[:100]:
	print(item)

# face
for item in plydata.elements[1].data[:100]:
	print(item[0])
