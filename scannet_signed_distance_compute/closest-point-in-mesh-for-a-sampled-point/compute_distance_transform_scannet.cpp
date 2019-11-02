#include "happly.h"
#include "argparse.h"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <igl/AABB.h>
#include <iostream>
#include <algorithm>
#include <string>
#include <vector>
#include <ctime>
#include <iomanip>
#include <random>
#include <fstream>
#include <cmath>

bool is_byteorder_big_endian(){

	int num = 1;
	if(*(char *)&num == 1){
        return false;
    }else{
        return true;
    }
 
}

template <typename Derived>
void writeMatrixInt(std::ofstream &io, Eigen::MatrixBase<Derived> &mat, bool half_idx) {
	int num_elems = mat.rows() * mat.cols();
	int *data = new int[num_elems];
	int idx = 0;
	for (int i = 0; i < mat.rows(); ++i) {
		for (int j = 0; j < mat.cols(); ++j) {
			data[idx] = mat(i, j);
			if (half_idx) {
				data[idx] /= 2;
			}
			++idx;
		}
	}
	io.write((char *)(data), num_elems * sizeof(int));
	delete [] data;
	return;
}

template <typename Derived>
void writeMatrixFloat(std::ofstream &io, Eigen::MatrixBase<Derived> &mat, bool sqrt_val) {
	int num_elems = mat.rows() * mat.cols();
	float *data = new float[num_elems];
	int idx = 0;
	for (int i = 0; i < mat.rows(); ++i) {
		for (int j = 0; j < mat.cols(); ++j) {
			data[idx] = mat(i, j);
			if (sqrt_val) {
				data[idx] = std::sqrt(data[idx]);
			}
			++idx;
		}
	}
	io.write((char *)(data), num_elems * sizeof(float));
	delete [] data;
	return;
}

template <typename Derived>
void writeMatToFile(
	std::string filename,
	Eigen::MatrixBase<Derived> &mat,
	bool is_int,
	bool half_idx = false,
	bool sqrt_val = false
) {
	std::ofstream bio(filename.c_str(), std::ios::binary);
	if (is_int) {
		writeMatrixInt(bio, mat, half_idx);
	} else {
		writeMatrixFloat(bio, mat, sqrt_val);
	}
	bio.close();
	return;
}

// read sampled points from file to Eigen Matrix
void readFileToMat(
	std::string filename,
	Eigen::MatrixXd &P
) {
	// filename: contained sampled point in float
	std::ifstream bio(filename.c_str(), std::ios::binary);
	
	uint8_t version;
	uint8_t is_big_endian;
	uint8_t uint_size;
	uint32_t elem_size;
	uint32_t num_point;
	uint32_t dimension;
    

        
    bio.read((char *)(&version), sizeof(version));
    assert(version == 1);

    bio.read((char *)(&is_big_endian), sizeof(is_big_endian));
    assert((is_big_endian == 1 && is_byteorder_big_endian()) || (is_big_endian == 0 && !is_byteorder_big_endian()));
  
	bio.read((char *)(&uint_size), sizeof(uint_size));
    assert(uint_size==4);
           
    bio.read((char *)(&elem_size), sizeof(elem_size));
    assert(elem_size == 4); // the data must be in float 

    bio.read((char *)(&num_point), sizeof(num_point));
    bio.read((char *)(&dimension), sizeof(dimension));
    assert(dimension == 3);

      
    int num_elems = num_point*dimension;
    float* data = new float[num_elems];
    bio.read((char *)(data), num_elems* sizeof(float));

    P.resize(num_point, 3);
    for (int i = 0; i < num_point; i++){
    	for (int j = 0; j < 3; j++){
    		P(i, j) = data[i*3+j];
    	}
    }

    delete [] data;


	bio.close();
	return;
}

int main(int argc, char** argv) {

	// Parse arguments
	std::string line_break = "\n                           ";
	ArgumentParser parser("Argument parser example");
	parser.add_argument("-p", "--pointfile", "string, filename of the sampled point for which to compute distance transform" + line_break, true);
	parser.add_argument("-f", "--file", "string, filename (.ply) of the input mesh" + line_break, true);
	parser.add_argument("-o", "--output", "string, prefix of the output (.dat) filename" + line_break, true);
	parser.add_argument("-s", "--seed", "int, seed of random number generator" + line_break +
										"default: 7", false);
	parser.add_argument("-v", "--visualize", "int, number of points to visualize" + line_break +
											 "default: 0", false);
	try {
		parser.parse(argc, argv);
	} catch (const ArgumentParser::ArgumentNotFound& ex) {
		std::cout << ex.what() << std::endl;
		return 0;
	}
	if (parser.is_help()) {
		return 0;
	}

	std::string point_filename = parser.get<std::string>("p");
	std::string filename = parser.get<std::string>("f");
	std::string outfile = parser.get<std::string>("o");
	
	int seed;
	if (parser.exists("s")) {
		seed = parser.get<int>("s");
	} else {
		seed = 7;
	}
	int vis;
	if (parser.exists("v")) {
		vis = parser.get<int>("v");
	} else {
		vis = 0;
	}



	// Construct the data object by reading from file
	happly::PLYData mesh(filename, false);

	// Get mesh-style data from the object
	std::vector<float> x = mesh.getElement("vertex").getProperty<float>("x");
	std::vector<float> y = mesh.getElement("vertex").getProperty<float>("y");
	std::vector<float> z = mesh.getElement("vertex").getProperty<float>("z");
	std::vector< std::vector<int> > vertex_indices = mesh.getElement("face").getListProperty<int>("vertex_indices");
	std::cout << "Number of Vertices: " << x.size() << std::endl;
	std::cout << "Number of Faces:    " << vertex_indices.size() << std::endl;

	

	Eigen::MatrixXd P;
	readFileToMat(point_filename, P);
	int num_points = P.rows();

	// Construct AABB-tree-readable triangular mesh
	Eigen::MatrixXd V(x.size(), 3);
	for (int i = 0; i < x.size(); ++i) {
		V(i, 0) = x[i];
		V(i, 1) = y[i];
		V(i, 2) = z[i];
	}

	Eigen::MatrixXi F(vertex_indices.size(), 3);
	for (int i = 0; i < vertex_indices.size(); ++i) {
		F(i, 0) = vertex_indices[i][0];
		F(i, 1) = vertex_indices[i][1];
		F(i, 2) = vertex_indices[i][2];
	}

	clock_t t_beg = std::clock();
	igl::AABB<Eigen::MatrixXd, 3> tree;
	tree.init(V, F);
	clock_t t_end = std::clock();
	double elapsed_secs = double(t_end - t_beg) / 1000000;
	std::cout << "Constructing the AABB tree takes " << elapsed_secs << " seconds." << std::endl;

	//
	t_beg = std::clock();
	Eigen::VectorXd D;
	Eigen::VectorXi I;
	Eigen::MatrixXd C;
	tree.squared_distance(V, F, P, D, I, C);
	t_end = std::clock();
	elapsed_secs = double(t_end - t_beg) / 1000000;
	std::cout << "Calculation takes " << elapsed_secs << " seconds for " << num_points << " points." << std::endl;

	////////////////////////////////////////// is_int, half_idx, sqrt_val
	t_beg = std::clock();
	writeMatToFile(outfile + "/point_coordinate.dat"     , P, false, false, false);
	writeMatToFile(outfile + "/distance_to_mesh.dat"     , D, false, false, true );
	writeMatToFile(outfile + "/nearest_point_in_mesh.dat", C, false, false, false);
	writeMatToFile(outfile + "/nearest_face_index.dat"   , I, true , false, false);
	t_end = std::clock();
	elapsed_secs = double(t_end - t_beg) / 1000000;
	std::cout << "Wrting to file takes " << elapsed_secs << " seconds." << std::endl;

	if (vis != 0) {
		std::vector<float> xxx, yyy, zzz;
		for (int i = 0; i < vis; ++i) {
			xxx.push_back(P(i, 0));
			yyy.push_back(P(i, 1));
			zzz.push_back(P(i, 2));
			xxx.push_back(C(i, 0));
			yyy.push_back(C(i, 1));
			zzz.push_back(C(i, 2));
			xxx.push_back(C(i, 0));
			yyy.push_back(C(i, 1));
			zzz.push_back(C(i, 2));
		}
		std::vector< std::vector<int> > fff(vis, std::vector<int>(3, 0));
		for (int i = 0; i < vis; ++i) {
			fff[i][0] = i * 3;
			fff[i][1] = i * 3 + 1;
			fff[i][2] = i * 3 + 2;
		}

		happly::PLYData plyOut;

		// Add elements
		plyOut.addElement("vertex", vis * 3);
		plyOut.addElement("face", vis);

		// Add properties to those elements
		plyOut.getElement("vertex").addProperty<float>("x", xxx);
		plyOut.getElement("vertex").addProperty<float>("y", yyy);
		plyOut.getElement("vertex").addProperty<float>("z", zzz);
		plyOut.getElement("face").addListProperty<int>("vertex_indices", fff);

		// Write the object to file
		plyOut.write(outfile + "/mesh_for_vis.ply", happly::DataFormat::Binary);
	}

	return 0;
}
