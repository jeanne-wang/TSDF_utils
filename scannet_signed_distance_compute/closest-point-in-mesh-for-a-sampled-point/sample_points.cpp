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

int main(int argc, char** argv) {

	// Parse arguments
	std::string line_break = "\n                           ";
	ArgumentParser parser("Argument parser example");
	parser.add_argument("-d", "--density", "int/float" + line_break +
										   "if > 0, number of sampling points PER UNIT VOLUME" + line_break +
										   "if < 0, total number of sampling points (absolute value)" + line_break, true);
	parser.add_argument("-f", "--file", "string, filename (.ply) of the input mesh" + line_break, true);
	parser.add_argument("-o", "--output", "string, prefix of the output (.dat) filename" + line_break, true);
	parser.add_argument("-p", "--padding", "float, padding size of the bounding" + line_break +
										   "box where the point is sampling" + line_break +
										   "if > 0, relative padding, +-padding * dim_size" + line_break +
										   "if <= 0, absolute padding, +-|padding|" + line_break +
										   "each dimension is the same" + line_break +
										   "default: 0", false);
	parser.add_argument("-s", "--seed", "int, seed of random number generator" + line_break +
										"default: 7", false);
	parser.add_argument("-v", "--visualize", "int, number of points to visualize" + line_break +
											 "if > 0, number of visualized points PER UNIT VOLUME" + line_break +
											 "if < 0, total number of visualized points (absolute value)" + line_break +
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
	std::string filename = parser.get<std::string>("f");
	std::string outfile = parser.get<std::string>("o");
	float density = parser.get<float>("d");
	if (density == 0) {
		std::cout << "Density cannot be 0." << std::endl;
		return 0;
	}
	float padding;
	if (parser.exists("p")) {
		padding = parser.get<float>("p");
	} else {
		padding = 0;
	}
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

	/*
	std::cout << "Density:  " << density << std::endl;
	std::cout << "Filename: " << filename << std::endl;
	std::cout << "Padding:  " << padding << std::endl;
	std::cout << "Seed:     " << seed << std::endl;
	std::cout << "\nPress ENTER to continue.";
	std::cin.ignore();
	std::cout << "ENTER pressed.\n";
	*/

	// Initialize random number generator
	std::mt19937 gen(seed);
	std::uniform_real_distribution<float> unif(0.0, 1.0);

	/*
	for (int n = 0; n < 10; ++n) {
		std::cout << unif(gen) << std::endl;
	}
	*/

	// Construct the data object by reading from file
	happly::PLYData mesh(filename, false);

	// Get mesh-style data from the object
	std::vector<float> x = mesh.getElement("vertex").getProperty<float>("x");
	std::vector<float> y = mesh.getElement("vertex").getProperty<float>("y");
	std::vector<float> z = mesh.getElement("vertex").getProperty<float>("z");
	std::vector< std::vector<int> > vertex_indices = mesh.getElement("face").getListProperty<int>("vertex_indices");

	// Check whether the mesh is correctly read
	/*
	for (int i = 0; i < 10; ++i) {
		std::cout << x[i] << " " << y[i] << " " << z[i] << std::endl;
	}

	for (int i = 0; i < 10; ++i) {
		for (int j = 0; j < vertex_indices[i].size(); ++j) {
			std::cout << vertex_indices[i][j] << " ";
		}
		std::cout << std::endl;
	}
	*/

	// Print the basic scene information
	float x_min = *std::min_element(x.begin(), x.end());
	float x_max = *std::max_element(x.begin(), x.end());
	float y_min = *std::min_element(y.begin(), y.end());
	float y_max = *std::max_element(y.begin(), y.end());
	float z_min = *std::min_element(z.begin(), z.end());
	float z_max = *std::max_element(z.begin(), z.end());
	float x_diff = x_max - x_min;
	float y_diff = y_max - y_min;
	float z_diff = z_max - z_min;
	float volume = x_diff * y_diff * z_diff;
	int num_points;
	if (density > 0) {
		num_points = std::ceil(volume * density);
	} else {
		num_points = std::ceil(-density);
	}

	std::cout << "\nMesh Information:\n        Min      Max     Diff" << std::endl;
	std::cout << std::fixed << std::setprecision(5) << "x  " << x_min << "  " << x_max << "  " << x_diff << std::endl;
	std::cout << std::fixed << std::setprecision(5) << "y  " << y_min << "  " << y_max << "  " << y_diff << std::endl;
	std::cout << std::fixed << std::setprecision(5) << "z  " << z_min << "  " << z_max << "  " << z_diff << std::endl;
	std::cout << std::fixed << std::setprecision(5) << "Volume:             " << volume << std::endl;
	std::cout << "Number of Vertices: " << x.size() << std::endl;
	std::cout << "Number of Faces:    " << vertex_indices.size() << std::endl;
	std::cout << std::fixed << "Totally " << num_points << " points will be sampled." << std::endl;
	// std::cout << "\nPress ENTER to continue.";
	// std::cin.ignore();
	// std::cout << "ENTER pressed.\n";

	float x_pad, y_pad, z_pad;
	if (padding <= 0) {
		x_pad = -padding;
		y_pad = -padding;
		z_pad = -padding;
	} else {
		x_pad = x_diff * padding;
		y_pad = y_diff * padding;
		z_pad = z_diff * padding;
	}
	float sx_min = x_min - x_pad;
	float sx_diff = x_max + x_pad - sx_min;
	float sy_min = y_min - y_pad;
	float sy_diff = y_max + y_pad - sy_min;
	float sz_min = z_min - z_pad;
	float sz_diff = z_max + z_pad - sz_min;

	Eigen::MatrixXd P(num_points, 3);
	for (int i = 0; i < num_points; ++i) {
		float rand_x = unif(gen);
		float rand_y = unif(gen);
		float rand_z = unif(gen);
		P(i, 0) = rand_x * sx_diff + sx_min;
		P(i, 1) = rand_y * sy_diff + sy_min;
		P(i, 2) = rand_z * sz_diff + sz_min;
	}
	
	// std::ofstream out_file;
	// out_file.open ("point_cloud.txt");
	// out_file << P;
	// out_file.close();
	// return 0;

	// Construct AABB-tree-readable triangular mesh
	Eigen::MatrixXd V(x.size(), 3);
	for (int i = 0; i < x.size(); ++i) {
		V(i, 0) = x[i];
		V(i, 1) = y[i];
		V(i, 2) = z[i];
	}

	Eigen::MatrixXi F(vertex_indices.size() * 2, 3);
	for (int i = 0; i < vertex_indices.size(); ++i) {
		F(i * 2, 0) = vertex_indices[i][0];
		F(i * 2, 1) = vertex_indices[i][1];
		F(i * 2, 2) = vertex_indices[i][2];
		F(i * 2 + 1, 0) = vertex_indices[i][0];
		F(i * 2 + 1, 1) = vertex_indices[i][2];
		F(i * 2 + 1, 2) = vertex_indices[i][3];
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
	writeMatToFile(outfile + "/point_coordinate.dat", P, false, false, false);
	writeMatToFile(outfile + "/distance_to_mesh.dat", D, false, false, true);
	writeMatToFile(outfile + "/nearest_point_in_mesh.dat", C, false, false, false);
	writeMatToFile(outfile + "/nearest_face_index.dat", I, true, true, false);
	t_end = std::clock();
	elapsed_secs = double(t_end - t_beg) / 1000000;
	std::cout << "Wrting to file takes " << elapsed_secs << " seconds." << std::endl;

	if (vis != 0) {
		if (vis > 0) {
			vis = std::ceil(volume * vis);
		} else {
			vis = -vis;
		}
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

	/*
	float step = 1.0; // 0.5 0.25 0.125 0.0625

	for (int s = 0; s < 1; ++s) {

		int x_beg = int(std::floor(x_min / step));
		int x_end = int(std::ceil(x_max / step));
		int y_beg = int(std::floor(y_min / step));
		int y_end = int(std::ceil(y_max / step));
		int z_beg = int(std::floor(z_min / step));
		int z_end = int(std::ceil(z_max / step));
		int nx = x_end - x_beg + 1;
		int ny = y_end - y_beg + 1;
		int nz = z_end - z_beg + 1;

		std::cout << x_beg << " " << x_end << " " << nx << std::endl;
		std::cout << y_beg << " " << y_end << " " << ny << std::endl;
		std::cout << z_beg << " " << z_end << " " << nz << std::endl;
		std::cout << "N: " << nx * ny * nz << std::endl;

		Eigen::MatrixXd P(nx * ny * nz, 3);

		int idx = 0;
		for (int i = x_beg; i <= x_end; ++i) {
			for (int j = y_beg; j <= y_end; ++j) {
				for (int k = z_beg; k <= z_end; ++k) {
					P(idx, 0) = i * step;
					P(idx, 1) = j * step;
					P(idx, 2) = k * step;
					++idx;
				}
			}
		}

		Eigen::VectorXd sqrD;
		Eigen::VectorXi I;
		Eigen::MatrixXd C;

		t_beg = std::clock();
		tree.squared_distance(V, F, P, sqrD, I, C);
		t_end = std::clock();

		elapsed_secs = double(t_end - t_beg) / 1000000;
		std::cout << "Calculation takes " << elapsed_secs << " seconds for " << idx << " points." << std::endl;

		// std::cout << sqrD << std::endl << std::endl;
		// std::cout << I << std::endl << std::endl;
		// std::cout << C << std::endl << std::endl;

		step /= 2.0;

	}
	*/

	return 0;

}
