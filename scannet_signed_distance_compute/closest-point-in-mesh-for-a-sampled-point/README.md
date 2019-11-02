# Closest Point In Mesh for a Sampled Point
### Install
[Eigen](https://github.com/eigenteam/eigen-git-mirror) and [libigl](https://github.com/libigl/libigl) are required.
### Compile
MacOS: `g++ -std=c++11 sample_points.cpp -o sample_points`

Ubuntu: `g++ -std=c++11 sample_points.cpp -o sample_points -lpthread`
### Example Run
`./sample_points -f "../Replica-Dataset/replica_v1/hotel_0/mesh.ply" -d -1000 -o hotel_0 -p 0.05 -v 1000`
