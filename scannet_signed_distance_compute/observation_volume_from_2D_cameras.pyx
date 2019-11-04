#cython: boundscheck=False
#cython: initializedcheck=False
#cython: cdivision=True

import numpy as np
cimport numpy as np
from libc.math cimport round
from libc.stdio cimport printf

cdef class ObservationVolume:

    cdef float[:, ::1] coords
    cdef int [::1] front_of_camera ## binary 1/0 
    cdef int [::1] behind_of_camera 

    def __init__(self, coords):

        self.coords = coords.astype(np.float32)

        num_point = self.coords.shape[0] 
        assert self.coords.shape[1] == 3
        self.front_of_camera = np.zeros([num_point],
                               dtype=np.int32)
        self.behind_of_camera = np.zeros([num_point],
                               dtype=np.int32)

    def get_volume(self):
        return np.array(self.front_of_camera), np.array(self.behind_of_camera)

    def fuse(self,
             np.float32_t[:, ::1] depth_proj_matrix,
             np.float32_t[:, ::1] depth_map):


        cdef float x, y, z
        cdef float depth_proj_x, depth_proj_y, depth_proj_z
        cdef int depth_image_proj_x, depth_image_proj_y
        cdef float depth
 

        for i in range(self.coords.shape[0]):

            if self.front_of_camera[i] == 1:
                continue
            x = self.coords[i, 0]
            y = self.coords[i, 1]
            z = self.coords[i, 2]


           # Compute the depth of the current voxel wrt. the camera.
           depth_proj_z = depth_proj_matrix[2, 0] * x + \
                          depth_proj_matrix[2, 1] * y + \
                          depth_proj_matrix[2, 2] * z + \
                          depth_proj_matrix[2, 3]

            # Check if voxel behind camera.
            if depth_proj_z <= 0:
                continue

            # Compute pixel location of the current voxel in the image.
            depth_proj_x = depth_proj_matrix[0, 0] * x + \
                           depth_proj_matrix[0, 1] * y + \
                           depth_proj_matrix[0, 2] * z + \
                           depth_proj_matrix[0, 3]
                    
            depth_proj_y = depth_proj_matrix[1, 0] * x + \
                           depth_proj_matrix[1, 1] * y + \
                           depth_proj_matrix[1, 2] * z + \
                           depth_proj_matrix[1, 3]

            depth_image_proj_x = <int>round(depth_proj_x / depth_proj_z)
            depth_image_proj_y = <int>round(depth_proj_y / depth_proj_z)

            # Check if projection is inside image.
            if (depth_image_proj_x < 0 or depth_image_proj_y < 0 or
                depth_image_proj_x >= depth_map.shape[1] or
                depth_image_proj_y >= depth_map.shape[0]):
                continue

            # Extract measured depth at projection.
            depth = depth_map[depth_image_proj_y, depth_image_proj_x]

            
            if depth_proj_z <= depth:
                self.front_of_camera[i] = 1
            else:
              self.behind_of_camera[i] += 1

                   


                   

                  