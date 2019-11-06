#cython: boundscheck=False
#cython: initializedcheck=False
#cython: cdivision=True

import numpy as np
cimport numpy as np
from libc.math cimport round
from libc.stdio cimport printf

cdef class Sampling:

    cdef float observed_threshold
    cdef float gaussian_variance
    cdef float[:, ::1] pts_surf
    cdef float[:, ::1] pts_along_camera_rays
    cdef int [::1] observed ## binary 1/0 

    def __init__(self, pts_surf, observed_threshold=0.01, gaussian_variance=0.1):
        
        self.observed_threshold = observed_threshold
        self.gaussian_variance = gaussian_variance
        self.pts_surf = pts_surf.astype(np.float32)

        assert self.pts_surf.shape[1] == 3
        num_pts_surf = self.pts_surf.shape[0] 

        self.pts_along_camera_rays = np.zeros([num_pts_surf*2, 3], dtype=np.float32)
        self.observed = np.zeros([num_pts_surf],
                               dtype=np.int32)
        np.random.seed(0)
    
    def get_sampled_points(self):
        num_observed = np.sum(self.observed)
        return np.array(self.pts_along_camera_rays)[:num_observed*2]

    def sample(self,
             np.float32_t[:, ::1] depth_extrinsics_matrix,
             np.float32_t[:, ::1] depth_extrinsics_matrix_inv,
             np.float32_t[:, ::1] depth_proj_matrix,
             np.float32_t[:, ::1] depth_map):

        cdef float x, y, z
        cdef float x_c, y_c, z_c
        cdef float depth_proj_x, depth_proj_y, depth_proj_z
        cdef int depth_image_proj_x, depth_image_proj_y
        cdef float depth
        cdef float x1, y1, z1, x2, y2, z2
        cdef float x1_c, y1_c, z1_c, x2_c, y2_c, z2_c
        


        for i in range(self.pts_surf.shape[0]):

            if self.observed[i] == 1:
                continue
            x = self.pts_surf[i, 0]
            y = self.pts_surf[i, 1]
            z = self.pts_surf[i, 2]

            ## coordinates in camera space
            x_c = depth_extrinsics_matrix[0, 0] * x + \
                  depth_extrinsics_matrix[0, 1] * y + \
                  depth_extrinsics_matrix[0, 2] * z + \
                  depth_extrinsics_matrix[0, 3]

            y_c = depth_extrinsics_matrix[1, 0] * x + \
                  depth_extrinsics_matrix[1, 1] * y + \
                  depth_extrinsics_matrix[1, 2] * z + \
                  depth_extrinsics_matrix[1, 3]

            z_c = depth_extrinsics_matrix[2, 0] * x + \
                  depth_extrinsics_matrix[2, 1] * y + \
                  depth_extrinsics_matrix[2, 2] * z + \
                  depth_extrinsics_matrix[2, 3]

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

            
            if abs(depth-depth_proj_z) <= self.observed_threshold:
                self.observed[i] = 1
                num_observed = np.sum(self.observed)
              
                ## generate two points along camera ray
                z1_c = depth_proj_z+np.random.normal(0, self.gaussian_variance)
                z2_c = depth_proj_z+np.random.normal(0, self.gaussian_variance)

                x1_c = x_c*z1_c/depth_proj_z
                y1_c = y_c*z1_c/depth_proj_z

                x2_c = x_c*z2_c/depth_proj_z
                y2_c = y_c*z2_c/depth_proj_z

                x1 = depth_extrinsics_matrix_inv[0, 0] * x1_c + \
                     depth_extrinsics_matrix_inv[0, 1] * y1_c + \
                     depth_extrinsics_matrix_inv[0, 2] * z1_c + \
                     depth_extrinsics_matrix_inv[0, 3]

                y1 = depth_extrinsics_matrix_inv[1, 0] * x1_c + \
                     depth_extrinsics_matrix_inv[1, 1] * y1_c + \
                     depth_extrinsics_matrix_inv[1, 2] * z1_c + \
                     depth_extrinsics_matrix_inv[1, 3]

                z1 = depth_extrinsics_matrix_inv[2, 0] * x1_c + \
                     depth_extrinsics_matrix_inv[2, 1] * y1_c + \
                     depth_extrinsics_matrix_inv[2, 2] * z1_c + \
                     depth_extrinsics_matrix_inv[2, 3]

                x2 = depth_extrinsics_matrix_inv[0, 0] * x2_c + \
                     depth_extrinsics_matrix_inv[0, 1] * y2_c + \
                     depth_extrinsics_matrix_inv[0, 2] * z2_c + \
                     depth_extrinsics_matrix_inv[0, 3]

                y2 = depth_extrinsics_matrix_inv[1, 0] * x2_c + \
                     depth_extrinsics_matrix_inv[1, 1] * y2_c + \
                     depth_extrinsics_matrix_inv[1, 2] * z2_c + \
                     depth_extrinsics_matrix_inv[1, 3]

                z2 = depth_extrinsics_matrix_inv[2, 0] * x2_c + \
                     depth_extrinsics_matrix_inv[2, 1] * y2_c + \
                     depth_extrinsics_matrix_inv[2, 2] * z2_c + \
                     depth_extrinsics_matrix_inv[2, 3]

            
                
                self.pts_along_camera_rays[(num_observed-1)*2,0] = x1
                self.pts_along_camera_rays[(num_observed-1)*2,1] = y1
                self.pts_along_camera_rays[(num_observed-1)*2,2] = z1

                self.pts_along_camera_rays[(num_observed-1)*2+1,0] = x2
                self.pts_along_camera_rays[(num_observed-1)*2+1,1] = y2
                self.pts_along_camera_rays[(num_observed-1)*2+1,2] = z2



                   


                   

                  