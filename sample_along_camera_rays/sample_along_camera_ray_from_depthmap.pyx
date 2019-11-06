#cython: boundscheck=False
#cython: initializedcheck=False
#cython: cdivision=True

import numpy as np
cimport numpy as np
from libc.math cimport round
from libc.stdio cimport printf
from libc.math cimport isnan

cdef class Sampling:

    cdef int num_sample_per_frame
    cdef int num_valid_frame
    cdef float gaussian_variance
    cdef float[:, ::1] pts_along_camera_rays

    def __init__(self, num_frame, num_sample_per_frame, gaussian_variance=0.1):
        
        self.num_sample_per_frame = num_sample_per_frame
        assert self.num_sample_per_frame % 2 == 0

        self.gaussian_variance = gaussian_variance
     
        self.pts_along_camera_rays = np.zeros([num_sample_per_frame * num_frame, 3], dtype=np.float32)

        self.num_valid_frame = 0
        np.random.seed(0)
    
    def get_sampled_points(self):
        return np.array(self.pts_along_camera_rays)[:self.num_sample_per_frame*self.num_valid_frame]

    def sample(self,
             np.float32_t[:, ::1] depth_K_inv,
             np.float32_t[:, ::1] depth_extrinsics_matrix_inv,
             np.float32_t[:, ::1] depth_map):

        cdef int depth_image_proj_x, depth_image_proj_y
        cdef float depth
        cdef float depth_proj_x, depth_proj_y, depth_proj_z
        cdef float x_c, y_c, z_c
        cdef float x1_c, y1_c, z1_c, x2_c, y2_c, z2_c
        cdef float x1, y1, z1, x2, y2, z2
        
        self.num_valid_frame += 1

        for i in range(int(self.num_sample_per_frame/2)):

            while True:
                depth_image_proj_x = np.random.randint(depth_map.shape[1])
                depth_image_proj_y = np.random.randint(depth_map.shape[0])
                depth = depth_map[depth_image_proj_y, depth_image_proj_x]
                if (depth != 0 and not isnan(depth)):
                    break


            depth_proj_x = depth_image_proj_x*depth
            depth_proj_y = depth_image_proj_y*depth
            depth_proj_z = depth

            ## back projected to camera coords system
            x_c = depth_K_inv[0, 0] * depth_proj_x + \
                  depth_K_inv[0, 1] * depth_proj_y + \
                  depth_K_inv[0, 2] * depth_proj_z

            y_c = depth_K_inv[1, 0] * depth_proj_x + \
                  depth_K_inv[1, 1] * depth_proj_y + \
                  depth_K_inv[1, 2] * depth_proj_z

            z_c = depth_K_inv[2, 0] * depth_proj_x + \
                  depth_K_inv[2, 1] * depth_proj_y + \
                  depth_K_inv[2, 2] * depth_proj_z

            assert (abs(z_c-depth_proj_z) <= 1e-4)
            ## generate two points along camera ray
            z1_c = depth+np.random.normal(0, self.gaussian_variance)
            z2_c = depth+np.random.normal(0, self.gaussian_variance)

            x1_c = x_c*z1_c/depth
            y1_c = y_c*z1_c/depth

            x2_c = x_c*z2_c/depth
            y2_c = y_c*z2_c/depth

            ## back projected to world coords system
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

                
            self.pts_along_camera_rays[(self.num_valid_frame-1)*self.num_sample_per_frame+i*2,0] = x1
            self.pts_along_camera_rays[(self.num_valid_frame-1)*self.num_sample_per_frame+i*2,1] = y1
            self.pts_along_camera_rays[(self.num_valid_frame-1)*self.num_sample_per_frame+i*2,2] = z1

            self.pts_along_camera_rays[(self.num_valid_frame-1)*self.num_sample_per_frame+i*2+1,0] = x2
            self.pts_along_camera_rays[(self.num_valid_frame-1)*self.num_sample_per_frame+i*2+1,1] = y2
            self.pts_along_camera_rays[(self.num_valid_frame-1)*self.num_sample_per_frame+i*2+1,2] = z2






                   


                   

                  