#cython: boundscheck=False
#cython: initializedcheck=False
#cython: cdivision=True

import numpy as np
cimport numpy as np
from libc.math cimport round


cdef class ColorSDFVolume:

    cdef float[:, ::1] bbox
    cdef float resolution
    cdef float max_distance
    cdef float[:, :, :, ::1] volume
    cdef float[:, :, ::1] sdf_weight_data
    cdef float[:, :, ::1] color_weight_data

    def __init__(self, bbox, resolution, resolution_factor):
        assert resolution > 0
        assert resolution_factor > 0
        

        self.bbox = bbox.astype(np.float32)
        self.resolution = resolution
        self.max_distance = resolution_factor * self.resolution


        volume_size = np.diff(bbox, axis=1)
        volume_shape = volume_size.ravel() / self.resolution
        volume_shape = np.ceil(volume_shape).astype(np.int32).tolist()
        self.volume = np.zeros(volume_shape + [4],
                               dtype=np.float32)

        # the last channel is for fused sdf, and we initialize it to truncated_distance, i.e., self.max_distance
        self.volume[:,:,:,-1] = self.max_distance 


        self.sdf_weight_data = np.zeros(volume_shape, dtype=np.float32)
        self.color_weight_data = np.zeros(volume_shape, dtype=np.float32)

    def get_volume(self):
        return np.array(self.volume)

    def fuse(self,
             np.float32_t[:, ::1] depth_proj_matrix,
             np.float32_t[:, ::1] color_proj_matrix,
             np.float32_t[:, ::1] depth_map,
             np.float32_t[:, :, ::1] color_map):
        assert color_map.shape[2] == self.volume.shape[3] - 1

        cdef int i, j, k
        cdef float x, y, z
        cdef float depth_proj_x, depth_proj_y, depth_proj_z
        cdef float color_proj_x, color_proj_y, color_proj_z
        cdef int depth_image_proj_x, depth_image_proj_y
        cdef int color_image_proj_x, color_image_proj_y
        cdef float depth, signed_distance, truncated_signed_distance
        cdef float prior_sdf_weight, prior_color_weight,new_sdf_weight, new_color_weight
        

        for i in range(self.volume.shape[0]):
            x = self.bbox[0, 0] + i * self.resolution
            for j in range(self.volume.shape[1]):
                y = self.bbox[1, 0] + j * self.resolution
                for k in range(self.volume.shape[2]):
                    z = self.bbox[2, 0] + k * self.resolution

                    # Compute the depth of the current voxel wrt. the camera.
                    depth_proj_z = depth_proj_matrix[2, 0] * x + \
                                   depth_proj_matrix[2, 1] * y + \
                                   depth_proj_matrix[2, 2] * z + \
                                   depth_proj_matrix[2, 3]
                    color_proj_z = color_proj_matrix[2, 0] * x + \
                                   color_proj_matrix[2, 1] * y + \
                                   color_proj_matrix[2, 2] * z + \
                                   color_proj_matrix[2, 3]

                    # Check if voxel behind camera.
                    if depth_proj_z <= 0 or color_proj_z <= 0:
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
                    color_proj_x = color_proj_matrix[0, 0] * x + \
                                   color_proj_matrix[0, 1] * y + \
                                   color_proj_matrix[0, 2] * z + \
                                   color_proj_matrix[0, 3]
                    color_proj_y = color_proj_matrix[1, 0] * x + \
                                   color_proj_matrix[1, 1] * y + \
                                   color_proj_matrix[1, 2] * z + \
                                   color_proj_matrix[1, 3]
                    depth_image_proj_x = <int>round(depth_proj_x / depth_proj_z)
                    depth_image_proj_y = <int>round(depth_proj_y / depth_proj_z)
                    color_image_proj_x = <int>round(color_proj_x / color_proj_z)
                    color_image_proj_y = <int>round(color_proj_y / color_proj_z)

                    # Check if projection is inside image.
                    if (depth_image_proj_x < 0 or depth_image_proj_y < 0 or
                        depth_image_proj_x >= depth_map.shape[1] or
                        depth_image_proj_y >= depth_map.shape[0] or
                        color_image_proj_x < 0 or color_image_proj_y < 0 or
                        color_image_proj_x >= color_map.shape[1] or
                        color_image_proj_y >= color_map.shape[0]):
                        continue

                    # Extract measured depth at projection.
                    depth = depth_map[depth_image_proj_y, depth_image_proj_x]

                    
                    signed_distance = depth - depth_proj_z

                    # sdf fusion 
                    if signed_distance >= -self.max_distance:

                        if signed_distance > 0:
                            truncated_distance = min(signed_distance, self.max_distance)
                        else:
                            truncated_distance = signed_distance
                        
                        prior_sdf_weight = self.sdf_weight_data[i, j, k]
                        new_sdf_weight = prior_sdf_weight+1.0
                        self.volume[i,j,k,-1] = (prior_sdf_weight * self.volume[i,j,k,-1] + 1.0  * truncated_distance)/new_sdf_weight
                        self.sdf_weight_data[i, j, k] = new_sdf_weight

                    # color fusion
                    if abs(signed_distance) > self.max_distance:
                        continue

                    prior_color_weight = self.color_weight_data[i,j,k]
                    new_color_weight = prior_color_weight+1.0

                    for ch in range(color_map.shape[2]):
                        self.volume[i, j, k, ch] = min((prior_color_weight * self.volume[i,j,k,ch] + 1.0 *  color_map[color_image_proj_y, color_image_proj_x, ch])/new_color_weight, 255.0)

                    self.color_weight_data[i,j,k] = new_color_weight

