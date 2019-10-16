#cython: boundscheck=False
#cython: initializedcheck=False
#cython: cdivision=True

import numpy as np
cimport numpy as np
from libc.math cimport round
from libc.stdio cimport printf


cdef class ColorSDFVolume:

    cdef float[:, ::1] bbox
    cdef float resolution
    cdef float max_distance
    cdef int viewport_width
    cdef int viewport_height
    cdef float[:, :, :, ::1] volume
    cdef float[:, :, ::1] sdf_weight_data
    cdef float[:, :, ::1] color_weight_data

    def __init__(self, bbox, viewport_height, viewport_width, resolution, resolution_factor):
        assert resolution > 0
        assert resolution_factor > 0
        self.max_distance = resolution * resolution_factor

        self.bbox = bbox.astype(np.float32)
        self.resolution = resolution
        self.viewport_height = viewport_height
        self.viewport_width = viewport_width

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
             np.float32_t[:, ::1] transform_matrix,
             np.float32_t[:, ::1] depth_map,
             np.float32_t[:, :, ::1] color_map):

        assert color_map.shape[2] == self.volume.shape[3]-1

        cdef int i, j, k
        cdef float x, y, z
        cdef float x_clip, y_clip, z_clip, w_clip
        cdef float x_ndc, y_ndc
        cdef int x_screen, y_screen

    
        cdef float depth ## measured depth in depth map
        cdef float signed_distance, truncated_signed_distance
        cdef float prior_sdf_weight, prior_color_weight,new_sdf_weight, new_color_weight

        for i in range(self.volume.shape[0]):
            x = self.bbox[0, 0] + i * self.resolution
            for j in range(self.volume.shape[1]):
                y = self.bbox[1, 0] + j * self.resolution
                for k in range(self.volume.shape[2]):
                    z = self.bbox[2, 0] + k * self.resolution

                    
                    # compute the coords in the clip volume
                    x_clip = transform_matrix[0, 0] * x + \
                                   transform_matrix[0, 1] * y + \
                                   transform_matrix[0, 2] * z + \
                                   transform_matrix[0, 3]

                    y_clip = transform_matrix[1, 0] * x + \
                                   transform_matrix[1, 1] * y + \
                                   transform_matrix[1, 2] * z + \
                                   transform_matrix[1, 3]

                    z_clip = transform_matrix[2, 0] * x + \
                                   transform_matrix[2, 1] * y + \
                                   transform_matrix[2, 2] * z + \
                                   transform_matrix[2, 3]

                    w_clip = transform_matrix[3, 0] * x + \
                                   transform_matrix[3, 1] * y + \
                                   transform_matrix[3, 2] * z + \
                                   transform_matrix[3, 3]

                    
                    # ignore invisible point which has positive z value
                    if w_clip <= 0:
                        continue

                    # clip points outside of view frustum
                    if (x_clip < -w_clip or x_clip > w_clip or y_clip < -w_clip or
                        y_clip > w_clip or z_clip < -w_clip or z_clip > w_clip):
                        continue


                    # compute ndc coords
                    x_ndc = x_clip/w_clip
                    y_ndc = y_clip/w_clip

                    ## compute viewport transform (assume the position of viewport is (0,0))
                    x_screen = <int>round((self.viewport_width * x_ndc + self.viewport_width)*0.5)
                    y_screen = <int>round((self.viewport_height * y_ndc + self.viewport_height)*0.5)
                    printf("x_screen: %d, y_screen: %d\n", x_screen, y_screen)


                    # Extract depth of visible surface
                    depth = depth_map[y_screen, x_screen]

                    # w_clip = -z_e, w_clip is the distance between the point to the camera in the camera coords space
                    signed_distance = depth-w_clip
                    #printf("depth: %f\n", depth)
                    #printf("w_clip: %f\n", w_clip)

                    ## sdf fusion
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
                        self.volume[i, j, k, ch] = min((prior_color_weight * self.volume[i,j,k,ch] + 1.0 *  color_map[y_screen, x_screen, ch])/new_color_weight, 255.0)

                    self.color_weight_data[i,j,k] = new_color_weight

