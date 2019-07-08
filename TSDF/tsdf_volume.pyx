#cython: boundscheck=False
#cython: initializedcheck=False
#cython: cdivision=True

import numpy as np
cimport numpy as np
from libc.math cimport round


cdef class TSDFVolume:

    cdef float[:, ::1] bbox
    cdef float free_space_vote
    cdef float occupied_space_vote
    cdef float resolution
    cdef float max_distance
    cdef float[:, :, :, ::1] volume

    def __init__(self, num_labels, bbox, resolution, resolution_factor,
                 free_space_vote=0.5, occupied_space_vote=1):
        assert num_labels > 0
        assert resolution > 0
        assert resolution_factor > 0
        assert free_space_vote >= 0
        assert occupied_space_vote >= 0

        self.bbox = bbox.astype(np.float32)
        self.resolution = resolution
        self.max_distance = resolution_factor * self.resolution
        self.free_space_vote = free_space_vote
        self.occupied_space_vote = occupied_space_vote

        volume_size = np.diff(bbox, axis=1)
        volume_shape = volume_size.ravel() / self.resolution
        volume_shape = np.ceil(volume_shape).astype(np.int32).tolist()
        self.volume = np.zeros(volume_shape + [num_labels + 1],
                               dtype=np.float32)

    def get_volume(self):
        return np.array(self.volume)

    def fuse(self,
             np.float32_t[:, ::1] depth_proj_matrix,
             np.float32_t[:, ::1] label_proj_matrix,
             np.float32_t[:, ::1] depth_map,
             np.float32_t[:, :, ::1] label_map):
        assert label_map.shape[2] == self.volume.shape[3] - 1

        cdef int i, j, k
        cdef float x, y, z
        cdef float depth_proj_x, depth_proj_y, depth_proj_z
        cdef float label_proj_x, label_proj_y, label_proj_z
        cdef int depth_image_proj_x, depth_image_proj_y
        cdef int label_image_proj_x, label_image_proj_y
        cdef float depth, signed_distance
        cdef int label
        cdef float label_prob

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
                    label_proj_z = label_proj_matrix[2, 0] * x + \
                                   label_proj_matrix[2, 1] * y + \
                                   label_proj_matrix[2, 2] * z + \
                                   label_proj_matrix[2, 3]

                    # Check if voxel behind camera.
                    if depth_proj_z <= 0 or label_proj_z <= 0:
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
                    label_proj_x = label_proj_matrix[0, 0] * x + \
                                   label_proj_matrix[0, 1] * y + \
                                   label_proj_matrix[0, 2] * z + \
                                   label_proj_matrix[0, 3]
                    label_proj_y = label_proj_matrix[1, 0] * x + \
                                   label_proj_matrix[1, 1] * y + \
                                   label_proj_matrix[1, 2] * z + \
                                   label_proj_matrix[1, 3]
                    depth_image_proj_x = <int>round(depth_proj_x / depth_proj_z)
                    depth_image_proj_y = <int>round(depth_proj_y / depth_proj_z)
                    label_image_proj_x = <int>round(label_proj_x / label_proj_z)
                    label_image_proj_y = <int>round(label_proj_y / label_proj_z)

                    # Check if projection is inside image.
                    if (depth_image_proj_x < 0 or depth_image_proj_y < 0 or
                        depth_image_proj_x >= depth_map.shape[1] or
                        depth_image_proj_y >= depth_map.shape[0] or
                        label_image_proj_x < 0 or label_image_proj_y < 0 or
                        label_image_proj_x >= label_map.shape[1] or
                        label_image_proj_y >= label_map.shape[0]):
                        continue

                    # Extract measured depth at projection.
                    depth = depth_map[depth_image_proj_y, depth_image_proj_x]

                    # Check if voxel is inside the truncated distance field.
                    signed_distance = depth - depth_proj_z
                    if abs(signed_distance) > self.max_distance:
                        # Check if voxel is between observed depth and camera.
                        if signed_distance > 0:
                            # Vote for free space.
                            self.volume[i, j, k, -1] -= self.free_space_vote
                        continue

                    # Accumulate the votes for each label.
                    for label in range(label_map.shape[2]):
                        label_prob = label_map[label_image_proj_y,
                                               label_image_proj_x,
                                               label]
                        if signed_distance < 0:
                            self.volume[i, j, k, label] -= \
                                label_prob * self.occupied_space_vote
                        else:
                            self.volume[i, j, k, label] += \
                                label_prob * self.occupied_space_vote
