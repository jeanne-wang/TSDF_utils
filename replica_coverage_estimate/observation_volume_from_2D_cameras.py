#cython: boundscheck=False
#cython: initializedcheck=False
#cython: cdivision=True

import numpy as np
cimport numpy as np
from libc.math cimport round


cdef class ObservedVolume:

    cdef int viewport_width
    cdef int viewport_height

    cdef float[:, ::1] coords
    cdef int [::1] observed ## binary 1/0 observed/not observed


    def __init__(self, viewport_height, viewport_width, np.float32_t[:, ::1] coords):
    
        self.viewport_height = viewport_height
        self.viewport_width = viewport_width

        num_point = coords.shape[0] 
        self.observed= np.zeros([num_point],
                               dtype=np.int32)

        self.coords = coords

    def get_volume(self):
        return np.array(self.observed)

    def fuse(self,
             np.float32_t[:, ::1] transform_matrix,
             np.float32_t[:, ::1] depth_map):


        cdef float x, y, z
        cdef float x_clip, y_clip, z_clip, w_clip
        cdef float x_ndc, y_ndc
        cdef int x_screen, y_screen

    
        cdef float depth ## measured depth in depth map

        for i in range(self.coords.shape[0]):

            if self.observed[i] == 1:
                continue
            x = self.coords[i, 0]
            y = self.coords[i, 1]
            z = self.coords[i, 2]


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

            # Extract depth of visible surface

            depth = depth_map[self.viewport_height-y_screen, x_screen] ## the index of y need to be flipped

            # w_clip = -z_e, w_clip is the distance between the point to the camera in the camera coords space
            if abs(w_clip-depth) <= 1e-2:
                self.observed[i] = 1

                   


                   

                  

