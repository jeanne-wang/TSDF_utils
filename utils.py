import sys
import numpy as np


def read_gvr_groudtruth(path):
    with open(path, "rb") as fid:
        version = np.fromfile(fid, count=1, dtype=np.uint8)
        assert version == 1

        is_big_endian = np.fromfile(fid, count=1, dtype=np.uint8)
        assert (is_big_endian == 1 and sys.byteorder == "big") or \
               (is_big_endian == 0 and sys.byteorder == "little")

        uint_size = np.fromfile(fid, count=1, dtype=np.uint8)
        assert uint_size == 4

        elem_size = np.fromfile(fid, count=1, dtype=np.uint32)
        if elem_size == 4:
            dtype = np.float32
        elif elem_size == 8:
            dtype = np.float64
        else:
            raise ValueError("Unsupported data type")

        num_labels = np.fromfile(fid, count=1, dtype=np.uint32)[0]
        width = np.fromfile(fid, count=1, dtype=np.uint32)[0]
        height = np.fromfile(fid, count=1, dtype=np.uint32)[0]
        depth = np.fromfile(fid, count=1, dtype=np.uint32)[0]

        num_elems = width * height * depth * num_labels
        assert num_elems > 0

        grid = np.fromfile(fid, count=num_elems, dtype=dtype)
        grid =  grid.reshape(depth, height, width, num_labels)

        grid = grid.transpose(2, 1, 0, 3)

        return grid


def read_gvr_datacost(path, num_labels):
    with open(path, "rb") as fid:
        version = np.fromfile(fid, count=1, dtype=np.uint8)
        assert version == 1

        is_big_endian = np.fromfile(fid, count=1, dtype=np.uint8)
        assert (is_big_endian == 1 and sys.byteorder == "big") or \
               (is_big_endian == 0 and sys.byteorder == "little")

        uint_size = np.fromfile(fid, count=1, dtype=np.uint8)
        assert uint_size == 4

        elem_size = np.fromfile(fid, count=1, dtype=np.uint32)
        if elem_size == 4:
            dtype = np.float32
        elif elem_size == 8:
            dtype = np.float64
        else:
            raise ValueError("Unsupported data type")

        width = np.fromfile(fid, count=1, dtype=np.uint32)[0]
        height = np.fromfile(fid, count=1, dtype=np.uint32)[0]
        depth = np.fromfile(fid, count=1, dtype=np.uint32)[0]
        assert (width % num_labels) == 0

        width //= num_labels

        num_elems = width * height * depth * num_labels
        assert num_elems > 0

        grid = np.fromfile(fid, count=num_elems, dtype=dtype)
        grid =  grid.reshape(depth, height, width, num_labels)

        grid = grid.transpose(2, 1, 0, 3)

        return grid
