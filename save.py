import numpy as np
import struct
import sys

def write_dat_groundtruth(data, outfile):

    with open(outfile, "wb") as fid:
        version = 1
        fid.write(struct.pack("B", version))

        # Endianness
        endianness = (sys.byteorder == "big")
        fid.write(struct.pack("B", endianness))

        # store sizes of data types written
        uint_size = np.dtype(np.uint32).itemsize
        fid.write(struct.pack("B", uint_size))

        # Store size of array type
        elem_size = data.dtype.itemsize
        fid.write(struct.pack("I", elem_size))

        # Store size of the array
        data_shape = np.shape(data)
        fid.write(struct.pack("I", data_shape[3]))
        fid.write(struct.pack("I", data_shape[0]))
        fid.write(struct.pack("I", data_shape[1]))
        fid.write(struct.pack("I", data_shape[2]))
        
        # Write the grid to file
        data_trans = data.transpose(3, 0, 1, 2)
        fid.write(data_trans.tobytes())
        fid.close()

    return

def write_dat_datacost(data, outfile):

    with open(outfile, "wb") as fid:
        version = 1
        fid.write(struct.pack("B", version))

        # Endianness
        endianness = (sys.byteorder == "big")
        fid.write(struct.pack("B", endianness))

        # store sizes of data types written
        uint_size = np.dtype(np.uint32).itemsize
        fid.write(struct.pack("B", uint_size))

        # Store size of array type
        elem_size = data.dtype.itemsize
        fid.write(struct.pack("I", elem_size))

        # Store size of the array
        data_shape = np.shape(data)
        fid.write(struct.pack("I", data_shape[3] * data_shape[0]))
        fid.write(struct.pack("I", data_shape[1]))
        fid.write(struct.pack("I", data_shape[2]))
        
        # Write the grid to file
        data_trans = data.transpose(3, 0, 1, 2)
        fid.write(data_trans.tobytes())
        fid.close()

    return