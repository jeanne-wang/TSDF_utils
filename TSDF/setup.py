import sys
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np

sys.argv[1:] = ["build_ext", "--inplace"]

ext_modules = [
    Extension("tsdf_volume", ["tsdf_volume.pyx"],
              include_dirs=[np.get_include()]),
    Extension("color_sdf_volume", ["color_sdf_volume.pyx"],
    		  include_dirs=[np.get_include()]),
    Extension("freespace_volume_from_2D_cameras", ["reespace_volume_from_2D_cameras.pyx"],
    		  include_dirs=[np.get_include()]),

]

setup(ext_modules=cythonize(ext_modules))
