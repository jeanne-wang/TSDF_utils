import sys
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np

sys.argv[1:] = ["build_ext", "--inplace"]

ext_modules = [
    Extension("observation_volume_from_2D_cameras", ["observation_volume_from_2D_cameras.pyx"],
    		  include_dirs=[np.get_include()]),
]

setup(ext_modules=cythonize(ext_modules))
