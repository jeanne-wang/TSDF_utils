import sys
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np

sys.argv[1:] = ["build_ext", "--inplace"]

ext_modules = [
    Extension("color_fusion_volume", ["color_fusion_volume.pyx"],
    		  include_dirs=[np.get_include()]),
]

setup(ext_modules=cythonize(ext_modules))
