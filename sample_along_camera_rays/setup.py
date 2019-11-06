import sys
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np

sys.argv[1:] = ["build_ext", "--inplace"]

ext_modules = [
    Extension("sample_along_camera_ray", ["sample_along_camera_ray.pyx"],
    		  include_dirs=[np.get_include()]),
    Extension("sample_along_camera_ray_from_depthmap", ["sample_along_camera_ray_from_depthmap.pyx"],
    		  include_dirs=[np.get_include()]),
]

setup(ext_modules=cythonize(ext_modules))
