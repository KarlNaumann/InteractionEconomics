""" Cython setup and compilation file"""
from setuptools import setup
import numpy

try:
    from Cython.Build import cythonize
except ImportError:
    # create closure for deferred import
    def cythonize (*args, ** kwargs ):
        from Cython.Build import cythonize
        return cythonize(*args, ** kwargs)

ext_options = {"compiler_directives": {"profile": True}, "annotate": True}
setup(
    ext_modules = cythonize('step_functions.pyx', **ext_options),
    include_dirs=[numpy.get_include()]
)
