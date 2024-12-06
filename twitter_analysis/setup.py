from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        "twitter_cy",
        ["twitter_cy.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=["-O3"]
    )
]

setup(
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            'language_level': "3",
            'boundscheck': False,
            'wraparound': False,
        }
    )
)
