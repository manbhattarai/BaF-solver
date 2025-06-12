from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize
from pathlib import Path
import platform

# The long description for the package on PyPI
# It's read from the README.md file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

link_args = []
# Platform-specific compile args
if platform.system() == "Windows":
    compile_args = ["/O2"]  # MSVC optimization
else:
    compile_args = ["-O3", "-Wall", "-Wextra"]

extensions = [
    Extension(
        "BaF_solver.interaction",
        sources=["BaF_solver/interaction.pyx"],
        extra_compile_args=compile_args,
    )
]

cython_compiler_directives = {
    'language_level': "3",
    'cdivision': True,
    'boundscheck': False,
    'wraparound': False,
    'nonecheck': False,
    'infer_types': True,
    'overflowcheck': False
}

setup(
    name="interaction",
    setup_requires=["cython"],
    install_requires=[
        "numpy", "scipy", "numba",
        "joblib",
        "sympy","symengine",
        "pywigxjpf",
    ],
    zip_safe=False, # Needed because we're compiling C extensions
    ext_modules=cythonize(
        extensions,
        compiler_directives=cython_compiler_directives,
    )
)
