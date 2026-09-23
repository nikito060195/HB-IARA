import glob
from setuptools import setup, Extension, find_packages
import pybind11

xdr_sources = glob.glob("backend/xdrfile*.c")

ext_modules = [
    Extension(
        "hbiara._core",  # Compilado como submódulo privado dentro de hbiara
        ["backend/hbond_core.cpp"] + xdr_sources,
        include_dirs=[pybind11.get_include(), ".", "backend"],
        language='c++',
        extra_compile_args=['-O3', '-march=native'],
    ),
]

setup(
    ext_modules=ext_modules,
    packages=find_packages(),
)
