import sys
import os
import platform
import glob

from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext

is_mac = platform.system() == "Darwin"
is_linux = platform.system() == "Linux"


extra_compile_args = ["-O3", "-Wall", "-std=c++17", "-fPIC"]
os.environ["CC"] = "gcc"
os.environ["CXX"] = "g++"
if is_mac:
    os.environ["CC"] = "clang"
    os.environ["CXX"] = "clang++"
    omp_path = os.popen("brew --prefix libomp").read().strip()
    extra_compile_args += ["-Xpreprocessor", "-fopenmp", f"-I{omp_path}/include"]
    extra_link_args = [f"-L{omp_path}/lib", "-lomp", "-undefined", "dynamic_lookup"]
elif is_linux:
    extra_compile_args += ["-fopenmp"]
    extra_link_args = ["-fopenmp"]
else:
    raise NotImplementedError

import pybind11
sources = glob.glob("pyscf_helper/src/*.cpp")
pybind11_include = pybind11.get_include()

ext_modules = [
    Extension(
        name="pyscf_helper.libs",
        sources=sources,
        include_dirs=[
            pybind11_include,
            "pyscf_helper/src",
        ],
        language="c++",
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    )
]
req = ["numpy",
       "pyscf",
       "pybind11",]

setup(
    name="pyscf_helper",
    version="0.1.0",
    author="zbwu1996@gmail.com",
    description="Helper tools for PySCF",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Quantum-Chemistry-Group-BNU",
    packages=find_packages(),
    ext_modules=ext_modules,
    zip_safe=False,
    install_requires=req,
)
