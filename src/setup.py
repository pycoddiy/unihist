import os
import subprocess
from setuptools import setup, find_packages

root_dir = os.path.dirname(os.path.abspath(__file__))
cmake_build_dir = os.path.join(root_dir, "..", "build")
cmake_dir = os.path.join(root_dir, ".")
install_dir = os.path.join(root_dir, "fast_histogram", "sycl")

DPCTL_MODULE_PATH = subprocess.check_output("python -m dpctl --cmakedir", shell=True)
cmake_cmd = ["cmake",
    cmake_dir,
    "-GNinja",
    "-DCMAKE_INSTALL_PREFIX=" + install_dir,
    "-DCMAKE_BUILD_TYPE=Release",
    "-DDPCTL_MODULE_PATH=" + DPCTL_MODULE_PATH.decode().strip(),
]

env = os.environ.copy()

env["CC"] = "icx"
env["CXX"] = "icpx"

subprocess.run(["mkdir", cmake_build_dir])
subprocess.run(["mkdir", install_dir])

subprocess.check_call(
    cmake_cmd, stderr=subprocess.STDOUT, shell=False, cwd=cmake_build_dir, env=env
)
subprocess.check_call(
    ["cmake", "--build", ".", "--config", "Release"], cwd=cmake_build_dir, env=env
)
subprocess.check_call(
    ["cmake", "--install", ".", "--config", "Release"], cwd=cmake_build_dir, env=env
)

setup(
    name='fast_histogram',
    version="1.0",
    packages=find_packages(where=root_dir, include=["*", "fast_histogram", "fast_histogram.*"])
)