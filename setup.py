# Setup script for the library# setup.py
from setuptools import setup, find_packages
setup(
    name="pydart",
    version="0.1.0",
    description="DP-based Partitioning & Scheduling for Deep Learning Inference",
    author="Parth Shinde",
    packages=find_packages(),
    install_requires=[
        # We omit torch here to allow separate CPU/GPU installation
        "networkx",
        "pandas",
        "matplotlib",
        "viztracer"
    ],
    python_requires=">=3.7",
)
