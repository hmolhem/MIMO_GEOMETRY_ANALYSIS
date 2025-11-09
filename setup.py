from setuptools import setup, find_packages

setup(
    name="mimo_geometry_analysis",
    version="0.1.0",
    description="MIMO antenna geometry analysis and visualization tool",
    author="MIMO Analysis Team",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "matplotlib>=3.5.0",
        "scipy>=1.7.0",
    ],
    python_requires=">=3.7",
)
