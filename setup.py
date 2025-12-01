"""
Setup script for SFR Analyzer module
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="sfr-analyzer",
    version="1.0.0",
    author="Sam Lai",
    description="Professional SFR (Spatial Frequency Response) Analyzer for raw camera images",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/SFR-analyzer",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    python_requires=">=3.8",
    install_requires=[
        "PyQt5>=5.15.0",
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "opencv-python>=4.5.0",
        "matplotlib>=3.3.0",
        "pillow>=10.0.0",
    ],
    entry_points={
        "console_scripts": [
            "sfr-analyzer=SFR_app_v2:main",
        ],
    },
)

