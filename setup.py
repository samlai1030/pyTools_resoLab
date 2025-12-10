"""
Setup script for pyTools_ResoLab - Professional SFR/MTF Analyzer
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="pyTools-ResoLab",
    version="2.4.0",
    author="Sam Lai",
    author_email="samlai1030@gmail.com",
    description="Professional SFR/MTF Analyzer for raw camera images - ISO 12233:2023 compliant",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/samlai1030/pyTools_ResoLab",
    project_urls={
        "Bug Reports": "https://github.com/samlai1030/pyTools_ResoLab/issues",
        "Source": "https://github.com/samlai1030/pyTools_ResoLab",
    },
    packages=find_packages(),
    py_modules=["main", "mainUI"],
    include_package_data=True,
    package_data={
        "": ["*.ui", "*.json"],
    },
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
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Multimedia :: Graphics :: Capture :: Digital Camera",
    ],
    keywords="sfr, mtf, image-analysis, camera, iso12233, image-quality, raw-image",
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "pytools-resolab=main:main",
        ],
        "gui_scripts": [
            "pytools-resolab-gui=main:main",
        ],
    },
)

