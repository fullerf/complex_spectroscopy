#!/usr/bin/env python

import sys
from pathlib import Path
import os

from pkg_resources import parse_version
from setuptools import find_packages, setup

requirements = [
    "numpy>=1.16.2",
    "gpflow>=2.1.4",
    "tensorflow>=2.1.0",
    "tensorflow-probability>0.10.0",
    "setuptools>=41.0.0",  # to satisfy dependency constraints
    "multipledispatch>=0.6", 
    "tabulate",
    "typing_extensions",
    "packaging"
    'scipy>=1.2.1',
    'h5py>=2.9.0',
]

if sys.version_info < (3, 7):
    requirements.append("dataclasses")
    
def read_file(filename):
    with open(filename, encoding="utf-8") as f:
        return f.read().strip()
    
packages = find_packages(".", exclude=["tests", "notebooks"])

version = read_file("VERSION")
readme_text = read_file("README.md")
    
setup(
    name="complex_spectroscopy",
    version=version,
    author="Franklin Fuller",
    author_email="fdfulker@gmail.com",
    description="Complex Signal Regression built on GPFlow and Tensorflow",
    long_description=readme_text,
    long_description_content_type="text/markdown",
    license="Apache License 2.0",
    keywords="machine-learning gaussian-processes kernels tensorflow signal-analysis",
    url="https://github.com/fullerf/complex_spectroscopy",
    project_urls={
        "Source on GitHub": "https://github.com/fullerf/complex_spectroscopy",
    },
    packages=packages,
    include_package_data=True,
    install_requires=requirements,
    extras_require={"ImageToTensorBoard": ["matplotlib"]},
    python_requires=">=3.6",
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3.6",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)