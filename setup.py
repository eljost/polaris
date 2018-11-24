#!/usr/bin/env python3

from setuptools import setup, find_packages
import sys

if sys.version_info.major < 3:
    raise SystemExit("Python 3 is required!")

setup(
    name="polaris",
    version="0.1",
    description="Polarizabilities from finite difference calculations",
    maintainer="Johannes Steinmetzer",
    maintainer_email="johannes.steinmetzer@uni-jena.de",
    license="GPL 3",
    platforms=["unix"],
    packages=find_packages(),
    install_requires=[
        "jinja2",
        "pyyaml",
        "numpy",
    ],
    entry_points={
        "console_scripts": [
            "polaris = polaris.main:run",
        ]
    },
)
