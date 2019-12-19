from setuptools import setup, Extension
import numpy as np

VERSION_MAJOR = 0
VERSION_MINOR = 1
VERSION_POINT = 0
VERSION_DEV = 1

VERSION = "%d.%d.%d" % (VERSION_MAJOR, VERSION_MINOR, VERSION_POINT)
if VERSION_DEV:
    VERSION = VERSION + ".dev%d" % VERSION_DEV

SCRIPTS = ["frb_eyra_analysis/blind_detection.py", ]


setup(
    name = 'frb_eyra_analysis',
    version = VERSION,
    packages = ['frb_eyra_analysis'],
    scripts = SCRIPTS,
    install_requires = ['numpy', 'matplotlib'],

    # metadata for upload to PyPI
    author = "Liam Connor",
    author_email = "liam.dean.connor@gmail.com",
    description = "Fast radio burst benchmarking tools",
    license = "GPL v2.0",
    url = "http://github.com/liamconnor/frb-eyra-analysis"
)
