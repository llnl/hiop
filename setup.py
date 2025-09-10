'''
This is the setup file for installing HiOpBBpy

Authors:    Tucker Hartland <hartland1@llnl.gov>
            Nai-Yuan Chiang <chiang7@llnl.gov>
'''

import sys
import os
import numpy as np
from setuptools import setup, find_packages


install_requires = ["smt"]

if os.getenv("ADD_CYIPOPT", "0") == "1":
    install_requires.append("cyipopt")

metadata = dict(
        name="hiopbbpy",
        version="1.2.0",
        description="HiOp black box optimization (HiOpBBpy)",
        author="Tucker hartland et al.",
        author_email="hartland1@llnl.gov",
        license="BSD-3",
        packages=find_packages(where="src"),
        package_dir={"": "src"},
        install_requires=install_requires,
        python_requires=">=3.9",
        zip_safe=False,
        url="https://github.com/LLNL/hiop",
        download_url="https://github.com/LLNL/hiop",
)

setup(**metadata)
