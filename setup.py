'''
This is the setup file for installing hiopbbpy

Authors:    Tucker Hartland <hartland1@llnl.gov>
            Nai-Yuan Chiang <chiang7@llnl.gov>
'''

import sys
import numpy as np
from setuptools import setup, find_packages


metadata = dict(
        name="hiopbbpy",
        version="0.0.4",
        description="HiOp black box optimization (hiopbbpy)",
        author="Tucker hartland et al.",
        author_email="hartland1@llnl.gov",
        license="BSD-3",
        packages=find_packages(where="src"),
        package_dir={"": "src"},
        install_requires=["smt","cyipopt"],
        python_requires=">=3.9",
        zip_safe=False,
        url="https://github.com/LLNL/hiop",
        download_url="https://github.com/LLNL/hiop",
)

setup(**metadata)
