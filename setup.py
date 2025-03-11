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
        version="0.0.1",
        description="HiOp black box optimization (hiopbbpy)",
        author="Tucker hartland et al.",
        author_email="hartland1@llnl.gov",
        license="BSD-3",
        packages=find_packages(where="src"),
        package_dir={"": "src"},
        install_requires=["smt"],
        python_requires=">=3.9",
        zip_safe=False,
        url="https://lc.llnl.gov/gitlab/ai4sci/hiopBBpy", #not public (for now)
        download_url="https://lc.llnl.gov/gitlab/ai4sci/hiopBBpy", # not a public url (for now)
)

setup(**metadata)
