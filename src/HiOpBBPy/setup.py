'''
This is the setup file for installing HiOpBB

Authors:    Tucker Hartland <hartland1@llnl.gov>
            Nai-Yuan Chiang <chiang7@llnl.gov>
'''

import sys
import numpy as np
from setuptools import setup

from hiopbb import __version__

metadata = dict(
        name="hiopbb",
        version=__version__,
        description="HiOp black box optimization (hiopbb)",
        author="Tucker hartland et al.",
        author_email="hartland1@llnl.gov",
        license="BSD-3",
        packages=[
            "hiopbb",
            "hiopbb.problems",
            "hiopbb.surrogate_modeling",
            "hiopbb.opt"
            ],
        install_requires=["smt"],
        python_requires=">=3.9",
        zip_safe=False,
        url="https://lc.llnl.gov/gitlab/ai4sci/hiopBBpy", #not public (for now)
        download_url="https://lc.llnl.gov/gitlab/ai4sci/hiopBBpy", # not a public url (for now)
)

setup(**metadata)
