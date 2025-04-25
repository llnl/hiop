"""
This file provides some helper functions for hiopbb.

Authors:    Tucker Hartland <hartland1@llnl.gov>
            Nai-Yuan Chiang <chiang7@llnl.gov>
"""

import numpy as np

def check_required_keys(user_dict, required_keys):
    for key in required_keys:
        if key not in user_dict:
            raise KeyError(f"Missing required key: '{key}'")

