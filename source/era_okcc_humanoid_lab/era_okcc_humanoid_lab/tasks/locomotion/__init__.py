# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Locomotion task package.

Import robot-specific subpackages so gym task registration executes at import time.
"""

from . import robots  # noqa: F401
