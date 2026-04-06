# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Cloudscalerl Environment."""

from .models import CloudScaleAction, CloudScaleObservation

__all__ = [
    "CloudScaleAction",
    "CloudScaleObservation",
    "CloudScaleEnv",
]


def __getattr__(name: str):
    if name == "CloudScaleEnv":
        from .client import CloudScaleEnv

        return CloudScaleEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
