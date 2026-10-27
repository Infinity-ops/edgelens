"""Inference markers: measure spans inside applications EdgeLens doesn't run
(C/C++ via edgelens_mark.h, Python via edgelens.mark, anything that can send
a UNIX datagram). See docs/markers.md."""

import os

from .protocol import DEFAULT_SOCKET, ENV_SOCKET, ITERATION, socket_path

INCLUDE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "include")
HEADER = os.path.join(INCLUDE_DIR, "edgelens_mark.h")

__all__ = ["DEFAULT_SOCKET", "ENV_SOCKET", "ITERATION", "socket_path", "INCLUDE_DIR", "HEADER"]
