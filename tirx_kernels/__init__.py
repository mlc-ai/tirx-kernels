# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""TIRX kernel library.

Private or experimental kernels can be layered on top of the released package by
setting ``TIRX_KERNELS_OVERLAY_PATHS`` to one or more ``tirx_kernels`` package
directories, separated by ``os.pathsep``.
"""

from __future__ import annotations

import os
from pathlib import Path

_OVERLAY_ENV = "TIRX_KERNELS_OVERLAY_PATHS"


def _iter_overlay_paths() -> list[Path]:
    paths = []
    raw = os.environ.get(_OVERLAY_ENV, "")
    for item in raw.split(os.pathsep):
        item = item.strip()
        if not item:
            continue
        path = Path(item).expanduser()
        if path.is_dir():
            paths.append(path.resolve())
    return paths


def _append_overlay_paths() -> None:
    seen = {str(Path(item).resolve()) for item in __path__}
    for path in _iter_overlay_paths():
        text = str(path)
        if text not in seen:
            __path__.append(text)
            seen.add(text)


_append_overlay_paths()
