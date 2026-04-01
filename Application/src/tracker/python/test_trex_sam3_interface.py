#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compatibility note for the refactored SAM3 adapter.

`trex_sam3_interface.py` no longer exposes the legacy standalone session API.
The supported surface is now:
- `create_session(request)`
- `set_conf_threshold(request)`
- `shutdown()`
- `predict(TRex.Sam3Input)`

This means standalone testing outside the embedded TRex runtime is no longer
supported by this helper script.
"""

from __future__ import annotations

import sys


def main() -> int:
    """Explain the new SAM3 adapter contract and exit."""
    sys.stderr.write(
        "test_trex_sam3_interface.py is obsolete after the SAM3 adapter refactor.\n"
        "Run SAM3 through the embedded TRex runtime using create_session(...) and\n"
        "predict(TRex.Sam3Input) instead of the removed set_frame/get_frame API.\n"
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
