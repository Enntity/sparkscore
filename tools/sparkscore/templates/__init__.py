# SPDX-License-Identifier: AGPL-3.0-only
"""Versioned prompt templates, one module per VERSION ("knowline-gen1.5" is
knowline_gen15.py). A module defines VERSION, INSTRUCTION, CUE,
LABEL_PREFIX, text(), messages(), options() and question_block() for
render.Renderer, or VERSION and its own Renderer class with the same
interface (render(), labels, option_ids, option_text, source, version)."""

import importlib

DEFAULT = "chat-v1"


def load(version=DEFAULT):
    module = importlib.import_module(__name__ + "." + version.replace("-", "_").replace(".", ""))
    if module.VERSION != version:
        raise ValueError(f"template module {module.__name__} is {module.VERSION}, not {version}")
    return module
