"""Shared Ollama inference options with GPU auto-detection enabled.

Ollama automatically uses Metal on supported Apple Silicon when ``num_gpu`` is
omitted.  OrbixAI previously passed ``num_gpu=0`` everywhere, which explicitly
disabled that behavior.  Set OLLAMA_NUM_GPU only as a troubleshooting override;
normal runs should leave it unset.
"""

from __future__ import annotations

import os


def ollama_options(**options):
    override = os.environ.get("OLLAMA_NUM_GPU")
    if override is not None and override.strip():
        try:
            options["num_gpu"] = int(override)
        except ValueError:
            pass
    return options
