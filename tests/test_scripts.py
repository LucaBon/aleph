"""Run the standalone e2e scripts and the canary benchmark under pytest.

They stay runnable on their own (``python tests/e2e_fake_llm.py``); this
wrapper makes ``pytest`` the single command that runs the whole suite.
Each script runs in its own subprocess, with the interpreter's bin dir first
on PATH so ``e2e_agent_mode.py`` finds the ``aleph`` console script.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

SCRIPTS = [
    "tests/e2e_fake_llm.py",
    "tests/e2e_agent_mode.py",
    "benchmark/canary.py",
]


@pytest.mark.parametrize("script", SCRIPTS)
def test_script(script: str) -> None:
    env = dict(os.environ)
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    env.pop("ANTHROPIC_API_KEY", None)  # these scripts must not need a live LLM
    proc = subprocess.run(
        [sys.executable, str(ROOT / script)],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=900,
    )
    if proc.returncode != 0:
        pytest.fail(
            f"{script} exited {proc.returncode}\n"
            f"--- stdout (tail) ---\n{proc.stdout[-4000:]}\n"
            f"--- stderr (tail) ---\n{proc.stderr[-4000:]}"
        )
