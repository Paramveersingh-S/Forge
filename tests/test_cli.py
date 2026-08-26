"""Tests for the CLI commands."""

import os
import subprocess
import sys

env = os.environ.copy()
env["PYTHONIOENCODING"] = "utf-8"


class TestCLI:
    """Test CLI commands via subprocess invocation."""

    def test_version(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "forge", "--version"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            env=env,
        )
        assert result.returncode == 0
        assert "0.1.0" in result.stdout

    def test_help(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "forge", "--help"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            env=env,
        )
        assert result.returncode == 0
        assert "forge" in result.stdout.lower()

    def test_train_help(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "forge", "train", "--help"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            env=env,
        )
        assert result.returncode == 0
        assert "config" in result.stdout.lower()

    def test_doctor_runs(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "forge", "doctor"],
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        assert result.returncode == 0
        assert "Python" in result.stdout
