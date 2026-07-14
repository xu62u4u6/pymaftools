"""Tests for the portable test runner."""

import sys

import run_tests as test_runner


def test_runner_uses_current_python_and_project_cwd(monkeypatch):
    captured = {}

    class Result:
        returncode = 0

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return Result()

    monkeypatch.setattr(test_runner.subprocess, "run", fake_run)

    result = test_runner.run_tests(test_type="core", verbose=False)

    assert result == 0
    assert captured["command"] == [sys.executable, "-m", "pytest", "tests/core/"]
    assert captured["kwargs"]["cwd"] == test_runner.Path(test_runner.__file__).parent
    assert captured["kwargs"]["check"] is False
