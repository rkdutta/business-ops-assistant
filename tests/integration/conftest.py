"""Availability checks for the real local dependencies integration tests
exercise (Ollama, Docker). Each test that needs one calls the matching
fixture explicitly and gets skipped — not failed — if that dependency
isn't up, since these are meant to run against a dev machine's local
services, not a hermetic CI sandbox."""

import subprocess

import pytest
import requests


@pytest.fixture(scope="session")
def require_ollama():
    try:
        requests.get("http://localhost:11434/api/tags", timeout=2).raise_for_status()
    except requests.RequestException:
        pytest.skip("Ollama isn't reachable on localhost:11434")


@pytest.fixture(scope="session")
def require_docker():
    result = subprocess.run(["docker", "info"], capture_output=True, timeout=10)
    if result.returncode != 0:
        pytest.skip("Docker daemon isn't running")
