# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for TensorlakeProvider. All tests use a fake tensorlake SDK."""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import pytest
from openenv.core.containers.runtime.tensorlake_provider import TensorlakeProvider

OPENENV_YAML = b"spec_version: 1\nname: echo\napp: server.app:app\nport: 8000\n"


def _make_sandbox(yaml_content=OPENENV_YAML):
    sandbox = MagicMock()
    if yaml_content is None:
        sandbox.read_file.side_effect = RuntimeError("not found")
    else:
        sandbox.read_file.return_value.value = yaml_content
    sandbox.start_process.return_value.pid = 42
    sandbox.update.return_value.ingress_endpoint = "https://sandbox.tensorlake.ai"
    sandbox.update.return_value.sandbox_id = "sbx-1"
    sandbox.get_process.return_value.status = "running"
    return sandbox


@pytest.fixture
def fake_sdk():
    """Install a fake ``tensorlake.sandbox`` module and yield its Sandbox mock."""
    tensorlake_mod = types.ModuleType("tensorlake")
    sandbox_mod = types.ModuleType("tensorlake.sandbox")
    sandbox_mod.Sandbox = MagicMock()
    sandbox_mod.Sandbox.create.return_value = _make_sandbox()
    sandbox_mod.sandbox_url_from_ingress_endpoint = lambda endpoint, sandbox_id, port: (
        f"https://{port}-{sandbox_id}.sandbox.tensorlake.ai"
    )
    tensorlake_mod.sandbox = sandbox_mod
    with patch.dict(
        sys.modules, {"tensorlake": tensorlake_mod, "tensorlake.sandbox": sandbox_mod}
    ):
        yield sandbox_mod.Sandbox


def test_start_container_launches_server_and_returns_public_url(fake_sdk):
    provider = TensorlakeProvider(image="echo-env", api_key="key", cpus=2)

    url = provider.start_container(env_vars={"A": "1"})

    assert url == "https://8000-sbx-1.sandbox.tensorlake.ai"
    create_kwargs = fake_sdk.create.call_args.kwargs
    assert create_kwargs["image"] == "echo-env"
    assert create_kwargs["api_key"] == "key"
    assert create_kwargs["cpus"] == 2
    sandbox = fake_sdk.create.return_value
    cmd, args = sandbox.start_process.call_args.args
    assert cmd == "sh"
    assert args[1] == (
        "cd /app/env && python -m uvicorn server.app:app --host 0.0.0.0 --port 8000"
    )
    assert sandbox.start_process.call_args.kwargs["env"] == {"A": "1"}
    sandbox.update.assert_called_once_with(
        allow_unauthenticated_access=True, exposed_ports=[8000]
    )


def test_explicit_cmd_skips_discovery(fake_sdk):
    provider = TensorlakeProvider(image="echo-env")

    provider.start_container(cmd="run-server")

    sandbox = fake_sdk.create.return_value
    sandbox.read_file.assert_not_called()
    assert sandbox.start_process.call_args.args[1] == ["-c", "run-server"]


def test_missing_openenv_yaml_terminates_sandbox(fake_sdk):
    fake_sdk.create.return_value = _make_sandbox(yaml_content=None)
    provider = TensorlakeProvider(image="echo-env")

    with pytest.raises(ValueError, match="cmd="):
        provider.start_container()

    fake_sdk.create.return_value.terminate.assert_called_once()


def test_rejects_bad_arguments(fake_sdk):
    provider = TensorlakeProvider()
    with pytest.raises(ValueError, match="requires an image"):
        provider.start_container()
    with pytest.raises(ValueError, match="port 8000"):
        provider.start_container("echo-env", port=9000)
    with pytest.raises(ValueError, match="Unsupported"):
        provider.start_container("echo-env", typo=True)
    fake_sdk.create.assert_not_called()


def test_accepts_autoenv_wait_timeout(fake_sdk):
    provider = TensorlakeProvider(image="echo-env")
    assert provider.start_container(wait_timeout=30.0).startswith("https://")


def test_stop_container_terminates_once(fake_sdk):
    provider = TensorlakeProvider(image="echo-env")
    provider.start_container()
    sandbox = fake_sdk.create.return_value

    provider.stop_container()
    provider.stop_container()

    sandbox.terminate.assert_called_once()


def test_failed_stop_can_be_retried(fake_sdk):
    provider = TensorlakeProvider(image="echo-env")
    provider.start_container()
    sandbox = fake_sdk.create.return_value
    sandbox.terminate.side_effect = [ConnectionError("network"), None]

    with pytest.raises(ConnectionError):
        provider.stop_container()
    provider.stop_container()

    assert sandbox.terminate.call_count == 2


def test_second_start_is_rejected(fake_sdk):
    provider = TensorlakeProvider(image="echo-env")
    provider.start_container()

    with pytest.raises(RuntimeError, match="already active"):
        provider.start_container()

    fake_sdk.create.assert_called_once()


def test_context_manager_exit_terminates_sandbox(fake_sdk):
    with TensorlakeProvider(image="echo-env") as provider:
        provider.start_container()

    fake_sdk.create.return_value.terminate.assert_called_once()


@pytest.mark.parametrize("surface", [False, True])
def test_wait_for_ready_reports_dead_server(fake_sdk, surface):
    import requests

    provider = TensorlakeProvider(image="echo-env", surface_server_logs=surface)
    url = provider.start_container(env_vars={"TOKEN": "s3cret"})
    sandbox = fake_sdk.create.return_value
    sandbox.get_process.return_value.status = "exited"
    sandbox.get_output.return_value.lines = ["token=s3cret", "ModuleNotFoundError"]

    with patch("requests.get", side_effect=requests.ConnectionError):
        with pytest.raises(RuntimeError, match="exited") as exc:
            provider.wait_for_ready(url, timeout_s=5)

    assert "s3cret" not in str(exc.value)
    assert ("ModuleNotFoundError" in str(exc.value)) is surface


def test_wait_for_ready_returns_on_healthy(fake_sdk):
    provider = TensorlakeProvider(image="echo-env")
    url = provider.start_container()

    with patch("requests.get", return_value=MagicMock(status_code=200)) as get:
        provider.wait_for_ready(url, timeout_s=5)

    get.assert_called_once_with(f"{url}/health", timeout=5.0)
