from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path
from types import ModuleType

import pytest


def _load_setup_binary_module() -> ModuleType:
    module_path = Path(__file__).resolve().parent.parent / "scripts" / "setup_binary.py"
    spec = importlib.util.spec_from_file_location("setup_binary", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def setup_binary_module() -> ModuleType:
    return _load_setup_binary_module()


def _missing_version_error(spec: str) -> subprocess.CalledProcessError:
    return subprocess.CalledProcessError(
        returncode=1,
        cmd=["npm", "pack", spec],
        stderr=f"npm error code ETARGET\nnpm error notarget No matching version found for {spec}.\n",
    )


def _other_npm_error(spec: str) -> subprocess.CalledProcessError:
    return subprocess.CalledProcessError(
        returncode=1,
        cmd=["npm", "pack", spec],
        stderr=f"npm error code E403\nnpm error 403 Forbidden while fetching {spec}.\n",
    )


def test_resolve_codex_sdk_npm_specs_prefers_exact_version(
    monkeypatch: pytest.MonkeyPatch, setup_binary_module: ModuleType
) -> None:
    monkeypatch.setattr(
        setup_binary_module, "read_pyproject_version", lambda _path: "0.114.1"
    )

    assert setup_binary_module.resolve_codex_sdk_npm_specs() == [
        "@openai/codex-sdk@0.114.1",
        "@openai/codex-sdk",
    ]


def test_npm_pack_codex_sdk_package_falls_back_when_exact_version_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, setup_binary_module: ModuleType
) -> None:
    calls: list[str] = []

    def fake_run_command(cmd: list[str], cwd: Path | None = None, check: bool = True):
        spec = cmd[-1]
        calls.append(spec)
        if spec.endswith("@0.114.1"):
            raise _missing_version_error(spec)
        return subprocess.CompletedProcess(cmd, 0, stdout="packed\n", stderr="")

    monkeypatch.setattr(setup_binary_module, "run_command", fake_run_command)

    used_spec = setup_binary_module.npm_pack_codex_sdk_package(
        tmp_path,
        ["@openai/codex-sdk@0.114.1", "@openai/codex-sdk"],
    )

    assert used_spec == "@openai/codex-sdk"
    assert calls == ["@openai/codex-sdk@0.114.1", "@openai/codex-sdk"]


def test_npm_pack_codex_sdk_package_does_not_hide_other_npm_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, setup_binary_module: ModuleType
) -> None:
    def fake_run_command(cmd: list[str], cwd: Path | None = None, check: bool = True):
        raise _other_npm_error(cmd[-1])

    monkeypatch.setattr(setup_binary_module, "run_command", fake_run_command)

    with pytest.raises(subprocess.CalledProcessError):
        setup_binary_module.npm_pack_codex_sdk_package(
            tmp_path,
            ["@openai/codex-sdk@0.114.1", "@openai/codex-sdk"],
        )
