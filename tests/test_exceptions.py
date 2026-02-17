from codex_sdk.exceptions import CodexCLIError


def test_codex_cli_error_without_stderr() -> None:
    """Cover the branch where stderr is empty and the message contains only the exit code."""
    exc = CodexCLIError(exit_code=2, stderr="")
    assert "code 2" in str(exc)
