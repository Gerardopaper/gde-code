"""Shared filesystem paths for GDE Code configuration."""

from pathlib import Path

GDEC_CONFIG_DIRNAME = ".gdec"
GDEC_ENV_FILENAME = ".env"
CLAUDE_WORKSPACE_DIRNAME = "agent_workspace"
GDEC_LOGS_DIRNAME = "logs"
SERVER_LOG_FILENAME = "server.log"


def config_dir_path() -> Path:
    """Return the default user config directory."""

    return Path.home() / GDEC_CONFIG_DIRNAME


def managed_env_path() -> Path:
    """Return the default user-managed env file path."""

    return config_dir_path() / GDEC_ENV_FILENAME


def default_claude_workspace_path() -> Path:
    """Return the default Claude workspace path."""

    return config_dir_path() / CLAUDE_WORKSPACE_DIRNAME


def server_log_path() -> Path:
    """Return the canonical server log path."""

    return config_dir_path() / GDEC_LOGS_DIRNAME / SERVER_LOG_FILENAME
