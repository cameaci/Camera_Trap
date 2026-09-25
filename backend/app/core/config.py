"""
Application configuration.

Following DEVELOPERS.md principles:
- Sensible defaults for bundled mode (PyInstaller)
- Can be overridden via environment variables
- Crash early if configuration is invalid
- Type hints everywhere

Every setting reads its env var with the WSP_ prefix
(WSP_USER_DATA_DIR, WSP_DATABASE_URL, ...). Unprefixed names
like DATABASE_URL are common enough in other tooling that a stray
export would silently redirect the app.
"""

from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# The per-user data folder: database, installed models, envs, logs.
USER_DATA_DIR_NAME = "WSP-CameraTrap"
# The repository this app is built and released from.
PROJECT_URL = "https://github.com/cameaci/Camera_Trap"
# The release in that repository that holds the runtime downloads: the
# prebuilt analysis environments and the MegaDetector weights.
RUNTIME_RELEASE_URL = f"{PROJECT_URL}/releases/download/runtime"


def get_default_user_data_dir() -> Path:
    """Get default user data directory."""
    return Path.home() / USER_DATA_DIR_NAME


def get_default_database_url() -> str:
    """Get default database URL in user's home directory."""
    db_path = get_default_user_data_dir() / "wsp-cameratrap.db"
    return f"sqlite:///{db_path}"


def get_default_models_dir() -> Path:
    """Get default models directory."""
    return get_default_user_data_dir() / "models"


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.

    Provides sensible defaults for bundled mode but can be overridden.
    Crashes if directories cannot be created.
    """

    model_config = SettingsConfigDict(
        env_prefix="WSP_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="forbid",  # Crash if unknown env vars are provided
    )

    # Application
    app_name: str = "WSP CameraTrap"  # WSP
    environment: Literal["development", "production", "test"] = "development"
    debug: bool = True
    # SQL statement echo is separate from `debug`: it logs every query +
    # params, which floods the log (and stalls the single-process server
    # on its synchronous writes) during bulk work like an analysis. Off
    # by default; flip it on only when actually debugging SQL.
    sql_echo: bool = False

    # API
    api_host: str = "127.0.0.1"
    api_port: int = 8000

    # Database - defaults to local SQLite in working directory
    database_url: str = Field(default_factory=get_default_database_url)

    # User data directory - defaults to ~/WSP-CameraTrap
    user_data_dir: Path = Field(default_factory=get_default_user_data_dir)

    # Redis
    redis_host: str = "127.0.0.1"
    redis_port: int = 6379

    # Models directory
    models_dir: Path = Field(default_factory=get_default_models_dir)

    # The WSP model library folder (a synced SharePoint/OneDrive folder or
    # a network share). Unset means the folder saved in the app, then the
    # usual OneDrive locations (see app/ml/model_library.py).
    model_library_dir: Path | None = None
    model_library_autodetect: bool = True

    # A OneDrive/SharePoint share link (or any https URL) to the WSP model
    # library bundle (a .zip). Unset means the link in wsp/config.json.
    model_library_url: str | None = None

    # Where prebuilt analysis environments are downloaded from. Unset
    # means RUNTIME_RELEASE_URL (this repository's "runtime" release).
    env_pack_url: str | None = None

    # Model catalog sync
    disable_model_updates: bool = Field(
        default=False,
        description="Disable automatic model catalog sync on startup"
    )

    # PyTorch wheel index mirror (mainland China). pip has no index
    # priority, so a mirror added through pip.ini competes with the
    # download.pytorch.org entry baked into the env YAMLs and can lose.
    # Only we can take that entry out, so this replaces it.
    pytorch_index_url: str | None = Field(
        default=None,
        description=(
            "Base URL of a PyTorch wheel index mirror, without the CUDA "
            "suffix, e.g. https://mirror.nju.edu.cn/pytorch/whl"
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _drop_blank_values(cls, data: dict) -> dict:
        """
        Treat blank strings as unset. An env var that is defined but
        empty (a half-filled GPO entry, `WSP_USER_DATA_DIR= app` in
        a shell script) would otherwise resolve paths against the
        working directory. Electron treats blank as unset too, so this
        keeps the two processes in agreement.
        """
        if isinstance(data, dict):
            return {
                k: v
                for k, v in data.items()
                if not (isinstance(v, str) and not v.strip())
            }
        return data

    @model_validator(mode="after")
    def _derive_from_user_data_dir(self) -> "Settings":
        """
        Derive database_url and models_dir from user_data_dir unless they
        were set explicitly (env var, .env file, or constructor argument).
        This is what makes WSP_USER_DATA_DIR alone relocate the
        whole app; without it the two defaults resolve to ~/WSP-CameraTrap
        regardless of the override.
        """
        # A relative path would silently resolve against the working
        # directory, which differs between Electron and the backend, so
        # the two processes would stop agreeing on where markers live.
        # Electron ignores non-absolute values; here we crash early for
        # anyone running the backend or CLI directly.
        if not self.user_data_dir.is_absolute():
            raise ValueError(
                f"WSP_USER_DATA_DIR must be an absolute path, "
                f"got {str(self.user_data_dir)!r}"
            )
        if "database_url" not in self.model_fields_set:
            self.database_url = f"sqlite:///{self.user_data_dir / 'wsp-cameratrap.db'}"
        if "models_dir" not in self.model_fields_set:
            self.models_dir = self.user_data_dir / "models"
        return self

    def __init__(self, **kwargs: object) -> None:
        """
        Initialize settings and validate critical paths exist or can be created.

        Crashes immediately if required directories cannot be set up.
        """
        super().__init__(**kwargs)

        # Ensure user data directory exists
        if not self.user_data_dir.exists():
            try:
                self.user_data_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to create user data directory at {self.user_data_dir}: {e}"
                ) from e

        # Ensure models directory exists
        if not self.models_dir.exists():
            try:
                self.models_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to create models directory at {self.models_dir}: {e}"
                ) from e


def get_settings() -> Settings:
    """
    Get application settings.

    Will crash if required environment variables are not set.
    This is intentional - we want to fail fast in development.
    """
    return Settings()
