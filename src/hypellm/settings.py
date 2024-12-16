from importlib import import_module
from typing import Optional, TYPE_CHECKING

from pydantic import Field, HttpUrl
from pydantic_settings import BaseSettings, SettingsConfigDict

if TYPE_CHECKING:
    from hypellm.impl import Impl


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="HYPELLM_", env_file=[".env"])

    model: str
    api_key: str
    api_version: Optional[str] = None
    base_url: Optional[HttpUrl] = None
    batch_size: int = Field(default=5, ge=1)
    concurrency: int = Field(default=10, ge=1)
    show_progress: bool = True

    impl_: str = Field(default="instructor", alias="impl")

    @property
    def impl(self) -> "Impl":
        return import_module(f"hypellm.impl.{self.impl_}")


settings = Settings()
