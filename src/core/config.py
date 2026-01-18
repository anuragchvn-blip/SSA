"""Configuration management using Pydantic Settings."""

import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import PostgresDsn, RedisDsn, validator


class DatabaseSettings(BaseSettings):
    """Database configuration settings."""
    
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "ssa_engine"
    postgres_user: str = "ssa_user"
    postgres_password: str = ""
    
    @property
    def sqlalchemy_database_uri(self) -> str:
        """Construct SQLAlchemy database URI."""
        if not self.postgres_password or self.postgres_password == "change_me_in_production":
            # Fallback to SQLite for demo mode
            return "sqlite:///./ssa_demo.db"
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "allow"


class RedisSettings(BaseSettings):
    """Redis configuration settings."""
    
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: Optional[str] = None
    redis_database: int = 0
    
    @property
    def redis_url(self) -> str:
        """Construct Redis connection URL."""
        if self.redis_password:
            return f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}/{self.redis_database}"
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_database}"
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "allow"


class SpaceTrackSettings(BaseSettings):
    """Space-Track API configuration."""
    
    # Space-Track API Configuration
    spacetrack_username: str = ""
    spacetrack_password: str = ""
    spacetrack_rate_limit: int = 300  # Requests per hour (free tier limit)
    spacetrack_base_url: str = "https://www.space-track.org"
    
    # TLE Update Schedule
    tle_update_interval_hours: int = 24  # Daily updates
    tle_staleness_threshold_hours: int = 72  # Flag TLEs >3 days old
    
    @classmethod
    def credentials_required(cls, v: str, info) -> str:
        """Validate that Space-Track credentials are provided."""
        if not v:
            raise ValueError(f"{info.field_name} is required for Space-Track API access")
        return v
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "allow"


class APISettings(BaseSettings):
    """API configuration settings."""
    
    jwt_secret_key: str = ""
    jwt_algorithm: str = "HS256"
    jwt_expiration_hours: int = 24
    api_rate_limit_requests_per_minute: int = 60
    api_max_concurrent_requests: int = 10
    
    @classmethod
    def secret_key_required(cls, v: str, info) -> str:
        """Validate that JWT secret key is provided."""
        if not v or v == "change_me_in_production":
            raise ValueError("JWT_SECRET_KEY must be set to a secure value")
        return v
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "allow"


class LoggingSettings(BaseSettings):
    """Logging configuration settings."""
    
    log_level: str = "INFO"
    log_format: str = "json"  # json or console
    sentry_dsn: Optional[str] = None
    
    @classmethod
    def validate_log_level(cls, v: str, info) -> str:
        """Validate log level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Invalid log level: {v}")
        return v.upper()
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "allow"


class PathSettings(BaseSettings):
    """Path configuration settings."""
    
    data_storage_path: str = "./data"
    model_version_path: str = "./models"
    orekit_data_path: str = "./orekit-data"
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "allow"


class Settings(BaseSettings):
    """Main application settings combining all configurations."""
    
    # Sub-configurations
    database: DatabaseSettings = DatabaseSettings()
    redis: RedisSettings = RedisSettings()
    spacetrack: SpaceTrackSettings = SpaceTrackSettings()
    api: APISettings = APISettings()
    logging: LoggingSettings = LoggingSettings()
    paths: PathSettings = PathSettings()
    
    # Application metadata
    app_name: str = "SSA Conjunction Analysis Engine"
    app_version: str = "1.0.0"
    debug: bool = False
    
    model_config = {
        "env_file": ".env",
        "case_sensitive": False,
        "extra": "allow",
        "protected_namespaces": ("settings_", "model_")
    }


# Global settings instance
settings = Settings()