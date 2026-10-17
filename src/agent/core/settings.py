from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from functools import lru_cache
from typing import Optional

class AgentSettings(BaseSettings):
    model:str = "ollama:llama3.1"
    fallback_model: Optional[str] = None
    max_tool_call_iterations:int = 10

    # Per LLM call, and how many times the provider SDK retries it
    # (SDK retries cover 429 / 5xx / connection errors with backoff).
    llm_timeout_seconds: float = 45.0
    llm_max_retries: int = 2

    # Hard ceiling for one whole /chat request (all LLM + tool steps).
    request_timeout_seconds: float = 120.0

    model_config = SettingsConfigDict(
        env_file=".env", 
        env_prefix="AGENT_", 
        case_sensitive=False,
        extra="ignore"
    )

class MCPSettings(BaseSettings):
    server_url: str = "http://127.0.0.1:8002/mcp"

    model_config = SettingsConfigDict(
        env_file=".env", 
        env_prefix="MCP_", 
        case_sensitive=False,
        extra="ignore"
    )

class LangFuseSettings(BaseSettings):
    public_key: Optional[str] = None
    secret_key: Optional[str] = None
    base_url:str = "https://us.cloud.langfuse.com"

    model_config = SettingsConfigDict(
        env_file=".env", 
        env_prefix="LANGFUSE_", 
        case_sensitive=False,
        extra="ignore"
    )

class PostgresSettings(BaseSettings):
    checkpoint_url:Optional[str] = None

    model_config = SettingsConfigDict(
        env_file=".env", 
        env_prefix="POSTGRES_", 
        case_sensitive=False,
        extra="ignore"
    )

class SecuritySettings(BaseSettings):
    api_key: Optional[str] = None
    rate_limit_per_minute: int = 20

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="SECURITY_",
        case_sensitive=False,
        extra="ignore"
    )
    
class Settings(BaseSettings):
    ENV: str = "development"
    PORT: int = 8003

    agent: AgentSettings = Field(default_factory=AgentSettings)
    mcp: MCPSettings = Field(default_factory=MCPSettings)
    langfuse: LangFuseSettings = Field(default_factory=LangFuseSettings)
    postgres: PostgresSettings = Field(default_factory=PostgresSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

@lru_cache
def get_settings() -> Settings:
    """
    Get cached settings instance
    
    - Settings only loaded once (performance)
    - Same instance reused everywhere (consistency)
    - Can be easily mocked in tests
    """
    return Settings()

settings = get_settings()