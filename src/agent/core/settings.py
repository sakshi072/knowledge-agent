from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from functools import lru_cache
from typing import Optional

class AgentSettings(BaseSettings):
    model:str = "ollama:llama3.1"
    max_tool_call_iterations:int = 10

    model_config = SettingsConfigDict(
        env_file=".env", 
        env_prefix="AGENT_", 
        case_sensitive=False,
        extra="ignore"
    )

class MCPSettings(BaseSettings):
    MCP_SERVER_URL: str = "http://localhost:8002/mcp"

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
    
class Settings(BaseSettings):
    ENV: str = "development"
    PORT: int = 8003

    agent: AgentSettings = Field(default_factory=AgentSettings)
    mcp: MCPSettings = Field(default_factory=MCPSettings)
    langfuse: LangFuseSettings = Field(default_factory=LangFuseSettings)
    postgres: PostgresSettings = Field(default_factory=PostgresSettings)

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