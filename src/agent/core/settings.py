from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from functools import lru_cache

class AgentSettings(BaseSettings):
    agent_model:str = "ollama:llama3.1"

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

class Settings(BaseSettings):
    ENV: str = "development"
    PORT: int = 8003

    agent: AgentSettings = Field(default_factory=AgentSettings)
    mcp: MCPSettings = Field(default_factory=MCPSettings)

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