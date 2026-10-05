"""Environment variable names used by application settings and persistence."""

ENVIRONMENT_FIELDS = {
    "base_url": "BASE_URL",
    "api_key": "API_KEY",
    "default_model": "DEFAULT_MODEL",
    "context_length": "CONTEXT_LENGTH",
    "api_type": "API_TYPE",
    "reasoning_effort": "REASONING_EFFORT",
    "coding_agent_persistent_history": "CODING_AGENT_PERSISTENT_HISTORY",
    "host": "EXMEKH_AGENT_HOST",
    "port": "EXMEKH_AGENT_PORT",
    "app_title": "EXMEKH_AGENT_TITLE",
}

PERSISTED_FIELDS = tuple(
    name for name in ENVIRONMENT_FIELDS if name != "context_length"
)
