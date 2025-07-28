"""
Configurações avançadas do sistema Vygotea
"""
import os
from typing import Optional, Dict, Any
from pydantic_settings import BaseSettings
from pydantic import Field
from enum import Enum
import structlog

class Environment(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"

class MLModelType(str, Enum):
    TRANSFORMER = "transformer"
    TRADITIONAL = "traditional"
    HYBRID = "hybrid"

class DatabaseType(str, Enum):
    SQLITE = "sqlite"
    POSTGRESQL = "postgresql"
    MONGODB = "mongodb"
    ELASTICSEARCH = "elasticsearch"

class Settings(BaseSettings):
    """Configurações do sistema Vygotea"""
    
    # Configurações da API
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    debug: bool = Field(default=False, env="DEBUG")
    environment: Environment = Field(default=Environment.DEVELOPMENT, env="ENVIRONMENT")
    version: str = Field(default="2.0.0", env="VERSION")
    
    # Configurações de Segurança
    secret_key: str = Field(default="your-secret-key-here", env="SECRET_KEY")
    algorithm: str = Field(default="HS256", env="ALGORITHM")
    access_token_expire_minutes: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    
    # Configurações de Log
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(default="json", env="LOG_FORMAT")
    
    # Configurações de Banco de Dados
    database_type: DatabaseType = Field(default=DatabaseType.SQLITE, env="DATABASE_TYPE")
    database_url: str = Field(default="sqlite:///./vygotea.db", env="DATABASE_URL")
    postgres_host: str = Field(default="localhost", env="POSTGRES_HOST")
    postgres_port: str = Field(default="5432", env="POSTGRES_PORT")
    postgres_db: str = Field(default="vygotea", env="POSTGRES_DB")
    postgres_user: str = Field(default="vygotea_user", env="POSTGRES_USER")
    postgres_password: str = Field(default="vygotea_password", env="POSTGRES_PASSWORD")
    mongodb_url: str = Field(default="mongodb://localhost:27017/vygotea", env="MONGODB_URL")
    elasticsearch_url: str = Field(default="http://localhost:9200", env="ELASTICSEARCH_URL")
    
    # Configurações de Cache
    redis_url: str = Field(default="redis://localhost:6379/0", env="REDIS_URL")
    cache_ttl: int = Field(default=3600, env="CACHE_TTL")
    
    # Configurações de ML/AI
    ml_model_type: MLModelType = Field(default=MLModelType.TRADITIONAL, env="ML_MODEL_TYPE")
    deep_learning_available: bool = Field(default=False, env="DEEP_LEARNING_AVAILABLE")
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    model_path: str = Field(default="./models", env="MODEL_PATH")
    training_data_path: str = Field(default="./data/training", env="TRAINING_DATA_PATH")
    
    # Configurações RAG
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2", env="EMBEDDING_MODEL")
    vector_dimension: int = Field(default=384, env="VECTOR_DIMENSION")
    similarity_threshold: float = Field(default=0.7, env="SIMILARITY_THRESHOLD")
    max_results: int = Field(default=10, env="MAX_RESULTS")
    
    # Configurações de Gamificação
    xp_multiplier: float = Field(default=1.0, env="XP_MULTIPLIER")
    streak_bonus: float = Field(default=0.1, env="STREAK_BONUS")
    difficulty_bonus: float = Field(default=0.2, env="DIFFICULTY_BONUS")
    badge_xp_reward: int = Field(default=100, env="BADGE_XP_REWARD")
    level_up_xp: int = Field(default=1000, env="LEVEL_UP_XP")
    
    # Configurações ZDP
    zdp_confidence_threshold: float = Field(default=0.7, env="ZDP_CONFIDENCE_THRESHOLD")
    zdp_zone_width: float = Field(default=0.2, env="ZDP_ZONE_WIDTH")
    zdp_adaptive_factor: float = Field(default=0.1, env="ZDP_ADAPTIVE_FACTOR")
    zdp_history_weight: float = Field(default=0.3, env="ZDP_HISTORY_WEIGHT")
    
    # Configurações de Intervenção
    intervention_urgency_threshold: float = Field(default=0.8, env="INTERVENTION_URGENCY_THRESHOLD")
    intervention_personalization_level: float = Field(default=0.7, env="INTERVENTION_PERSONALIZATION_LEVEL")
    intervention_context_window: int = Field(default=5, env="INTERVENTION_CONTEXT_WINDOW")
    
    # Configurações de Circuit Breaker
    circuit_breaker_failure_threshold: int = Field(default=5, env="CIRCUIT_BREAKER_FAILURE_THRESHOLD")
    circuit_breaker_recovery_timeout: int = Field(default=60, env="CIRCUIT_BREAKER_RECOVERY_TIMEOUT")
    circuit_breaker_half_open_max_calls: int = Field(default=3, env="CIRCUIT_BREAKER_HALF_OPEN_MAX_CALLS")
    
    # Configurações de Métricas
    enable_metrics: bool = Field(default=True, env="ENABLE_METRICS")
    metrics_port: int = Field(default=9090, env="METRICS_PORT")
    
    # Configurações de Rate Limiting
    rate_limit_requests: int = Field(default=100, env="RATE_LIMIT_REQUESTS")
    rate_limit_window: int = Field(default=60, env="RATE_LIMIT_WINDOW")
    
    # Configurações de Filas
    celery_broker_url: str = Field(default="redis://localhost:6379/1", env="CELERY_BROKER_URL")
    celery_result_backend: str = Field(default="redis://localhost:6379/2", env="CELERY_RESULT_BACKEND")
    
    # Configurações de Monitoramento
    prometheus_enabled: bool = Field(default=True, env="PROMETHEUS_ENABLED")
    health_check_interval: int = Field(default=30, env="HEALTH_CHECK_INTERVAL")
    
    # Configurações de Desenvolvimento
    reload: bool = Field(default=True, env="RELOAD")
    workers: int = Field(default=1, env="WORKERS")
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "allow"

# Configuração de logging estruturado
def setup_logging():
    """Configura logging estruturado com structlog"""
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

# Instância global das configurações
settings = Settings()

# Setup inicial
setup_logging()
logger = structlog.get_logger(__name__)