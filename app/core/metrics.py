"""
Sistema de métricas com Prometheus para observabilidade
"""
from prometheus_client import Counter, Histogram, Gauge, Summary
import time
from functools import wraps
import structlog

logger = structlog.get_logger(__name__)

# Métricas de API
api_requests_total = Counter(
    'api_requests_total',
    'Total number of API requests',
    ['method', 'endpoint', 'status']
)

api_request_duration = Histogram(
    'api_request_duration_seconds',
    'API request duration in seconds',
    ['method', 'endpoint']
)

# Métricas de ML
ml_predictions_total = Counter(
    'ml_predictions_total',
    'Total number of ML predictions',
    ['model_type', 'domain']
)

ml_prediction_accuracy = Gauge(
    'ml_prediction_accuracy',
    'ML model prediction accuracy',
    ['model_type', 'domain']
)

ml_model_confidence = Histogram(
    'ml_model_confidence',
    'ML model confidence scores',
    ['model_type', 'domain']
)

# Métricas de ZDP
zdp_assessments_total = Counter(
    'zdp_assessments_total',
    'Total number of ZDP assessments',
    ['user_id', 'domain']
)

zdp_level_changes = Counter(
    'zdp_level_changes_total',
    'Total number of ZDP level changes',
    ['user_id', 'domain', 'direction']
)

# Métricas de Gamificação
gamification_events_total = Counter(
    'gamification_events_total',
    'Total number of gamification events',
    ['event_type', 'user_id']
)

user_xp_total = Gauge(
    'user_xp_total',
    'Total XP for users',
    ['user_id']
)

# Métricas de RAG
rag_queries_total = Counter(
    'rag_queries_total',
    'Total number of RAG queries',
    ['query_type']
)

rag_response_time = Histogram(
    'rag_response_time_seconds',
    'RAG response time in seconds',
    ['query_type']
)

# Métricas de Circuit Breaker
circuit_breaker_state = Gauge(
    'circuit_breaker_state',
    'Circuit breaker state (0=closed, 1=half-open, 2=open)',
    ['breaker_name']
)

circuit_breaker_failures = Counter(
    'circuit_breaker_failures_total',
    'Total circuit breaker failures',
    ['breaker_name']
)

# Métricas de Cache
cache_hits = Counter(
    'cache_hits_total',
    'Total cache hits',
    ['cache_type']
)

cache_misses = Counter(
    'cache_misses_total',
    'Total cache misses',
    ['cache_type']
)

def track_api_metrics(func):
    """Decorator para rastrear métricas de API"""
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        start_time = time.time()
        method = "GET"  # Default, pode ser melhorado
        endpoint = func.__name__
        
        try:
            result = await func(*args, **kwargs)
            status = "success"
            api_requests_total.labels(method=method, endpoint=endpoint, status=status).inc()
            return result
        except Exception as e:
            status = "error"
            api_requests_total.labels(method=method, endpoint=endpoint, status=status).inc()
            raise
        finally:
            duration = time.time() - start_time
            api_request_duration.labels(method=method, endpoint=endpoint).observe(duration)
    
    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        start_time = time.time()
        method = "GET"  # Default, pode ser melhorado
        endpoint = func.__name__
        
        try:
            result = func(*args, **kwargs)
            status = "success"
            api_requests_total.labels(method=method, endpoint=endpoint, status=status).inc()
            return result
        except Exception as e:
            status = "error"
            api_requests_total.labels(method=method, endpoint=endpoint, status=status).inc()
            raise
        finally:
            duration = time.time() - start_time
            api_request_duration.labels(method=method, endpoint=endpoint).observe(duration)
    
    import asyncio
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    return sync_wrapper

def track_ml_metrics(model_type: str, domain: str):
    """Decorator para rastrear métricas de ML"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                ml_predictions_total.labels(model_type=model_type, domain=domain).inc()
                
                # Se o resultado incluir confiança, registra
                if hasattr(result, 'confidence'):
                    ml_model_confidence.labels(model_type=model_type, domain=domain).observe(result.confidence)
                
                return result
            except Exception as e:
                logger.error(f"ML prediction failed for {model_type}/{domain}: {e}")
                raise
            finally:
                duration = time.time() - start_time
                # Pode adicionar métrica de duração se necessário
        
        return wrapper
    return decorator

def track_zdp_metrics(user_id: str, domain: str):
    """Decorator para rastrear métricas de ZDP"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
                zdp_assessments_total.labels(user_id=user_id, domain=domain).inc()
                return result
            except Exception as e:
                logger.error(f"ZDP assessment failed for user {user_id}, domain {domain}: {e}")
                raise
        
        return wrapper
    return decorator

def track_gamification_metrics(event_type: str, user_id: str):
    """Decorator para rastrear métricas de gamificação"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
                gamification_events_total.labels(event_type=event_type, user_id=user_id).inc()
                return result
            except Exception as e:
                logger.error(f"Gamification event failed for user {user_id}, event {event_type}: {e}")
                raise
        
        return wrapper
    return decorator

def update_circuit_breaker_metrics(breaker_name: str, state: str):
    """Atualiza métricas do circuit breaker"""
    state_map = {"closed": 0, "half_open": 1, "open": 2}
    circuit_breaker_state.labels(breaker_name=breaker_name).set(state_map.get(state, 0))

def record_circuit_breaker_failure(breaker_name: str):
    """Registra falha do circuit breaker"""
    circuit_breaker_failures.labels(breaker_name=breaker_name).inc()

def record_cache_hit(cache_type: str):
    """Registra hit no cache"""
    cache_hits.labels(cache_type=cache_type).inc()

def record_cache_miss(cache_type: str):
    """Registra miss no cache"""
    cache_misses.labels(cache_type=cache_type).inc()

def update_user_xp(user_id: str, xp: int):
    """Atualiza XP do usuário nas métricas"""
    user_xp_total.labels(user_id=user_id).set(xp)