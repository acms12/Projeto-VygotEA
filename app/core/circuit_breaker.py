"""
Circuit Breaker Pattern para robustez e resiliência
"""
import time
import asyncio
from enum import Enum
from typing import Callable, Any, Optional
import structlog
from functools import wraps

logger = structlog.get_logger(__name__)

class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class CircuitBreaker:
    """
    Implementação do padrão Circuit Breaker para proteção contra falhas
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        timeout: int = 60,
        recovery_timeout: int = 300,
        name: str = "default"
    ):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.recovery_timeout = recovery_timeout
        self.name = name
        
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = None
        self.success_count = 0
        
    def __call__(self, func: Callable) -> Callable:
        """Decorator para aplicar circuit breaker a uma função"""
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            return await self._execute_async(func, *args, **kwargs)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            return self._execute_sync(func, *args, **kwargs)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    def _execute_sync(self, func: Callable, *args, **kwargs) -> Any:
        """Executa função síncrona com circuit breaker"""
        if not self._can_execute():
            raise CircuitBreakerOpenError(f"Circuit breaker '{self.name}' is open")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
    
    async def _execute_async(self, func: Callable, *args, **kwargs) -> Any:
        """Executa função assíncrona com circuit breaker"""
        if not self._can_execute():
            raise CircuitBreakerOpenError(f"Circuit breaker '{self.name}' is open")
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
    
    def _can_execute(self) -> bool:
        """Verifica se a operação pode ser executada"""
        if self.state == CircuitState.CLOSED:
            return True
        
        if self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
                logger.info(f"Circuit breaker '{self.name}' transitioning to half-open")
                return True
            return False
        
        if self.state == CircuitState.HALF_OPEN:
            return True
        
        return False
    
    def _on_success(self):
        """Chamado quando uma operação é bem-sucedida"""
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.success_count = 0
            logger.info(f"Circuit breaker '{self.name}' closed after successful operation")
    
    def _on_failure(self):
        """Chamado quando uma operação falha"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.state == CircuitState.CLOSED and self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            logger.warning(f"Circuit breaker '{self.name}' opened after {self.failure_count} failures")
        
        elif self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
            logger.warning(f"Circuit breaker '{self.name}' reopened after failure in half-open state")

class CircuitBreakerOpenError(Exception):
    """Exceção lançada quando o circuit breaker está aberto"""
    pass

# Circuit breakers globais
ml_model_circuit_breaker = CircuitBreaker(
    failure_threshold=3,
    timeout=30,
    recovery_timeout=60,
    name="ml_model"
)

database_circuit_breaker = CircuitBreaker(
    failure_threshold=5,
    timeout=60,
    recovery_timeout=300,
    name="database"
)

external_api_circuit_breaker = CircuitBreaker(
    failure_threshold=3,
    timeout=30,
    recovery_timeout=120,
    name="external_api"
)