"""
Aplicação principal Vygotea com integração completa de todos os sistemas
"""
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
import structlog
import time
from contextlib import asynccontextmanager

from app.core.config import settings
from app.core.metrics import api_requests_total, api_request_duration
from app.api.routes import router
from prometheus_client import make_asgi_app

# Configurar logging
logger = structlog.get_logger(__name__)

# Middleware para métricas
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gerenciador de ciclo de vida da aplicação"""
    logger.info("Starting Vygotea application...")
    yield
    logger.info("Shutting down Vygotea application...")

# Criar aplicação FastAPI
app = FastAPI(
    title="Vygotea - Sistema de Tutoria Inteligente",
    description="""
    Sistema avançado de tutoria baseado em Zona de Desenvolvimento Proximal (ZDP) 
    com gamificação, aprendizado de máquina e geração de respostas inteligentes.
    
    ## Características Principais
    
    * **Análise ZDP Avançada**: Modelos adaptativos para cálculo de níveis e zonas proximais
    * **Motor de Gamificação**: Sistema de XP, badges e streaks dinâmicos
    * **Base de Conhecimento RAG**: Sistema robusto de recuperação e geração de conhecimento
    * **Modelos ML Adaptativos**: Treinamento contínuo com feedback do usuário
    * **Geração de Respostas Inteligente**: Integração com LLMs para respostas contextuais
    * **Monitoramento e Observabilidade**: Métricas, logs estruturados e circuit breakers
    
    ## Endpoints Principais
    
    * `/api/v1/chat` - Chat integrado com todos os sistemas
    * `/api/v1/zdp/assess` - Avaliação de ZDP
    * `/api/v1/gamification/event` - Processamento de eventos de gamificação
    * `/api/v1/rag/query` - Consulta à base de conhecimento
    * `/api/v1/intervention/generate` - Geração de intervenções personalizadas
    * `/api/v1/ml/predict-difficulty` - Predição de dificuldade
    """,
    version=settings.version,
    docs_url=None,
    redoc_url=None,
    lifespan=lifespan
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Em produção, especificar origens específicas
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Middleware para métricas e logging
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """Middleware para coletar métricas e logs"""
    start_time = time.time()
    
    # Log da requisição
    logger.info(
        "Incoming request",
        method=request.method,
        url=str(request.url),
        client_ip=request.client.host if request.client else None
    )
    
    try:
        response = await call_next(request)
        
        # Registrar métricas
        duration = time.time() - start_time
        api_requests_total.labels(
            method=request.method,
            endpoint=request.url.path,
            status=response.status_code
        ).inc()
        
        api_request_duration.labels(
            method=request.method,
            endpoint=request.url.path
        ).observe(duration)
        
        # Log da resposta
        logger.info(
            "Request completed",
            method=request.method,
            url=str(request.url),
            status_code=response.status_code,
            duration=duration
        )
        
        return response
        
    except Exception as e:
        # Registrar erro
        duration = time.time() - start_time
        api_requests_total.labels(
            method=request.method,
            endpoint=request.url.path,
            status=500
        ).inc()
        
        logger.error(
            "Request failed",
            method=request.method,
            url=str(request.url),
            error=str(e),
            duration=duration
        )
        
        raise

# Middleware para tratamento de erros
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handler global para exceções"""
    logger.error(
        "Unhandled exception",
        method=request.method,
        url=str(request.url),
        error=str(exc),
        error_type=type(exc).__name__
    )
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "Ocorreu um erro interno no servidor",
            "timestamp": time.time()
        }
    )

# Incluir rotas da API
app.include_router(router)

# Endpoint para métricas Prometheus
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Endpoint raiz
@app.get("/")
async def root():
    """Endpoint raiz com informações do sistema"""
    return {
        "name": "Vygotea",
        "version": settings.version,
        "description": "Sistema de Tutoria Inteligente",
        "status": "operational",
        "documentation": "/docs",
        "metrics": "/metrics",
        "health": "/api/v1/system/health"
    }

# Endpoint de saúde
@app.get("/health")
async def health_check():
    """Verificação de saúde básica"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": settings.version
    }

# Endpoint de documentação customizada
@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    """Documentação Swagger customizada"""
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=f"{app.title} - Documentação",
        swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5.9.0/swagger-ui-bundle.js",
        swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5.9.0/swagger-ui.css",
    )

# Configuração customizada do OpenAPI
def custom_openapi():
    """Configuração customizada do OpenAPI"""
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    
    # Adicionar informações de contato
    openapi_schema["info"]["contact"] = {
        "name": "Equipe Vygotea",
        "email": "contato@vygotea.com"
    }
    
    # Adicionar tags para organização
    openapi_schema["tags"] = [
        {
            "name": "ZDP",
            "description": "Operações relacionadas à Zona de Desenvolvimento Proximal"
        },
        {
            "name": "Gamificação",
            "description": "Sistema de gamificação e recompensas"
        },
        {
            "name": "RAG",
            "description": "Sistema de Recuperação e Geração de Conhecimento"
        },
        {
            "name": "Intervenção",
            "description": "Geração de intervenções personalizadas"
        },
        {
            "name": "ML",
            "description": "Sistema de Machine Learning"
        },
        {
            "name": "Sistema",
            "description": "Operações gerais do sistema"
        }
    ]
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

# Endpoint para informações do sistema
@app.get("/api/v1/system/info")
async def system_info():
    """Informações detalhadas do sistema"""
    return {
        "name": "Vygotea",
        "version": settings.version,
        "environment": settings.environment.value,
        "description": "Sistema de Tutoria Inteligente",
        "features": {
            "zdp_analysis": True,
            "gamification": True,
            "rag_system": True,
            "ml_models": True,
            "intervention_engine": True,
            "circuit_breakers": True,
            "metrics": True,
            "logging": True
        },
        "config": {
            "ml_model_type": settings.ml_model_type.value,
            "database_type": settings.database_type.value,
            "enable_metrics": settings.enable_metrics,
            "log_level": settings.log_level
        },
        "endpoints": {
            "api_base": "/api/v1",
            "docs": "/docs",
            "metrics": "/metrics",
            "health": "/health"
        }
    }

# Endpoint para status dos serviços
@app.get("/api/v1/system/services")
async def services_status():
    """Status dos serviços do sistema"""
    return {
        "services": {
            "zdp_service": {
                "status": "operational",
                "description": "Serviço de análise ZDP"
            },
            "gamification_service": {
                "status": "operational",
                "description": "Serviço de gamificação"
            },
            "rag_service": {
                "status": "operational",
                "description": "Serviço de base de conhecimento"
            },
            "intervention_service": {
                "status": "operational",
                "description": "Serviço de intervenção"
            },
            "ml_service": {
                "status": "operational",
                "description": "Serviço de machine learning"
            }
        },
        "overall_status": "operational",
        "timestamp": time.time()
    }

if __name__ == "__main__":
    import uvicorn
    
    logger.info(f"Starting Vygotea server on {settings.host}:{settings.port}")
    
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )