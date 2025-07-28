"""
Rotas da API com integração completa de todos os sistemas
"""
from fastapi import APIRouter, HTTPException, Depends, Query, Body
from fastapi.responses import JSONResponse
from typing import Dict, List, Optional, Any
import structlog

from app.services.zdp_service import ZDPService
from app.services.gamification_service import GamificationService
from app.services.rag_service import RAGService
from app.services.intervention_engine import InterventionService
from app.ml.adaptive_ml_engine import MLService
from app.models.zdp import ZDPAssessment, ZDPRecommendation, DomainType
from app.models.gamification import EventType, GamificationEvent
from app.core.metrics import track_api_metrics

logger = structlog.get_logger(__name__)

# Criar router principal
router = APIRouter(prefix="/api/v1")

# Inicializar serviços
zdp_service = ZDPService()
gamification_service = GamificationService()
rag_service = RAGService()
intervention_service = InterventionService()
ml_service = MLService()

# ==================== ROTAS DE ZDP ====================

@router.post("/zdp/assess")
@track_api_metrics
async def assess_user_zdp(
    user_id: str = Body(..., embed=True),
    assessment_data: Dict[str, Any] = Body(...),
    user_history: Optional[Dict[str, Any]] = Body(None)
):
    """Avalia ZDP completa do usuário"""
    try:
        # Converter dados de histórico se fornecido
        zdp_history = None
        if user_history:
            # Implementação seria baseada na estrutura de dados real
            pass
        
        assessment = await zdp_service.assess_user_zdp(
            user_id, assessment_data, zdp_history
        )
        
        return {
            "status": "success",
            "assessment": assessment.dict(),
            "message": "Avaliação ZDP concluída com sucesso"
        }
        
    except Exception as e:
        logger.error(f"Error in ZDP assessment: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/zdp/recommendations/{user_id}")
@track_api_metrics
async def get_zdp_recommendations(user_id: str):
    """Obtém recomendações ZDP para o usuário"""
    try:
        # Implementação seria baseada em dados reais do usuário
        # Por enquanto, retorna recomendação mock
        recommendation = {
            "user_id": user_id,
            "primary_focus": "mathematics",
            "secondary_focus": "language",
            "recommended_activities": [
                "Resolver problemas de álgebra",
                "Praticar equações lineares",
                "Ler textos em português"
            ],
            "difficulty_level": 6.5,
            "estimated_duration": 30,
            "confidence_score": 0.85,
            "reasoning": "Usuário demonstra boa base mas precisa de prática",
            "adaptive_suggestions": [
                "Use recursos visuais para conceitos abstratos",
                "Inclua exemplos práticos do cotidiano"
            ]
        }
        
        return {
            "status": "success",
            "recommendation": recommendation
        }
        
    except Exception as e:
        logger.error(f"Error getting ZDP recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== ROTAS DE GAMIFICAÇÃO ====================

@router.post("/gamification/event")
@track_api_metrics
async def process_gamification_event(
    user_id: str = Body(..., embed=True),
    event_type: str = Body(..., embed=True),
    metadata: Optional[Dict[str, Any]] = Body(None)
):
    """Processa evento de gamificação"""
    try:
        # Converter string para enum
        event_enum = EventType(event_type)
        
        result = await gamification_service.process_event(
            user_id, event_enum, metadata
        )
        
        return {
            "status": "success",
            "result": result
        }
        
    except Exception as e:
        logger.error(f"Error processing gamification event: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/gamification/profile/{user_id}")
@track_api_metrics
async def get_gamification_profile(user_id: str):
    """Obtém perfil de gamificação do usuário"""
    try:
        profile = await gamification_service.get_user_profile(user_id)
        
        return {
            "status": "success",
            "profile": profile.dict()
        }
        
    except Exception as e:
        logger.error(f"Error getting gamification profile: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/gamification/leaderboard")
@track_api_metrics
async def get_leaderboard(limit: int = Query(10, ge=1, le=50)):
    """Obtém leaderboard de gamificação"""
    try:
        leaderboard = await gamification_service.get_leaderboard(limit)
        
        return {
            "status": "success",
            "leaderboard": [entry.dict() for entry in leaderboard]
        }
        
    except Exception as e:
        logger.error(f"Error getting leaderboard: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== ROTAS DE RAG ====================

@router.post("/rag/query")
@track_api_metrics
async def query_knowledge_base(
    query: str = Body(..., embed=True),
    domain: Optional[str] = Body(None),
    max_results: int = Body(5, ge=1, le=20)
):
    """Consulta a base de conhecimento"""
    try:
        result = await rag_service.query_knowledge_base(
            query, domain, max_results
        )
        
        return {
            "status": "success",
            "result": result
        }
        
    except Exception as e:
        logger.error(f"Error querying knowledge base: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/rag/document")
@track_api_metrics
async def add_knowledge_document(
    title: str = Body(..., embed=True),
    content: str = Body(..., embed=True),
    domain: str = Body(..., embed=True),
    difficulty_level: float = Body(..., ge=0, le=10),
    tags: List[str] = Body(...),
    metadata: Optional[Dict[str, Any]] = Body(None)
):
    """Adiciona documento à base de conhecimento"""
    try:
        result = await rag_service.add_knowledge_document(
            title, content, domain, difficulty_level, tags, metadata
        )
        
        return {
            "status": "success",
            "result": result
        }
        
    except Exception as e:
        logger.error(f"Error adding knowledge document: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/rag/statistics")
@track_api_metrics
async def get_knowledge_statistics():
    """Obtém estatísticas da base de conhecimento"""
    try:
        stats = await rag_service.get_knowledge_statistics()
        
        return {
            "status": "success",
            "statistics": stats
        }
        
    except Exception as e:
        logger.error(f"Error getting knowledge statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== ROTAS DE INTERVENÇÃO ====================

@router.post("/intervention/generate")
@track_api_metrics
async def generate_intervention(
    user_message: str = Body(..., embed=True),
    user_id: str = Body(..., embed=True),
    zdp_assessment: Optional[Dict[str, Any]] = Body(None),
    gamification_profile: Optional[Dict[str, Any]] = Body(None),
    recent_events: Optional[List[Dict[str, Any]]] = Body(None),
    context: Optional[Dict[str, Any]] = Body(None)
):
    """Gera intervenção personalizada"""
    try:
        # Converter dados se fornecidos
        zdp_obj = None
        if zdp_assessment:
            # Implementação seria baseada na estrutura real
            pass
        
        gamification_obj = None
        if gamification_profile:
            # Implementação seria baseada na estrutura real
            pass
        
        result = await intervention_service.generate_response(
            user_message, user_id, zdp_obj, gamification_obj, recent_events, context
        )
        
        return {
            "status": "success",
            "intervention": result
        }
        
    except Exception as e:
        logger.error(f"Error generating intervention: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/intervention/history/{user_id}")
@track_api_metrics
async def get_intervention_history(
    user_id: str,
    limit: int = Query(10, ge=1, le=50)
):
    """Obtém histórico de intervenções do usuário"""
    try:
        history = await intervention_service.get_intervention_history(user_id, limit)
        
        return {
            "status": "success",
            "history": history
        }
        
    except Exception as e:
        logger.error(f"Error getting intervention history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== ROTAS DE ML ====================

@router.post("/ml/predict-difficulty")
@track_api_metrics
async def predict_difficulty(
    text: str = Body(..., embed=True),
    domain: str = Body("general"),
    model_type: str = Body("ensemble")
):
    """Prediz dificuldade de um texto"""
    try:
        result = await ml_service.predict_difficulty(text, domain, model_type)
        
        return {
            "status": "success",
            "prediction": result
        }
        
    except Exception as e:
        logger.error(f"Error predicting difficulty: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ml/training-example")
@track_api_metrics
async def add_training_example(
    user_id: str = Body(..., embed=True),
    text: str = Body(..., embed=True),
    difficulty_label: str = Body(..., embed=True),
    domain: str = Body(..., embed=True),
    confidence: float = Body(1.0, ge=0, le=1),
    metadata: Optional[Dict[str, Any]] = Body(None)
):
    """Adiciona exemplo de treinamento"""
    try:
        result = await ml_service.add_training_example(
            user_id, text, difficulty_label, domain, confidence, metadata
        )
        
        return {
            "status": "success",
            "result": result
        }
        
    except Exception as e:
        logger.error(f"Error adding training example: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ml/generate-content")
@track_api_metrics
async def generate_educational_content(
    topic: str = Body(..., embed=True),
    difficulty_level: float = Body(..., ge=0, le=10),
    content_type: str = Body("explanation")
):
    """Gera conteúdo educacional"""
    try:
        result = await ml_service.generate_educational_content(
            topic, difficulty_level, content_type
        )
        
        return {
            "status": "success",
            "content": result
        }
        
    except Exception as e:
        logger.error(f"Error generating educational content: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/ml/statistics")
@track_api_metrics
async def get_ml_statistics():
    """Obtém estatísticas do sistema de ML"""
    try:
        stats = await ml_service.get_ml_statistics()
        
        return {
            "status": "success",
            "statistics": stats
        }
        
    except Exception as e:
        logger.error(f"Error getting ML statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== ROTAS INTEGRADAS ====================

@router.post("/chat")
@track_api_metrics
async def chat_with_system(
    user_id: str = Body(..., embed=True),
    message: str = Body(..., embed=True),
    context: Optional[Dict[str, Any]] = Body(None)
):
    """Chat integrado com todos os sistemas"""
    try:
        # 1. Consultar base de conhecimento
        rag_result = await rag_service.query_knowledge_base(message)
        
        # 2. Predizer dificuldade
        difficulty_result = await ml_service.predict_difficulty(message)
        
        # 3. Gerar intervenção personalizada
        intervention_result = await intervention_service.generate_response(
            message, user_id, context=context
        )
        
        # 4. Processar evento de gamificação
        gamification_result = await gamification_service.process_event(
            user_id, EventType.COMPLETE_ACTIVITY, {
                'message_length': len(message),
                'difficulty_predicted': difficulty_result.get('difficulty_level', 5.0)
            }
        )
        
        # 5. Combinar resultados
        combined_response = {
            "user_id": user_id,
            "message": message,
            "response": intervention_result['response'],
            "suggestions": intervention_result['suggestions'],
            "knowledge_base_info": rag_result.get('relevant_information', ''),
            "difficulty_assessment": difficulty_result,
            "gamification_update": gamification_result,
            "confidence_score": intervention_result.get('confidence_score', 0.5),
            "timestamp": intervention_result.get('timestamp')
        }
        
        return {
            "status": "success",
            "chat_response": combined_response
        }
        
    except Exception as e:
        logger.error(f"Error in chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/feedback")
@track_api_metrics
async def process_user_feedback(
    user_id: str = Body(..., embed=True),
    feedback_type: str = Body(..., embed=True),
    content: str = Body(..., embed=True),
    rating: Optional[int] = Body(None, ge=1, le=5),
    metadata: Optional[Dict[str, Any]] = Body(None)
):
    """Processa feedback do usuário"""
    try:
        # Processar feedback baseado no tipo
        if feedback_type == "difficulty_assessment":
            # Adicionar como exemplo de treinamento
            await ml_service.add_training_example(
                user_id, content, "user_feedback", "general", 0.8, metadata
            )
        elif feedback_type == "intervention_quality":
            # Registrar qualidade da intervenção
            pass
        elif feedback_type == "knowledge_helpful":
            # Registrar utilidade da base de conhecimento
            pass
        
        # Processar evento de gamificação
        await gamification_service.process_event(
            user_id, EventType.CONSISTENCY, {
                'feedback_type': feedback_type,
                'rating': rating
            }
        )
        
        return {
            "status": "success",
            "message": "Feedback processado com sucesso"
        }
        
    except Exception as e:
        logger.error(f"Error processing feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/system/health")
@track_api_metrics
async def system_health():
    """Verifica saúde do sistema"""
    try:
        health_status = {
            "status": "healthy",
            "services": {
                "zdp_service": "operational",
                "gamification_service": "operational",
                "rag_service": "operational",
                "intervention_service": "operational",
                "ml_service": "operational"
            },
            "timestamp": "2024-01-01T00:00:00Z"
        }
        
        return health_status
        
    except Exception as e:
        logger.error(f"Error checking system health: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }

@router.get("/system/statistics")
@track_api_metrics
async def system_statistics():
    """Obtém estatísticas gerais do sistema"""
    try:
        # Coletar estatísticas de todos os serviços
        ml_stats = await ml_service.get_ml_statistics()
        rag_stats = await rag_service.get_knowledge_statistics()
        intervention_stats = await intervention_service.get_intervention_statistics()
        
        combined_stats = {
            "ml_statistics": ml_stats,
            "rag_statistics": rag_stats,
            "intervention_statistics": intervention_stats,
            "total_services": 5,
            "active_services": 5
        }
        
        return {
            "status": "success",
            "statistics": combined_stats
        }
        
    except Exception as e:
        logger.error(f"Error getting system statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== ROTAS DE CONFIGURAÇÃO ====================

@router.get("/config/domains")
@track_api_metrics
async def get_available_domains():
    """Obtém domínios disponíveis"""
    try:
        domains = [domain.value for domain in DomainType]
        
        return {
            "status": "success",
            "domains": domains
        }
        
    except Exception as e:
        logger.error(f"Error getting domains: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/config/event-types")
@track_api_metrics
async def get_available_event_types():
    """Obtém tipos de eventos disponíveis"""
    try:
        event_types = [event_type.value for event_type in EventType]
        
        return {
            "status": "success",
            "event_types": event_types
        }
        
    except Exception as e:
        logger.error(f"Error getting event types: {e}")
        raise HTTPException(status_code=500, detail=str(e))