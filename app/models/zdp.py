"""
Modelos de dados para Zona de Desenvolvimento Proximal (ZDP)
"""
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, validator
from enum import Enum
from datetime import datetime
import numpy as np

class DomainType(str, Enum):
    MATHEMATICS = "mathematics"
    LANGUAGE = "language"
    SCIENCE = "science"
    HISTORY = "history"
    ARTS = "arts"
    TECHNOLOGY = "technology"
    PHYSICAL_EDUCATION = "physical_education"
    SOCIAL_STUDIES = "social_studies"

class SupportLevel(str, Enum):
    MINIMAL = "minimal"
    MODERATE = "moderate"
    EXTENSIVE = "extensive"
    SCAFFOLDING = "scaffolding"

class ConfidenceLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"

class LearningStyle(str, Enum):
    VISUAL = "visual"
    AUDITORY = "auditory"
    KINESTHETIC = "kinesthetic"
    READING_WRITING = "reading_writing"

class AdaptiveFactor(BaseModel):
    """Fatores adaptativos para cálculo dinâmico de ZDP"""
    historical_variance: float = Field(..., ge=0, le=1)
    learning_rate: float = Field(..., ge=0, le=2)
    confidence_trend: float = Field(..., ge=-1, le=1)
    engagement_level: float = Field(..., ge=0, le=1)
    difficulty_preference: float = Field(..., ge=0, le=1)
    
    @validator('historical_variance')
    def validate_variance(cls, v):
        if v < 0 or v > 1:
            raise ValueError('historical_variance must be between 0 and 1')
        return v

class DomainAssessment(BaseModel):
    """Avaliação detalhada de um domínio específico"""
    domain: DomainType
    current_level: float = Field(..., ge=0, le=10)
    confidence: float = Field(..., ge=0, le=1)
    proximal_zone_lower: float = Field(..., ge=0, le=10)
    proximal_zone_upper: float = Field(..., ge=0, le=10)
    support_level: SupportLevel
    adaptive_factors: AdaptiveFactor
    last_assessment: datetime
    assessment_count: int = Field(..., ge=0)
    improvement_rate: float = Field(0, ge=-1, le=1)
    
    @validator('proximal_zone_upper')
    def validate_zone_upper(cls, v, values):
        if 'proximal_zone_lower' in values and v <= values['proximal_zone_lower']:
            raise ValueError('proximal_zone_upper must be greater than proximal_zone_lower')
        return v
    
    @property
    def zone_width(self) -> float:
        """Largura da zona proximal"""
        return self.proximal_zone_upper - self.proximal_zone_lower
    
    @property
    def development_potential(self) -> float:
        """Potencial de desenvolvimento baseado em fatores adaptativos"""
        base_potential = self.zone_width * self.confidence
        adaptive_boost = (
            self.adaptive_factors.learning_rate * 
            self.adaptive_factors.engagement_level * 
            (1 + self.adaptive_factors.confidence_trend)
        )
        return min(base_potential + adaptive_boost, 10.0)

class ZDPAssessment(BaseModel):
    """Avaliação completa de ZDP do usuário"""
    user_id: str
    timestamp: datetime = Field(default_factory=datetime.now)
    domains: Dict[DomainType, DomainAssessment]
    overall_level: float = Field(..., ge=0, le=10)
    support_level: SupportLevel
    learning_style: LearningStyle
    confidence_level: ConfidenceLevel
    readiness_for_next_level: float = Field(..., ge=0, le=1)
    
    @property
    def weakest_domains(self) -> List[DomainType]:
        """Retorna os domínios mais fracos ordenados"""
        sorted_domains = sorted(
            self.domains.items(),
            key=lambda x: x[1].current_level
        )
        return [domain for domain, _ in sorted_domains[:2]]
    
    @property
    def strongest_domains(self) -> List[DomainType]:
        """Retorna os domínios mais fortes ordenados"""
        sorted_domains = sorted(
            self.domains.items(),
            key=lambda x: x[1].current_level,
            reverse=True
        )
        return [domain for domain, _ in sorted_domains[:2]]
    
    @property
    def highest_potential_domains(self) -> List[DomainType]:
        """Retorna domínios com maior potencial de desenvolvimento"""
        sorted_domains = sorted(
            self.domains.items(),
            key=lambda x: x[1].development_potential,
            reverse=True
        )
        return [domain for domain, _ in sorted_domains[:3]]

class ZDPRecommendation(BaseModel):
    """Recomendação personalizada baseada em ZDP"""
    user_id: str
    timestamp: datetime = Field(default_factory=datetime.now)
    primary_focus: DomainType
    secondary_focus: Optional[DomainType] = None
    recommended_activities: List[str]
    difficulty_level: float = Field(..., ge=0, le=10)
    estimated_duration: int = Field(..., ge=1)  # em minutos
    confidence_score: float = Field(..., ge=0, le=1)
    reasoning: str
    adaptive_suggestions: List[str] = []
    
    class Config:
        schema_extra = {
            "example": {
                "user_id": "user123",
                "primary_focus": "mathematics",
                "secondary_focus": "language",
                "recommended_activities": [
                    "Resolver problemas de álgebra",
                    "Praticar equações lineares"
                ],
                "difficulty_level": 6.5,
                "estimated_duration": 30,
                "confidence_score": 0.85,
                "reasoning": "Usuário demonstra boa base em matemática mas precisa de prática em álgebra",
                "adaptive_suggestions": [
                    "Usar recursos visuais para conceitos abstratos",
                    "Incluir exemplos práticos do cotidiano"
                ]
            }
        }

class ZDPHistory(BaseModel):
    """Histórico de avaliações ZDP"""
    user_id: str
    assessments: List[ZDPAssessment]
    
    @property
    def progress_trend(self) -> Dict[DomainType, List[float]]:
        """Tendência de progresso por domínio"""
        trends = {}
        for domain in DomainType:
            domain_assessments = [
                assessment.domains[domain].current_level 
                for assessment in self.assessments 
                if domain in assessment.domains
            ]
            if domain_assessments:
                trends[domain] = domain_assessments
        return trends
    
    @property
    def overall_progress(self) -> float:
        """Progresso geral baseado na média dos níveis"""
        if not self.assessments:
            return 0.0
        
        first_level = self.assessments[0].overall_level
        last_level = self.assessments[-1].overall_level
        return last_level - first_level

class AdaptiveLearningProfile(BaseModel):
    """Perfil de aprendizado adaptativo"""
    user_id: str
    learning_style: LearningStyle
    preferred_difficulty: float = Field(..., ge=0, le=10)
    engagement_threshold: float = Field(..., ge=0, le=1)
    frustration_threshold: float = Field(..., ge=0, le=1)
    optimal_session_duration: int = Field(..., ge=5, le=120)  # em minutos
    preferred_feedback_frequency: int = Field(..., ge=1, le=10)  # a cada N atividades
    adaptive_preferences: Dict[str, Any] = {}
    
    class Config:
        schema_extra = {
            "example": {
                "user_id": "user123",
                "learning_style": "visual",
                "preferred_difficulty": 7.0,
                "engagement_threshold": 0.6,
                "frustration_threshold": 0.8,
                "optimal_session_duration": 25,
                "preferred_feedback_frequency": 3,
                "adaptive_preferences": {
                    "visual_aids": True,
                    "step_by_step_guidance": True,
                    "immediate_feedback": False
                }
            }
        }