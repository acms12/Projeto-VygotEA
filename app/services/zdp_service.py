"""
Serviço de Zona de Desenvolvimento Proximal (ZDP) com modelos adaptativos
"""
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import structlog
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import json

from app.models.zdp import (
    ZDPAssessment, DomainAssessment, AdaptiveFactor, 
    ZDPRecommendation, ZDPHistory, DomainType, SupportLevel,
    ConfidenceLevel, LearningStyle
)
from app.core.circuit_breaker import ml_model_circuit_breaker
from app.core.metrics import track_zdp_metrics

logger = structlog.get_logger(__name__)

class AdaptiveZDPCalculator:
    """Calculadora adaptativa de ZDP com modelos de ML"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.regression_model = LinearRegression()
        self.historical_data = {}
        self.domain_models = {}
        
    def calculate_current_level(
        self, 
        user_id: str, 
        domain: DomainType, 
        assessment_data: Dict[str, Any],
        historical_assessments: List[DomainAssessment]
    ) -> Tuple[float, float]:
        """
        Calcula nível atual com modelo adaptativo baseado em histórico
        """
        try:
            # Análise de tendência temporal
            if len(historical_assessments) >= 3:
                trend_analysis = self._analyze_trend(historical_assessments)
                base_level = self._calculate_base_level(assessment_data)
                
                # Ajuste baseado em tendência e variância
                trend_adjustment = trend_analysis['slope'] * 0.3
                variance_factor = 1 - (trend_analysis['variance'] * 0.2)
                
                adjusted_level = base_level + trend_adjustment
                confidence = self._calculate_confidence(
                    assessment_data, 
                    trend_analysis['consistency']
                ) * variance_factor
                
                return adjusted_level, confidence
            
            else:
                # Primeira avaliação - usar modelo mais conservador
                base_level = self._calculate_base_level(assessment_data)
                confidence = self._calculate_confidence(assessment_data, 0.5)
                return base_level, confidence
                
        except Exception as e:
            logger.error(f"Error calculating current level: {e}")
            # Fallback para cálculo simples
            return self._fallback_calculation(assessment_data)
    
    def determine_proximal_zone(
        self, 
        current_level: float, 
        confidence: float,
        adaptive_factors: AdaptiveFactor,
        domain: DomainType
    ) -> Tuple[float, float]:
        """
        Determina zona proximal dinamicamente baseada em fatores adaptativos
        """
        try:
            # Largura base da zona
            base_width = 0.2
            
            # Ajuste baseado na confiança
            confidence_adjustment = (1 - confidence) * 0.3
            
            # Ajuste baseado no estilo de aprendizado
            learning_rate_factor = adaptive_factors.learning_rate * 0.2
            
            # Ajuste baseado na variância histórica
            variance_adjustment = adaptive_factors.historical_variance * 0.15
            
            # Largura final da zona
            zone_width = base_width + confidence_adjustment + learning_rate_factor + variance_adjustment
            
            # Limitar largura da zona
            zone_width = max(0.1, min(zone_width, 0.5))
            
            # Calcular limites da zona
            zone_lower = max(0, current_level - (zone_width * current_level))
            zone_upper = min(10, current_level + (zone_width * current_level))
            
            return zone_lower, zone_upper
            
        except Exception as e:
            logger.error(f"Error determining proximal zone: {e}")
            # Fallback para zona fixa
            return max(0, current_level - 0.2), min(10, current_level + 0.2)
    
    def calculate_development_potential(
        self, 
        domain_assessment: DomainAssessment,
        user_history: ZDPHistory
    ) -> float:
        """
        Calcula potencial de desenvolvimento considerando interdependências
        """
        try:
            # Potencial base
            base_potential = domain_assessment.zone_width * domain_assessment.confidence
            
            # Fatores adaptativos
            adaptive_boost = (
                domain_assessment.adaptive_factors.learning_rate * 
                domain_assessment.adaptive_factors.engagement_level * 
                (1 + domain_assessment.adaptive_factors.confidence_trend)
            )
            
            # Análise de interdependência entre domínios
            interdependence_factor = self._calculate_interdependence_factor(
                domain_assessment.domain, 
                user_history
            )
            
            # Potencial final
            final_potential = base_potential + adaptive_boost
            final_potential *= interdependence_factor
            
            return min(final_potential, 10.0)
            
        except Exception as e:
            logger.error(f"Error calculating development potential: {e}")
            return domain_assessment.zone_width * domain_assessment.confidence
    
    def generate_recommendations(
        self, 
        assessment: ZDPAssessment,
        user_history: ZDPHistory
    ) -> ZDPRecommendation:
        """
        Gera recomendações personalizadas usando análise avançada
        """
        try:
            # Identificar domínios prioritários
            primary_focus = self._identify_primary_focus(assessment, user_history)
            secondary_focus = self._identify_secondary_focus(assessment, primary_focus)
            
            # Gerar atividades recomendadas
            activities = self._generate_activities(
                primary_focus, 
                assessment.domains[primary_focus],
                assessment.learning_style
            )
            
            # Calcular nível de dificuldade ideal
            difficulty = self._calculate_optimal_difficulty(
                assessment.domains[primary_focus],
                assessment.support_level
            )
            
            # Gerar sugestões adaptativas
            adaptive_suggestions = self._generate_adaptive_suggestions(
                assessment, 
                primary_focus
            )
            
            return ZDPRecommendation(
                user_id=assessment.user_id,
                primary_focus=primary_focus,
                secondary_focus=secondary_focus,
                recommended_activities=activities,
                difficulty_level=difficulty,
                estimated_duration=self._estimate_duration(difficulty, assessment),
                confidence_score=self._calculate_recommendation_confidence(assessment),
                reasoning=self._generate_reasoning(assessment, primary_focus),
                adaptive_suggestions=adaptive_suggestions
            )
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return self._fallback_recommendation(assessment)
    
    def determine_support_level(
        self, 
        assessment: ZDPAssessment,
        user_history: ZDPHistory
    ) -> SupportLevel:
        """
        Determina nível de suporte de forma granular
        """
        try:
            # Análise de múltiplos fatores
            confidence_score = np.mean([d.confidence for d in assessment.domains.values()])
            engagement_level = np.mean([d.adaptive_factors.engagement_level for d in assessment.domains.values()])
            learning_rate = np.mean([d.adaptive_factors.learning_rate for d in assessment.domains.values()])
            
            # Histórico de sucesso
            success_rate = self._calculate_success_rate(user_history)
            
            # Score composto
            support_score = (
                confidence_score * 0.3 +
                engagement_level * 0.25 +
                learning_rate * 0.2 +
                success_rate * 0.25
            )
            
            # Mapear para nível de suporte
            if support_score >= 0.8:
                return SupportLevel.MINIMAL
            elif support_score >= 0.6:
                return SupportLevel.MODERATE
            elif support_score >= 0.4:
                return SupportLevel.SCAFFOLDING
            else:
                return SupportLevel.EXTENSIVE
                
        except Exception as e:
            logger.error(f"Error determining support level: {e}")
            return SupportLevel.MODERATE
    
    def _analyze_trend(self, assessments: List[DomainAssessment]) -> Dict[str, float]:
        """Analisa tendência temporal dos dados"""
        if len(assessments) < 2:
            return {'slope': 0, 'variance': 0.5, 'consistency': 0.5}
        
        levels = [a.current_level for a in assessments]
        timestamps = [a.last_assessment.timestamp() for a in assessments]
        
        # Regressão linear para tendência
        X = np.array(timestamps).reshape(-1, 1)
        y = np.array(levels)
        
        self.regression_model.fit(X, y)
        slope = self.regression_model.coef_[0]
        
        # Calcular variância
        variance = np.var(levels)
        
        # Calcular consistência (menor variância = maior consistência)
        consistency = max(0, 1 - variance)
        
        return {
            'slope': slope,
            'variance': variance,
            'consistency': consistency
        }
    
    def _calculate_base_level(self, assessment_data: Dict[str, Any]) -> float:
        """Calcula nível base a partir dos dados de avaliação"""
        # Implementação seria baseada nos dados específicos
        # Por exemplo, média ponderada de diferentes métricas
        scores = assessment_data.get('scores', [])
        weights = assessment_data.get('weights', [])
        
        if scores and weights:
            return np.average(scores, weights=weights)
        elif scores:
            return np.mean(scores)
        else:
            return 5.0  # Valor padrão
    
    def _calculate_confidence(
        self, 
        assessment_data: Dict[str, Any], 
        consistency: float
    ) -> float:
        """Calcula nível de confiança da avaliação"""
        # Fatores que influenciam a confiança
        data_quality = assessment_data.get('data_quality', 0.7)
        response_consistency = assessment_data.get('response_consistency', 0.8)
        
        confidence = (data_quality + response_consistency + consistency) / 3
        return min(confidence, 1.0)
    
    def _calculate_interdependence_factor(
        self, 
        domain: DomainType, 
        user_history: ZDPHistory
    ) -> float:
        """Calcula fator de interdependência entre domínios"""
        # Mapeamento de interdependências conhecidas
        interdependencies = {
            DomainType.MATHEMATICS: [DomainType.SCIENCE, DomainType.TECHNOLOGY],
            DomainType.LANGUAGE: [DomainType.HISTORY, DomainType.SOCIAL_STUDIES],
            DomainType.SCIENCE: [DomainType.MATHEMATICS, DomainType.TECHNOLOGY],
            # ... outros mapeamentos
        }
        
        related_domains = interdependencies.get(domain, [])
        if not related_domains or not user_history.assessments:
            return 1.0
        
        # Calcular força dos domínios relacionados
        latest_assessment = user_history.assessments[-1]
        related_strengths = []
        
        for related_domain in related_domains:
            if related_domain in latest_assessment.domains:
                related_strengths.append(
                    latest_assessment.domains[related_domain].current_level
                )
        
        if related_strengths:
            avg_related_strength = np.mean(related_strengths)
            # Fator baseado na força dos domínios relacionados
            return 1.0 + (avg_related_strength - 5.0) * 0.1
        
        return 1.0
    
    def _identify_primary_focus(
        self, 
        assessment: ZDPAssessment, 
        user_history: ZDPHistory
    ) -> DomainType:
        """Identifica foco primário baseado em múltiplos critérios"""
        # Priorizar domínios com maior potencial de desenvolvimento
        potential_scores = {}
        
        for domain, domain_assessment in assessment.domains.items():
            potential = domain_assessment.development_potential
            engagement = domain_assessment.adaptive_factors.engagement_level
            confidence = domain_assessment.confidence
            
            # Score composto
            score = potential * 0.4 + engagement * 0.3 + confidence * 0.3
            potential_scores[domain] = score
        
        return max(potential_scores.items(), key=lambda x: x[1])[0]
    
    def _identify_secondary_focus(
        self, 
        assessment: ZDPAssessment, 
        primary_focus: DomainType
    ) -> Optional[DomainType]:
        """Identifica foco secundário"""
        # Evitar mesmo domínio do foco primário
        candidates = [
            domain for domain in assessment.domains.keys() 
            if domain != primary_focus
        ]
        
        if not candidates:
            return None
        
        # Escolher baseado no potencial de desenvolvimento
        return max(
            candidates,
            key=lambda d: assessment.domains[d].development_potential
        )
    
    def _generate_activities(
        self, 
        domain: DomainType, 
        domain_assessment: DomainAssessment,
        learning_style: LearningStyle
    ) -> List[str]:
        """Gera atividades personalizadas"""
        # Base de atividades por domínio e estilo de aprendizado
        activity_templates = {
            DomainType.MATHEMATICS: {
                LearningStyle.VISUAL: [
                    "Resolver problemas com diagramas",
                    "Criar gráficos para visualizar conceitos",
                    "Usar manipulativos virtuais"
                ],
                LearningStyle.AUDITORY: [
                    "Explicar conceitos em voz alta",
                    "Participar de discussões matemáticas",
                    "Criar mnemônicos para fórmulas"
                ],
                # ... outros estilos
            }
            # ... outros domínios
        }
        
        activities = activity_templates.get(domain, {}).get(learning_style, [])
        
        # Personalizar baseado no nível e confiança
        if domain_assessment.current_level < 5:
            activities.extend([
                "Praticar conceitos básicos",
                "Revisar fundamentos"
            ])
        elif domain_assessment.confidence < 0.7:
            activities.extend([
                "Reforçar conceitos com exemplos",
                "Praticar com feedback imediato"
            ])
        
        return activities[:5]  # Limitar a 5 atividades
    
    def _calculate_optimal_difficulty(
        self, 
        domain_assessment: DomainAssessment,
        support_level: SupportLevel
    ) -> float:
        """Calcula dificuldade ideal baseada na zona proximal"""
        zone_center = (domain_assessment.proximal_zone_lower + domain_assessment.proximal_zone_upper) / 2
        
        # Ajustar baseado no nível de suporte
        support_adjustments = {
            SupportLevel.MINIMAL: 0.2,
            SupportLevel.MODERATE: 0.1,
            SupportLevel.SCAFFOLDING: 0.0,
            SupportLevel.EXTENSIVE: -0.1
        }
        
        adjustment = support_adjustments.get(support_level, 0.0)
        return min(max(zone_center + adjustment, 0), 10)
    
    def _estimate_duration(
        self, 
        difficulty: float, 
        assessment: ZDPAssessment
    ) -> int:
        """Estima duração da atividade em minutos"""
        base_duration = 20  # minutos base
        
        # Ajustar baseado na dificuldade
        difficulty_factor = 1 + (difficulty - 5) * 0.1
        
        # Ajustar baseado no estilo de aprendizado
        style_factors = {
            LearningStyle.VISUAL: 1.0,
            LearningStyle.AUDITORY: 0.9,
            LearningStyle.KINESTHETIC: 1.2,
            LearningStyle.READING_WRITING: 1.1
        }
        
        style_factor = style_factors.get(assessment.learning_style, 1.0)
        
        return int(base_duration * difficulty_factor * style_factor)
    
    def _calculate_recommendation_confidence(self, assessment: ZDPAssessment) -> float:
        """Calcula confiança da recomendação"""
        # Baseado na consistência dos dados
        confidences = [d.confidence for d in assessment.domains.values()]
        return np.mean(confidences)
    
    def _generate_reasoning(
        self, 
        assessment: ZDPAssessment, 
        primary_focus: DomainType
    ) -> str:
        """Gera explicação para a recomendação"""
        domain_assessment = assessment.domains[primary_focus]
        
        reasoning_parts = []
        
        if domain_assessment.current_level < 5:
            reasoning_parts.append("Você está construindo uma base sólida")
        elif domain_assessment.current_level < 7:
            reasoning_parts.append("Você tem uma boa compreensão dos fundamentos")
        else:
            reasoning_parts.append("Você demonstra domínio avançado")
        
        if domain_assessment.confidence < 0.7:
            reasoning_parts.append("e pode se beneficiar de mais prática")
        else:
            reasoning_parts.append("e está pronto para desafios mais complexos")
        
        return " ".join(reasoning_parts)
    
    def _generate_adaptive_suggestions(
        self, 
        assessment: ZDPAssessment, 
        primary_focus: DomainType
    ) -> List[str]:
        """Gera sugestões adaptativas"""
        suggestions = []
        domain_assessment = assessment.domains[primary_focus]
        
        if domain_assessment.adaptive_factors.engagement_level < 0.6:
            suggestions.append("Tente conectar os conceitos com situações do cotidiano")
        
        if domain_assessment.adaptive_factors.confidence_trend < 0:
            suggestions.append("Revisite conceitos anteriores para reforçar a base")
        
        if assessment.learning_style == LearningStyle.VISUAL:
            suggestions.append("Use recursos visuais como gráficos e diagramas")
        elif assessment.learning_style == LearningStyle.AUDITORY:
            suggestions.append("Explique os conceitos em voz alta")
        
        return suggestions
    
    def _calculate_success_rate(self, user_history: ZDPHistory) -> float:
        """Calcula taxa de sucesso baseada no histórico"""
        if not user_history.assessments:
            return 0.5
        
        # Implementação seria baseada em dados de sucesso das atividades
        # Por enquanto, retorna valor baseado no progresso
        progress = user_history.overall_progress
        return min(max(progress / 2.0, 0.0), 1.0)  # Normalizar
    
    def _fallback_calculation(self, assessment_data: Dict[str, Any]) -> Tuple[float, float]:
        """Cálculo de fallback em caso de erro"""
        return 5.0, 0.5
    
    def _fallback_recommendation(self, assessment: ZDPAssessment) -> ZDPRecommendation:
        """Recomendação de fallback"""
        return ZDPRecommendation(
            user_id=assessment.user_id,
            primary_focus=list(assessment.domains.keys())[0],
            recommended_activities=["Praticar conceitos básicos"],
            difficulty_level=5.0,
            estimated_duration=20,
            confidence_score=0.5,
            reasoning="Recomendação baseada em dados disponíveis"
        )

class ZDPService:
    """Serviço principal de ZDP"""
    
    def __init__(self):
        self.calculator = AdaptiveZDPCalculator()
    
    @ml_model_circuit_breaker
    @track_zdp_metrics
    async def assess_user_zdp(
        self, 
        user_id: str, 
        assessment_data: Dict[str, Any],
        user_history: Optional[ZDPHistory] = None
    ) -> ZDPAssessment:
        """Avalia ZDP completa do usuário"""
        try:
            domains = {}
            
            # Avaliar cada domínio
            for domain in DomainType:
                domain_data = assessment_data.get(domain.value, {})
                historical_domain_assessments = []
                
                if user_history:
                    historical_domain_assessments = [
                        assessment.domains[domain].current_level 
                        for assessment in user_history.assessments 
                        if domain in assessment.domains
                    ]
                
                current_level, confidence = self.calculator.calculate_current_level(
                    user_id, domain, domain_data, historical_domain_assessments
                )
                
                # Calcular zona proximal
                adaptive_factors = self._extract_adaptive_factors(domain_data)
                zone_lower, zone_upper = self.calculator.determine_proximal_zone(
                    current_level, confidence, adaptive_factors, domain
                )
                
                # Determinar nível de suporte
                support_level = self._determine_domain_support_level(
                    current_level, confidence, adaptive_factors
                )
                
                # Criar avaliação do domínio
                domain_assessment = DomainAssessment(
                    domain=domain,
                    current_level=current_level,
                    confidence=confidence,
                    proximal_zone_lower=zone_lower,
                    proximal_zone_upper=zone_upper,
                    support_level=support_level,
                    adaptive_factors=adaptive_factors,
                    last_assessment=datetime.now(),
                    assessment_count=len(historical_domain_assessments) + 1
                )
                
                domains[domain] = domain_assessment
            
            # Calcular nível geral
            overall_level = np.mean([d.current_level for d in domains.values()])
            
            # Determinar estilo de aprendizado
            learning_style = self._determine_learning_style(assessment_data)
            
            # Determinar nível de confiança geral
            confidence_level = self._determine_confidence_level(domains)
            
            # Determinar prontidão para próximo nível
            readiness = self._calculate_readiness(domains, overall_level)
            
            return ZDPAssessment(
                user_id=user_id,
                domains=domains,
                overall_level=overall_level,
                support_level=self.calculator.determine_support_level(
                    ZDPAssessment(user_id=user_id, domains=domains, overall_level=overall_level, 
                                support_level=SupportLevel.MODERATE, learning_style=learning_style,
                                confidence_level=confidence_level, readiness_for_next_level=readiness),
                    user_history or ZDPHistory(user_id=user_id, assessments=[])
                ),
                learning_style=learning_style,
                confidence_level=confidence_level,
                readiness_for_next_level=readiness
            )
            
        except Exception as e:
            logger.error(f"Error in ZDP assessment: {e}")
            raise
    
    def _extract_adaptive_factors(self, domain_data: Dict[str, Any]) -> AdaptiveFactor:
        """Extrai fatores adaptativos dos dados"""
        return AdaptiveFactor(
            historical_variance=domain_data.get('variance', 0.3),
            learning_rate=domain_data.get('learning_rate', 1.0),
            confidence_trend=domain_data.get('confidence_trend', 0.0),
            engagement_level=domain_data.get('engagement', 0.7),
            difficulty_preference=domain_data.get('difficulty_pref', 0.5)
        )
    
    def _determine_domain_support_level(
        self, 
        level: float, 
        confidence: float, 
        adaptive_factors: AdaptiveFactor
    ) -> SupportLevel:
        """Determina nível de suporte para um domínio específico"""
        if level < 3 or confidence < 0.4:
            return SupportLevel.EXTENSIVE
        elif level < 6 or confidence < 0.7:
            return SupportLevel.SCAFFOLDING
        elif level < 8:
            return SupportLevel.MODERATE
        else:
            return SupportLevel.MINIMAL
    
    def _determine_learning_style(self, assessment_data: Dict[str, Any]) -> LearningStyle:
        """Determina estilo de aprendizado"""
        # Implementação seria baseada em análise de comportamento
        # Por enquanto, retorna estilo padrão
        return LearningStyle.VISUAL
    
    def _determine_confidence_level(self, domains: Dict[DomainType, DomainAssessment]) -> ConfidenceLevel:
        """Determina nível de confiança geral"""
        avg_confidence = np.mean([d.confidence for d in domains.values()])
        
        if avg_confidence >= 0.9:
            return ConfidenceLevel.VERY_HIGH
        elif avg_confidence >= 0.7:
            return ConfidenceLevel.HIGH
        elif avg_confidence >= 0.5:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW
    
    def _calculate_readiness(
        self, 
        domains: Dict[DomainType, DomainAssessment], 
        overall_level: float
    ) -> float:
        """Calcula prontidão para próximo nível"""
        # Baseado na consistência dos domínios e nível geral
        domain_levels = [d.current_level for d in domains.values()]
        consistency = 1 - np.std(domain_levels) / np.mean(domain_levels)
        
        readiness = (overall_level / 10.0) * 0.7 + consistency * 0.3
        return min(max(readiness, 0.0), 1.0)