"""
Motor de Intervenção Inteligente com geração de respostas contextuais
"""
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime
import structlog
import json
import re

from app.models.zdp import ZDPAssessment, ZDPRecommendation, SupportLevel, LearningStyle
from app.models.gamification import UserGamificationProfile, EventType
from app.core.circuit_breaker import external_api_circuit_breaker
from app.core.metrics import track_api_metrics

logger = structlog.get_logger(__name__)

class InterventionTemplate:
    """Template de intervenção personalizada"""
    
    def __init__(
        self,
        template_id: str,
        name: str,
        content: str,
        conditions: Dict[str, Any],
        priority: int = 1,
        is_active: bool = True
    ):
        self.template_id = template_id
        self.name = name
        self.content = content
        self.conditions = conditions
        self.priority = priority
        self.is_active = is_active

class IntelligentInterventionEngine:
    """Motor de intervenção inteligente"""
    
    def __init__(self):
        self.templates = {}
        self.intervention_history = []
        self.response_patterns = {}
        
        self._initialize_templates()
        self._initialize_response_patterns()
    
    def _initialize_templates(self):
        """Inicializa templates de intervenção"""
        templates_data = [
            {
                'id': 'encouragement_low_confidence',
                'name': 'Encouragement for Low Confidence',
                'content': 'Vejo que você está enfrentando alguns desafios. Lembre-se: cada pequeno progresso é uma vitória! Vamos trabalhar juntos para superar essa dificuldade.',
                'conditions': {
                    'confidence_level': 'low',
                    'support_level': ['extensive', 'scaffolding']
                },
                'priority': 1
            },
            {
                'id': 'challenge_high_confidence',
                'name': 'Challenge for High Confidence',
                'content': 'Excelente trabalho! Você demonstrou domínio sólido. Que tal tentar algo mais desafiador? Acredito que você está pronto para o próximo nível.',
                'conditions': {
                    'confidence_level': 'high',
                    'support_level': ['minimal', 'moderate']
                },
                'priority': 2
            },
            {
                'id': 'guidance_moderate_support',
                'name': 'Guidance for Moderate Support',
                'content': 'Você está no caminho certo! Vou te dar algumas dicas para ajudar no próximo passo. Lembre-se de revisar os conceitos básicos antes de avançar.',
                'conditions': {
                    'support_level': 'moderate'
                },
                'priority': 3
            },
            {
                'id': 'celebration_achievement',
                'name': 'Celebration for Achievement',
                'content': 'Parabéns! Você conseguiu! Esse é o resultado do seu esforço e dedicação. Continue assim e você vai longe!',
                'conditions': {
                    'event_type': 'achievement',
                    'gamification_event': True
                },
                'priority': 1
            },
            {
                'id': 'motivation_streak',
                'name': 'Motivation for Streak',
                'content': 'Incrível! Você está mantendo uma sequência impressionante de estudos. Essa consistência é a chave do sucesso. Mantenha o ritmo!',
                'conditions': {
                    'streak_days': {'min': 7},
                    'gamification_event': True
                },
                'priority': 2
            },
            {
                'id': 'struggle_support',
                'name': 'Support for Struggle',
                'content': 'Entendo que isso pode estar difícil. Não se preocupe, todos passamos por momentos assim. Vamos quebrar isso em partes menores e trabalhar passo a passo.',
                'conditions': {
                    'difficulty_level': {'min': 7},
                    'confidence_level': 'low'
                },
                'priority': 1
            },
            {
                'id': 'learning_style_adaptation',
                'name': 'Learning Style Adaptation',
                'content': 'Baseado no seu estilo de aprendizado, sugiro que você {learning_style_suggestion}. Isso pode tornar o processo mais eficaz para você.',
                'conditions': {
                    'learning_style': 'any'
                },
                'priority': 3
            }
        ]
        
        for template_data in templates_data:
            template = InterventionTemplate(**template_data)
            self.templates[template.template_id] = template
    
    def _initialize_response_patterns(self):
        """Inicializa padrões de resposta"""
        self.response_patterns = {
            'encouragement': [
                "Você está fazendo um ótimo progresso!",
                "Continue assim, você está no caminho certo!",
                "Cada esforço conta, mantenha a persistência!",
                "Acredite em si mesmo, você é capaz!",
                "Pequenos passos levam a grandes conquistas!"
            ],
            'guidance': [
                "Vamos trabalhar isso juntos, passo a passo.",
                "Que tal tentarmos uma abordagem diferente?",
                "Vou te dar algumas dicas que podem ajudar.",
                "Vamos quebrar isso em partes menores.",
                "Tente conectar isso com algo que você já sabe."
            ],
            'celebration': [
                "Parabéns pela conquista!",
                "Excelente trabalho!",
                "Você merece comemorar essa vitória!",
                "Que resultado incrível!",
                "Continue assim, você está arrasando!"
            ],
            'challenge': [
                "Que tal tentar algo mais desafiador?",
                "Você está pronto para o próximo nível!",
                "Vamos aumentar um pouco a dificuldade?",
                "Acredito que você pode ir além!",
                "Que tal testar seus limites?"
            ]
        }
    
    def generate_intervention(
        self, 
        user_message: str,
        zdp_assessment: Optional[ZDPAssessment] = None,
        gamification_profile: Optional[UserGamificationProfile] = None,
        recent_events: List[Dict[str, Any]] = None,
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Gera intervenção personalizada baseada no contexto
        """
        try:
            # Analisar contexto
            context_analysis = self._analyze_context(
                user_message, zdp_assessment, gamification_profile, recent_events, context
            )
            
            # Selecionar template apropriado
            selected_template = self._select_template(context_analysis)
            
            # Gerar resposta personalizada
            personalized_response = self._generate_personalized_response(
                selected_template, context_analysis, user_message
            )
            
            # Adicionar sugestões específicas
            specific_suggestions = self._generate_specific_suggestions(context_analysis)
            
            # Determinar nível de urgência
            urgency_level = self._determine_urgency(context_analysis)
            
            # Registrar intervenção
            intervention_record = {
                'timestamp': datetime.now(),
                'user_message': user_message,
                'template_used': selected_template.template_id if selected_template else None,
                'context_analysis': context_analysis,
                'urgency_level': urgency_level
            }
            self.intervention_history.append(intervention_record)
            
            return {
                'response_text': personalized_response,
                'suggestions': specific_suggestions,
                'urgency_level': urgency_level,
                'template_used': selected_template.template_id if selected_template else None,
                'context_analysis': context_analysis,
                'confidence_score': self._calculate_response_confidence(context_analysis)
            }
            
        except Exception as e:
            logger.error(f"Error generating intervention: {e}")
            return {
                'response_text': "Entendo sua pergunta. Como posso te ajudar hoje?",
                'suggestions': ["Continue explorando os recursos disponíveis"],
                'urgency_level': 'normal',
                'error': str(e)
            }
    
    def _analyze_context(
        self, 
        user_message: str,
        zdp_assessment: Optional[ZDPAssessment],
        gamification_profile: Optional[UserGamificationProfile],
        recent_events: List[Dict[str, Any]],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analisa contexto para personalização"""
        analysis = {
            'message_sentiment': self._analyze_sentiment(user_message),
            'user_state': self._analyze_user_state(zdp_assessment, gamification_profile),
            'recent_activity': self._analyze_recent_activity(recent_events),
            'learning_needs': self._identify_learning_needs(user_message, zdp_assessment),
            'motivation_level': self._assess_motivation(gamification_profile, recent_events),
            'difficulty_level': self._assess_difficulty(user_message, zdp_assessment),
            'support_needs': self._assess_support_needs(zdp_assessment, gamification_profile)
        }
        
        return analysis
    
    def _analyze_sentiment(self, message: str) -> str:
        """Analisa sentimento da mensagem do usuário"""
        message_lower = message.lower()
        
        # Palavras positivas
        positive_words = ['ótimo', 'bom', 'legal', 'gosto', 'interessante', 'fácil', 'consegui']
        # Palavras negativas
        negative_words = ['difícil', 'não consigo', 'problema', 'confuso', 'frustrado', 'cansado']
        # Palavras de dúvida
        doubt_words = ['não sei', 'como', 'por que', 'quando', 'onde', 'quem']
        
        positive_count = sum(1 for word in positive_words if word in message_lower)
        negative_count = sum(1 for word in negative_words if word in message_lower)
        doubt_count = sum(1 for word in doubt_words if word in message_lower)
        
        if negative_count > positive_count:
            return 'negative'
        elif doubt_count > 0:
            return 'confused'
        elif positive_count > negative_count:
            return 'positive'
        else:
            return 'neutral'
    
    def _analyze_user_state(
        self, 
        zdp_assessment: Optional[ZDPAssessment],
        gamification_profile: Optional[UserGamificationProfile]
    ) -> Dict[str, Any]:
        """Analisa estado atual do usuário"""
        state = {
            'confidence_level': 'medium',
            'support_level': 'moderate',
            'learning_style': 'visual',
            'engagement_level': 'medium',
            'progress_level': 'intermediate'
        }
        
        if zdp_assessment:
            state['confidence_level'] = zdp_assessment.confidence_level.value
            state['support_level'] = zdp_assessment.support_level.value
            state['learning_style'] = zdp_assessment.learning_style.value
        
        if gamification_profile:
            if gamification_profile.current_streak > 7:
                state['engagement_level'] = 'high'
            elif gamification_profile.current_streak > 3:
                state['engagement_level'] = 'medium'
            else:
                state['engagement_level'] = 'low'
            
            if gamification_profile.current_level > 10:
                state['progress_level'] = 'advanced'
            elif gamification_profile.current_level > 5:
                state['progress_level'] = 'intermediate'
            else:
                state['progress_level'] = 'beginner'
        
        return state
    
    def _analyze_recent_activity(self, recent_events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analisa atividade recente do usuário"""
        if not recent_events:
            return {'activity_level': 'low', 'last_activity': None, 'event_types': []}
        
        activity_analysis = {
            'activity_level': 'medium',
            'last_activity': recent_events[-1].get('timestamp'),
            'event_types': [event.get('type') for event in recent_events],
            'success_rate': 0.0,
            'difficulty_trend': 'stable'
        }
        
        # Calcular taxa de sucesso
        success_events = [e for e in recent_events if e.get('success', False)]
        if recent_events:
            activity_analysis['success_rate'] = len(success_events) / len(recent_events)
        
        # Determinar nível de atividade
        if len(recent_events) > 10:
            activity_analysis['activity_level'] = 'high'
        elif len(recent_events) < 3:
            activity_analysis['activity_level'] = 'low'
        
        return activity_analysis
    
    def _identify_learning_needs(
        self, 
        message: str, 
        zdp_assessment: Optional[ZDPAssessment]
    ) -> List[str]:
        """Identifica necessidades de aprendizado"""
        needs = []
        message_lower = message.lower()
        
        # Análise baseada em palavras-chave
        if any(word in message_lower for word in ['matemática', 'álgebra', 'equação']):
            needs.append('mathematics')
        if any(word in message_lower for word in ['português', 'gramática', 'literatura']):
            needs.append('language')
        if any(word in message_lower for word in ['ciência', 'física', 'química']):
            needs.append('science')
        
        # Análise baseada na ZDP
        if zdp_assessment:
            weakest_domains = zdp_assessment.weakest_domains
            for domain in weakest_domains:
                if domain.value not in needs:
                    needs.append(domain.value)
        
        return needs
    
    def _assess_motivation(
        self, 
        gamification_profile: Optional[UserGamificationProfile],
        recent_events: List[Dict[str, Any]]
    ) -> str:
        """Avalia nível de motivação"""
        if not gamification_profile:
            return 'medium'
        
        # Fatores de motivação
        factors = []
        
        # Streak atual
        if gamification_profile.current_streak > 7:
            factors.append(1.0)
        elif gamification_profile.current_streak > 3:
            factors.append(0.7)
        else:
            factors.append(0.3)
        
        # Nível atual
        if gamification_profile.current_level > 10:
            factors.append(0.9)
        elif gamification_profile.current_level > 5:
            factors.append(0.6)
        else:
            factors.append(0.4)
        
        # Atividade recente
        if recent_events and len(recent_events) > 5:
            factors.append(0.8)
        elif recent_events and len(recent_events) > 2:
            factors.append(0.5)
        else:
            factors.append(0.2)
        
        avg_motivation = np.mean(factors)
        
        if avg_motivation > 0.7:
            return 'high'
        elif avg_motivation > 0.4:
            return 'medium'
        else:
            return 'low'
    
    def _assess_difficulty(
        self, 
        message: str, 
        zdp_assessment: Optional[ZDPAssessment]
    ) -> float:
        """Avalia nível de dificuldade percebida"""
        difficulty_indicators = [
            'difícil', 'complexo', 'complicado', 'não entendo', 'confuso',
            'desafiador', 'problemático', 'intrincado'
        ]
        
        message_lower = message.lower()
        difficulty_score = sum(1 for word in difficulty_indicators if word in message_lower)
        
        # Normalizar para escala 0-10
        base_difficulty = min(difficulty_score * 2, 10)
        
        # Ajustar baseado na ZDP
        if zdp_assessment:
            avg_level = np.mean([d.current_level for d in zdp_assessment.domains.values()])
            # Se o usuário está em nível baixo, aumentar dificuldade percebida
            if avg_level < 5:
                base_difficulty += 2
        
        return min(base_difficulty, 10)
    
    def _assess_support_needs(
        self, 
        zdp_assessment: Optional[ZDPAssessment],
        gamification_profile: Optional[UserGamificationProfile]
    ) -> str:
        """Avalia necessidades de suporte"""
        if not zdp_assessment:
            return 'moderate'
        
        # Baseado no nível de suporte da ZDP
        support_mapping = {
            SupportLevel.EXTENSIVE: 'high',
            SupportLevel.SCAFFOLDING: 'high',
            SupportLevel.MODERATE: 'moderate',
            SupportLevel.MINIMAL: 'low'
        }
        
        return support_mapping.get(zdp_assessment.support_level, 'moderate')
    
    def _select_template(self, context_analysis: Dict[str, Any]) -> Optional[InterventionTemplate]:
        """Seleciona template apropriado baseado no contexto"""
        matching_templates = []
        
        for template in self.templates.values():
            if not template.is_active:
                continue
            
            if self._template_matches_context(template, context_analysis):
                matching_templates.append(template)
        
        if not matching_templates:
            return None
        
        # Selecionar template com maior prioridade
        return max(matching_templates, key=lambda t: t.priority)
    
    def _template_matches_context(
        self, 
        template: InterventionTemplate, 
        context: Dict[str, Any]
    ) -> bool:
        """Verifica se template corresponde ao contexto"""
        conditions = template.conditions
        
        for key, expected_value in conditions.items():
            if key == 'confidence_level':
                if context['user_state']['confidence_level'] != expected_value:
                    return False
            elif key == 'support_level':
                if isinstance(expected_value, list):
                    if context['user_state']['support_level'] not in expected_value:
                        return False
                else:
                    if context['user_state']['support_level'] != expected_value:
                        return False
            elif key == 'learning_style':
                if expected_value != 'any' and context['user_state']['learning_style'] != expected_value:
                    return False
            elif key == 'streak_days':
                # Implementação seria baseada em dados de gamificação
                pass
            elif key == 'difficulty_level':
                if isinstance(expected_value, dict):
                    min_difficulty = expected_value.get('min', 0)
                    if context['difficulty_level'] < min_difficulty:
                        return False
        
        return True
    
    def _generate_personalized_response(
        self, 
        template: Optional[InterventionTemplate],
        context_analysis: Dict[str, Any],
        user_message: str
    ) -> str:
        """Gera resposta personalizada"""
        if not template:
            # Resposta padrão baseada no sentimento
            sentiment = context_analysis['message_sentiment']
            if sentiment == 'negative':
                return "Entendo que isso pode estar difícil. Vamos trabalhar juntos para superar essa dificuldade."
            elif sentiment == 'confused':
                return "Vou te ajudar a esclarecer essa dúvida. Vamos por partes."
            else:
                return "Como posso te ajudar hoje?"
        
        # Personalizar template
        response = template.content
        
        # Substituir placeholders
        if '{learning_style_suggestion}' in response:
            learning_style = context_analysis['user_state']['learning_style']
            suggestion = self._get_learning_style_suggestion(learning_style)
            response = response.replace('{learning_style_suggestion}', suggestion)
        
        # Adicionar elementos personalizados
        if context_analysis['motivation_level'] == 'low':
            response += " Lembre-se: cada pequeno passo conta!"
        elif context_analysis['motivation_level'] == 'high':
            response += " Continue assim, você está arrasando!"
        
        return response
    
    def _get_learning_style_suggestion(self, learning_style: str) -> str:
        """Obtém sugestão baseada no estilo de aprendizado"""
        suggestions = {
            'visual': 'use diagramas e gráficos para visualizar os conceitos',
            'auditory': 'explique os conceitos em voz alta ou discuta com alguém',
            'kinesthetic': 'use exemplos práticos e experimentos',
            'reading_writing': 'escreva resumos e faça anotações detalhadas'
        }
        
        return suggestions.get(learning_style, 'experimente diferentes abordagens')
    
    def _generate_specific_suggestions(self, context_analysis: Dict[str, Any]) -> List[str]:
        """Gera sugestões específicas baseadas no contexto"""
        suggestions = []
        
        # Sugestões baseadas no sentimento
        sentiment = context_analysis['message_sentiment']
        if sentiment == 'negative':
            suggestions.append("Tente quebrar o problema em partes menores")
            suggestions.append("Revisite conceitos básicos relacionados")
        elif sentiment == 'confused':
            suggestions.append("Peça exemplos práticos")
            suggestions.append("Tente explicar o conceito para si mesmo")
        
        # Sugestões baseadas no estilo de aprendizado
        learning_style = context_analysis['user_state']['learning_style']
        if learning_style == 'visual':
            suggestions.append("Crie mapas mentais ou diagramas")
        elif learning_style == 'auditory':
            suggestions.append("Grave explicações em áudio")
        elif learning_style == 'kinesthetic':
            suggestions.append("Use objetos físicos para demonstrar")
        
        # Sugestões baseadas na motivação
        motivation = context_analysis['motivation_level']
        if motivation == 'low':
            suggestions.append("Estabeleça metas pequenas e alcançáveis")
            suggestions.append("Celebre cada pequena conquista")
        
        # Sugestões baseadas nas necessidades de aprendizado
        learning_needs = context_analysis['learning_needs']
        for need in learning_needs:
            if need == 'mathematics':
                suggestions.append("Pratique com exercícios progressivos")
            elif need == 'language':
                suggestions.append("Leia textos variados sobre o tema")
            elif need == 'science':
                suggestions.append("Conecte com fenômenos do cotidiano")
        
        return suggestions[:5]  # Limitar a 5 sugestões
    
    def _determine_urgency(self, context_analysis: Dict[str, Any]) -> str:
        """Determina nível de urgência da intervenção"""
        urgency_factors = []
        
        # Sentimento negativo
        if context_analysis['message_sentiment'] == 'negative':
            urgency_factors.append(1.0)
        
        # Baixa motivação
        if context_analysis['motivation_level'] == 'low':
            urgency_factors.append(0.8)
        
        # Alta dificuldade
        if context_analysis['difficulty_level'] > 8:
            urgency_factors.append(0.9)
        
        # Baixa confiança
        if context_analysis['user_state']['confidence_level'] == 'low':
            urgency_factors.append(0.7)
        
        # Alta necessidade de suporte
        if context_analysis['support_needs'] == 'high':
            urgency_factors.append(0.6)
        
        if not urgency_factors:
            return 'normal'
        
        avg_urgency = np.mean(urgency_factors)
        
        if avg_urgency > 0.8:
            return 'high'
        elif avg_urgency > 0.5:
            return 'medium'
        else:
            return 'normal'
    
    def _calculate_response_confidence(self, context_analysis: Dict[str, Any]) -> float:
        """Calcula confiança da resposta gerada"""
        confidence_factors = []
        
        # Mais dados disponíveis = maior confiança
        if context_analysis['user_state']['confidence_level'] != 'medium':
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.5)
        
        # Análise de sentimento clara
        if context_analysis['message_sentiment'] != 'neutral':
            confidence_factors.append(0.7)
        else:
            confidence_factors.append(0.4)
        
        # Necessidades de aprendizado identificadas
        if context_analysis['learning_needs']:
            confidence_factors.append(0.9)
        else:
            confidence_factors.append(0.3)
        
        return np.mean(confidence_factors) if confidence_factors else 0.5

class InterventionService:
    """Serviço principal de intervenção"""
    
    def __init__(self):
        self.engine = IntelligentInterventionEngine()
    
    @external_api_circuit_breaker
    @track_api_metrics
    async def generate_response(
        self, 
        user_message: str,
        user_id: str,
        zdp_assessment: Optional[ZDPAssessment] = None,
        gamification_profile: Optional[UserGamificationProfile] = None,
        recent_events: List[Dict[str, Any]] = None,
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Gera resposta personalizada para o usuário"""
        try:
            intervention = self.engine.generate_intervention(
                user_message, zdp_assessment, gamification_profile, recent_events, context
            )
            
            return {
                'user_id': user_id,
                'response': intervention['response_text'],
                'suggestions': intervention['suggestions'],
                'urgency_level': intervention['urgency_level'],
                'confidence_score': intervention.get('confidence_score', 0.5),
                'template_used': intervention.get('template_used'),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {
                'user_id': user_id,
                'response': "Desculpe, ocorreu um erro ao processar sua mensagem. Como posso te ajudar?",
                'suggestions': ["Tente reformular sua pergunta"],
                'urgency_level': 'normal',
                'confidence_score': 0.0,
                'error': str(e)
            }
    
    async def get_intervention_history(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Obtém histórico de intervenções do usuário"""
        try:
            user_interventions = [
                record for record in self.engine.intervention_history
                if record.get('user_id') == user_id
            ]
            
            # Ordenar por timestamp e limitar
            sorted_interventions = sorted(
                user_interventions, 
                key=lambda x: x['timestamp'], 
                reverse=True
            )[:limit]
            
            return [
                {
                    'timestamp': intervention['timestamp'].isoformat(),
                    'user_message': intervention['user_message'],
                    'template_used': intervention['template_used'],
                    'urgency_level': intervention['urgency_level']
                }
                for intervention in sorted_interventions
            ]
            
        except Exception as e:
            logger.error(f"Error getting intervention history: {e}")
            return []
    
    async def add_intervention_template(
        self, 
        template_id: str,
        name: str,
        content: str,
        conditions: Dict[str, Any],
        priority: int = 1
    ) -> Dict[str, Any]:
        """Adiciona novo template de intervenção"""
        try:
            template = InterventionTemplate(
                template_id=template_id,
                name=name,
                content=content,
                conditions=conditions,
                priority=priority
            )
            
            self.engine.templates[template_id] = template
            
            return {
                'status': 'success',
                'message': 'Template adicionado com sucesso',
                'template_id': template_id
            }
            
        except Exception as e:
            logger.error(f"Error adding intervention template: {e}")
            return {
                'status': 'error',
                'message': f'Erro ao adicionar template: {str(e)}'
            }
    
    async def get_intervention_statistics(self) -> Dict[str, Any]:
        """Obtém estatísticas do sistema de intervenção"""
        try:
            total_interventions = len(self.engine.intervention_history)
            
            # Estatísticas por template
            template_usage = {}
            for record in self.engine.intervention_history:
                template = record.get('template_used')
                if template:
                    template_usage[template] = template_usage.get(template, 0) + 1
            
            # Estatísticas por urgência
            urgency_stats = {}
            for record in self.engine.intervention_history:
                urgency = record.get('urgency_level', 'normal')
                urgency_stats[urgency] = urgency_stats.get(urgency, 0) + 1
            
            return {
                'total_interventions': total_interventions,
                'template_usage': template_usage,
                'urgency_distribution': urgency_stats,
                'available_templates': len(self.engine.templates),
                'active_templates': len([t for t in self.engine.templates.values() if t.is_active])
            }
            
        except Exception as e:
            logger.error(f"Error getting intervention statistics: {e}")
            return {'error': str(e)}