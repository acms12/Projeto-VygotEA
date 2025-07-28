"""
Serviço de gamificação avançado com eventos dinâmicos e badges adaptativos
"""
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import structlog
import json
import re

from app.models.gamification import (
    UserGamificationProfile, GamificationEvent, EventType, Badge,
    DynamicBadgeRule, GamificationConfig, LeaderboardEntry, Achievement,
    BadgeType
)
from app.core.circuit_breaker import database_circuit_breaker
from app.core.metrics import track_gamification_metrics, update_user_xp

logger = structlog.get_logger(__name__)

class AdaptiveGamificationEngine:
    """Motor de gamificação adaptativo"""
    
    def __init__(self, config: GamificationConfig):
        self.config = config
        self.badge_rules = []
        self.achievements = {}
        self.user_profiles = {}
        
    def calculate_xp_earned(
        self, 
        event_type: EventType, 
        base_xp: int,
        user_profile: UserGamificationProfile,
        event_metadata: Dict[str, Any]
    ) -> int:
        """
        Calcula XP ganho com base em múltiplos fatores adaptativos
        """
        try:
            # XP base
            xp = base_xp
            
            # Multiplicador de dificuldade
            difficulty_multiplier = self._calculate_difficulty_multiplier(event_metadata)
            xp *= difficulty_multiplier
            
            # Bônus de streak
            if user_profile.current_streak > 0:
                streak_bonus = min(
                    user_profile.current_streak * self.config.streak_bonus,
                    self.config.max_streak_bonus * self.config.streak_bonus
                )
                xp += int(xp * streak_bonus)
            
            # Bônus de nível
            level_bonus = (user_profile.current_level - 1) * self.config.level_bonus
            xp += int(xp * level_bonus)
            
            # Bônus específicos por tipo de evento
            event_bonus = self._calculate_event_bonus(event_type, event_metadata)
            xp += event_bonus
            
            # Multiplicador global
            xp = int(xp * self.config.xp_multiplier)
            
            # Garantir XP mínimo
            return max(xp, 1)
            
        except Exception as e:
            logger.error(f"Error calculating XP: {e}")
            return base_xp
    
    def check_new_badges(
        self, 
        user_profile: UserGamificationProfile,
        event: GamificationEvent
    ) -> List[Badge]:
        """
        Verifica novos badges baseado em regras dinâmicas
        """
        try:
            new_badges = []
            
            # Verificar badges estáticos
            static_badges = self._check_static_badges(user_profile, event)
            new_badges.extend(static_badges)
            
            # Verificar badges dinâmicos
            if self.config.dynamic_badges:
                dynamic_badges = self._check_dynamic_badges(user_profile, event)
                new_badges.extend(dynamic_badges)
            
            # Verificar achievements
            achievement_badges = self._check_achievements(user_profile, event)
            new_badges.extend(achievement_badges)
            
            return new_badges
            
        except Exception as e:
            logger.error(f"Error checking badges: {e}")
            return []
    
    def update_streak(
        self, 
        user_profile: UserGamificationProfile,
        event: GamificationEvent
    ) -> bool:
        """
        Atualiza streak do usuário com lógica adaptativa
        """
        try:
            if event.event_type == EventType.LOGIN:
                return self._update_login_streak(user_profile, event)
            elif event.event_type == EventType.COMPLETE_ACTIVITY:
                return self._update_activity_streak(user_profile, event)
            else:
                return False
                
        except Exception as e:
            logger.error(f"Error updating streak: {e}")
            return False
    
    def generate_personalized_rewards(
        self, 
        user_profile: UserGamificationProfile,
        event: GamificationEvent
    ) -> Dict[str, Any]:
        """
        Gera recompensas personalizadas baseadas no perfil do usuário
        """
        try:
            rewards = {
                'xp_boost': 1.0,
                'special_badges': [],
                'bonus_activities': [],
                'unlock_features': []
            }
            
            # Análise de padrões de comportamento
            behavior_patterns = self._analyze_behavior_patterns(user_profile)
            
            # Ajustar recompensas baseado nos padrões
            if behavior_patterns['consistency'] > 0.8:
                rewards['xp_boost'] = 1.2
                rewards['special_badges'].append('consistency_master')
            
            if behavior_patterns['exploration'] > 0.7:
                rewards['bonus_activities'].append('exploration_challenge')
            
            if behavior_patterns['helping_others'] > 0.6:
                rewards['unlock_features'].append('mentor_mode')
            
            return rewards
            
        except Exception as e:
            logger.error(f"Error generating personalized rewards: {e}")
            return {'xp_boost': 1.0, 'special_badges': [], 'bonus_activities': [], 'unlock_features': []}
    
    def _calculate_difficulty_multiplier(self, event_metadata: Dict[str, Any]) -> float:
        """Calcula multiplicador baseado na dificuldade"""
        difficulty = event_metadata.get('difficulty', 5.0)
        score = event_metadata.get('score', 70)
        time_spent = event_metadata.get('time_spent', 300)
        
        # Multiplicador baseado na dificuldade
        difficulty_factor = 1.0 + (difficulty - 5.0) * 0.1
        
        # Bônus por pontuação alta
        if score >= 90:
            difficulty_factor += 0.2
        elif score >= 80:
            difficulty_factor += 0.1
        
        # Bônus por conclusão rápida (se aplicável)
        expected_time = event_metadata.get('expected_time', 300)
        if time_spent < expected_time * 0.8:
            difficulty_factor += 0.1
        
        return max(difficulty_factor, 0.5)
    
    def _calculate_event_bonus(self, event_type: EventType, metadata: Dict[str, Any]) -> int:
        """Calcula bônus específico por tipo de evento"""
        base_bonus = {
            EventType.PERFECT_SCORE: 50,
            EventType.FAST_COMPLETION: 30,
            EventType.HELP_OTHERS: 40,
            EventType.CONSISTENCY: 25,
            EventType.EXPLORATION: 35
        }
        
        bonus = base_bonus.get(event_type, 0)
        
        # Ajustar baseado em metadados específicos
        if event_type == EventType.PERFECT_SCORE and metadata.get('score', 0) == 100:
            bonus += 25
        
        if event_type == EventType.FAST_COMPLETION:
            time_saved = metadata.get('time_saved', 0)
            bonus += int(time_saved / 60) * 5  # 5 XP por minuto economizado
        
        return bonus
    
    def _check_static_badges(
        self, 
        user_profile: UserGamificationProfile,
        event: GamificationEvent
    ) -> List[Badge]:
        """Verifica badges estáticos baseados em limiares fixos"""
        new_badges = []
        
        # Badges baseados em XP total
        xp_thresholds = {
            BadgeType.BEGINNER: 100,
            BadgeType.INTERMEDIATE: 500,
            BadgeType.ADVANCED: 1000,
            BadgeType.EXPERT: 2000,
            BadgeType.MASTER: 5000,
            BadgeType.LEGENDARY: 10000
        }
        
        for badge_type, threshold in xp_thresholds.items():
            if (user_profile.total_xp >= threshold and 
                not any(badge.badge_type == badge_type for badge in user_profile.badges_earned)):
                new_badges.append(Badge(
                    id=f"{badge_type.value}_badge",
                    name=f"{badge_type.value.title()} Badge",
                    description=f"Reached {threshold} XP",
                    badge_type=badge_type,
                    xp_reward=threshold // 10,
                    rarity=0.1 if badge_type in [BadgeType.MASTER, BadgeType.LEGENDARY] else 0.3
                ))
        
        # Badges baseados em streak
        streak_thresholds = [7, 30, 100, 365]
        for threshold in streak_thresholds:
            if (user_profile.current_streak >= threshold and 
                not any(f"streak_{threshold}" in badge.id for badge in user_profile.badges_earned)):
                new_badges.append(Badge(
                    id=f"streak_{threshold}",
                    name=f"{threshold} Day Streak",
                    description=f"Maintained a {threshold}-day streak",
                    badge_type=BadgeType.CUSTOM,
                    xp_reward=threshold * 10,
                    rarity=0.2
                ))
        
        return new_badges
    
    def _check_dynamic_badges(
        self, 
        user_profile: UserGamificationProfile,
        event: GamificationEvent
    ) -> List[Badge]:
        """Verifica badges dinâmicos baseados em regras personalizadas"""
        new_badges = []
        
        # Regras dinâmicas baseadas em padrões
        rules = [
            {
                'condition': lambda p: p.total_activities_completed >= 50 and p.perfect_scores >= 10,
                'badge': Badge(
                    id="quality_learner",
                    name="Quality Learner",
                    description="Completed 50+ activities with 10+ perfect scores",
                    badge_type=BadgeType.SPECIALIST,
                    xp_reward=200,
                    rarity=0.15
                )
            },
            {
                'condition': lambda p: p.help_others_count >= 5,
                'badge': Badge(
                    id="helpful_mentor",
                    name="Helpful Mentor",
                    description="Helped 5+ other learners",
                    badge_type=BadgeType.CUSTOM,
                    xp_reward=150,
                    rarity=0.2
                )
            },
            {
                'condition': lambda p: p.current_streak >= 14 and p.total_xp >= 1000,
                'badge': Badge(
                    id="dedicated_learner",
                    name="Dedicated Learner",
                    description="2+ week streak with 1000+ XP",
                    badge_type=BadgeType.ADVANCED,
                    xp_reward=300,
                    rarity=0.1
                )
            }
        ]
        
        for rule in rules:
            if rule['condition'](user_profile):
                badge = rule['badge']
                if not any(b.id == badge.id for b in user_profile.badges_earned):
                    new_badges.append(badge)
        
        return new_badges
    
    def _check_achievements(
        self, 
        user_profile: UserGamificationProfile,
        event: GamificationEvent
    ) -> List[Badge]:
        """Verifica achievements e retorna badges correspondentes"""
        new_badges = []
        
        # Atualizar progresso dos achievements
        for achievement_id, achievement in self.achievements.items():
            if achievement_id not in user_profile.badges_earned:
                # Atualizar progresso baseado no evento
                progress_value = self._calculate_achievement_progress(achievement, event)
                achievement.update_progress(progress_value)
                
                # Se completou, criar badge
                if achievement.is_completed:
                    new_badges.append(Badge(
                        id=f"achievement_{achievement_id}",
                        name=achievement.name,
                        description=achievement.description,
                        badge_type=BadgeType.CUSTOM,
                        xp_reward=achievement.xp_reward,
                        rarity=0.3
                    ))
        
        return new_badges
    
    def _calculate_achievement_progress(self, achievement: Achievement, event: GamificationEvent) -> int:
        """Calcula progresso para um achievement baseado no evento"""
        if achievement.category == "activities" and event.event_type == EventType.COMPLETE_ACTIVITY:
            return 1
        elif achievement.category == "perfect_scores" and event.event_type == EventType.PERFECT_SCORE:
            return 1
        elif achievement.category == "helping_others" and event.event_type == EventType.HELP_OTHERS:
            return 1
        
        return 0
    
    def _update_login_streak(self, user_profile: UserGamificationProfile, event: GamificationEvent) -> bool:
        """Atualiza streak de login"""
        if not user_profile.last_activity:
            user_profile.current_streak = 1
            return True
        
        days_since_last = (event.timestamp.date() - user_profile.last_activity.date()).days
        
        if days_since_last == 0:
            # Mesmo dia, não atualizar streak
            return False
        elif days_since_last == 1:
            # Dia consecutivo
            user_profile.current_streak += 1
            if user_profile.current_streak > user_profile.longest_streak:
                user_profile.longest_streak = user_profile.current_streak
            return True
        else:
            # Quebrou o streak
            user_profile.current_streak = 1
            return True
    
    def _update_activity_streak(self, user_profile: UserGamificationProfile, event: GamificationEvent) -> bool:
        """Atualiza streak de atividades"""
        # Implementação similar ao login streak, mas para atividades
        return self._update_login_streak(user_profile, event)
    
    def _analyze_behavior_patterns(self, user_profile: UserGamificationProfile) -> Dict[str, float]:
        """Analisa padrões de comportamento do usuário"""
        patterns = {
            'consistency': 0.0,
            'exploration': 0.0,
            'helping_others': 0.0,
            'speed': 0.0,
            'quality': 0.0
        }
        
        if not user_profile.recent_events:
            return patterns
        
        # Análise de consistência (frequência de atividades)
        activity_dates = [e.timestamp.date() for e in user_profile.recent_events if e.event_type == EventType.COMPLETE_ACTIVITY]
        if activity_dates:
            unique_dates = len(set(activity_dates))
            total_days = (max(activity_dates) - min(activity_dates)).days + 1
            patterns['consistency'] = unique_dates / total_days if total_days > 0 else 0.0
        
        # Análise de exploração (diversidade de atividades)
        activity_types = [e.metadata.get('activity_type', 'unknown') for e in user_profile.recent_events if e.event_type == EventType.COMPLETE_ACTIVITY]
        if activity_types:
            unique_types = len(set(activity_types))
            patterns['exploration'] = min(unique_types / 10.0, 1.0)  # Normalizar
        
        # Análise de ajuda aos outros
        help_events = [e for e in user_profile.recent_events if e.event_type == EventType.HELP_OTHERS]
        patterns['helping_others'] = min(len(help_events) / 5.0, 1.0)
        
        # Análise de velocidade
        fast_completion_events = [e for e in user_profile.recent_events if e.event_type == EventType.FAST_COMPLETION]
        patterns['speed'] = min(len(fast_completion_events) / 3.0, 1.0)
        
        # Análise de qualidade
        perfect_score_events = [e for e in user_profile.recent_events if e.event_type == EventType.PERFECT_SCORE]
        patterns['quality'] = min(len(perfect_score_events) / 5.0, 1.0)
        
        return patterns

class GamificationService:
    """Serviço principal de gamificação"""
    
    def __init__(self):
        self.config = GamificationConfig()
        self.engine = AdaptiveGamificationEngine(self.config)
        self._initialize_achievements()
    
    @database_circuit_breaker
    @track_gamification_metrics
    async def process_event(
        self, 
        user_id: str, 
        event_type: EventType,
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Processa evento de gamificação e retorna resultados
        """
        try:
            # Obter ou criar perfil do usuário
            user_profile = await self._get_user_profile(user_id)
            
            # Calcular XP ganho
            base_xp = self._get_base_xp(event_type, metadata or {})
            xp_earned = self.engine.calculate_xp_earned(
                event_type, base_xp, user_profile, metadata or {}
            )
            
            # Criar evento
            event = GamificationEvent(
                user_id=user_id,
                event_type=event_type,
                xp_earned=xp_earned,
                metadata=metadata or {}
            )
            
            # Atualizar perfil
            user_profile.add_event(event)
            
            # Verificar novos badges
            new_badges = self.engine.check_new_badges(user_profile, event)
            
            # Atualizar streak
            streak_updated = self.engine.update_streak(user_profile, event)
            
            # Gerar recompensas personalizadas
            personalized_rewards = self.engine.generate_personalized_rewards(user_profile, event)
            
            # Salvar perfil atualizado
            await self._save_user_profile(user_profile)
            
            # Atualizar métricas
            update_user_xp(user_id, user_profile.total_xp)
            
            return {
                'xp_earned': xp_earned,
                'new_badges': [badge.dict() for badge in new_badges],
                'streak_updated': streak_updated,
                'personalized_rewards': personalized_rewards,
                'current_level': user_profile.current_level,
                'level_progress': user_profile.level_progress,
                'current_streak': user_profile.current_streak
            }
            
        except Exception as e:
            logger.error(f"Error processing gamification event: {e}")
            raise
    
    async def get_user_profile(self, user_id: str) -> UserGamificationProfile:
        """Obtém perfil de gamificação do usuário"""
        return await self._get_user_profile(user_id)
    
    async def get_leaderboard(self, limit: int = 10) -> List[LeaderboardEntry]:
        """Obtém leaderboard dos usuários"""
        try:
            # Implementação seria baseada em consulta ao banco de dados
            # Por enquanto, retorna dados mock
            return [
                LeaderboardEntry(
                    user_id=f"user_{i}",
                    username=f"User {i}",
                    total_xp=10000 - i * 500,
                    level=15 - i,
                    rank=i + 1,
                    badges_count=8 - i,
                    streak=12 - i
                )
                for i in range(limit)
            ]
        except Exception as e:
            logger.error(f"Error getting leaderboard: {e}")
            return []
    
    def _get_base_xp(self, event_type: EventType, metadata: Dict[str, Any]) -> int:
        """Obtém XP base para tipo de evento"""
        base_xp_map = {
            EventType.LOGIN: 10,
            EventType.COMPLETE_ACTIVITY: 50,
            EventType.EARN_BADGE: 0,  # XP já incluído no badge
            EventType.ACHIEVE_STREAK: 25,
            EventType.LEVEL_UP: 100,
            EventType.HELP_OTHERS: 30,
            EventType.PERFECT_SCORE: 75,
            EventType.FAST_COMPLETION: 40,
            EventType.CONSISTENCY: 20,
            EventType.EXPLORATION: 35
        }
        
        base_xp = base_xp_map.get(event_type, 10)
        
        # Ajustar baseado em metadados
        if event_type == EventType.COMPLETE_ACTIVITY:
            score = metadata.get('score', 70)
            if score >= 90:
                base_xp += 25
            elif score >= 80:
                base_xp += 15
        
        return base_xp
    
    async def _get_user_profile(self, user_id: str) -> UserGamificationProfile:
        """Obtém perfil do usuário do banco de dados"""
        # Implementação seria baseada em consulta ao banco
        # Por enquanto, retorna perfil mock
        return UserGamificationProfile(
            user_id=user_id,
            total_xp=1000,
            current_level=5,
            current_streak=3,
            longest_streak=7,
            badges_earned=[],
            recent_events=[],
            total_activities_completed=25,
            perfect_scores=3,
            help_others_count=1
        )
    
    async def _save_user_profile(self, profile: UserGamificationProfile):
        """Salva perfil do usuário no banco de dados"""
        # Implementação seria baseada em persistência no banco
        logger.info(f"Saving profile for user {profile.user_id}")
    
    def _initialize_achievements(self):
        """Inicializa achievements do sistema"""
        self.achievements = {
            'math_master': Achievement(
                id='math_master',
                name='Math Master',
                description='Complete 100 mathematics activities',
                category='activities',
                target_value=100,
                xp_reward=500
            ),
            'perfect_learner': Achievement(
                id='perfect_learner',
                name='Perfect Learner',
                description='Get 20 perfect scores',
                category='perfect_scores',
                target_value=20,
                xp_reward=300
            ),
            'helpful_community': Achievement(
                id='helpful_community',
                name='Helpful Community Member',
                description='Help 10 other learners',
                category='helping_others',
                target_value=10,
                xp_reward=400
            )
        }