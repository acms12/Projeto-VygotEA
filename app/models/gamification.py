"""
Modelos de dados para sistema de gamificação avançado
"""
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, validator
from enum import Enum
from datetime import datetime, timedelta
import structlog

logger = structlog.get_logger(__name__)

class BadgeType(str, Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"
    SPECIALIST = "specialist"
    MASTER = "master"
    LEGENDARY = "legendary"
    CUSTOM = "custom"

class EventType(str, Enum):
    LOGIN = "login"
    COMPLETE_ACTIVITY = "complete_activity"
    EARN_BADGE = "earn_badge"
    ACHIEVE_STREAK = "achieve_streak"
    LEVEL_UP = "level_up"
    HELP_OTHERS = "help_others"
    PERFECT_SCORE = "perfect_score"
    FAST_COMPLETION = "fast_completion"
    CONSISTENCY = "consistency"
    EXPLORATION = "exploration"

class Badge(BaseModel):
    """Badge/Conquista do sistema"""
    id: str
    name: str
    description: str
    badge_type: BadgeType
    icon_url: Optional[str] = None
    xp_reward: int = Field(..., ge=0)
    requirements: Dict[str, Any] = {}
    rarity: float = Field(..., ge=0, le=1)  # 0 = comum, 1 = raro
    is_hidden: bool = False
    category: str = "general"
    
    class Config:
        schema_extra = {
            "example": {
                "id": "math_master",
                "name": "Mestre da Matemática",
                "description": "Completou 100 exercícios de matemática",
                "badge_type": "master",
                "xp_reward": 500,
                "requirements": {"math_activities": 100},
                "rarity": 0.1,
                "category": "mathematics"
            }
        }

class GamificationEvent(BaseModel):
    """Evento de gamificação"""
    user_id: str
    event_type: EventType
    timestamp: datetime = Field(default_factory=datetime.now)
    xp_earned: int = Field(0, ge=0)
    metadata: Dict[str, Any] = {}
    session_id: Optional[str] = None
    
    class Config:
        schema_extra = {
            "example": {
                "user_id": "user123",
                "event_type": "complete_activity",
                "xp_earned": 50,
                "metadata": {
                    "activity_id": "math_001",
                    "score": 85,
                    "time_spent": 300
                }
            }
        }

class UserGamificationProfile(BaseModel):
    """Perfil de gamificação do usuário"""
    user_id: str
    total_xp: int = Field(0, ge=0)
    current_level: int = Field(1, ge=1)
    current_streak: int = Field(0, ge=0)
    longest_streak: int = Field(0, ge=0)
    badges_earned: List[str] = []
    recent_events: List[GamificationEvent] = []
    last_activity: Optional[datetime] = None
    total_activities_completed: int = Field(0, ge=0)
    perfect_scores: int = Field(0, ge=0)
    help_others_count: int = Field(0, ge=0)
    
    @property
    def xp_to_next_level(self) -> int:
        """XP necessário para o próximo nível"""
        return self._calculate_xp_for_level(self.current_level + 1) - self.total_xp
    
    @property
    def level_progress(self) -> float:
        """Progresso no nível atual (0-1)"""
        current_level_xp = self._calculate_xp_for_level(self.current_level)
        next_level_xp = self._calculate_xp_for_level(self.current_level + 1)
        level_xp_range = next_level_xp - current_level_xp
        user_xp_in_level = self.total_xp - current_level_xp
        return min(user_xp_in_level / level_xp_range, 1.0)
    
    @property
    def is_streak_active(self) -> bool:
        """Verifica se o streak está ativo (última atividade foi hoje)"""
        if not self.last_activity:
            return False
        return self.last_activity.date() == datetime.now().date()
    
    def _calculate_xp_for_level(self, level: int) -> int:
        """Calcula XP necessário para um nível específico"""
        # Fórmula exponencial para crescimento de XP
        return int(100 * (level ** 1.5))
    
    def add_event(self, event: GamificationEvent):
        """Adiciona evento e atualiza perfil"""
        self.recent_events.append(event)
        self.total_xp += event.xp_earned
        self.last_activity = event.timestamp
        
        # Manter apenas os últimos 50 eventos
        if len(self.recent_events) > 50:
            self.recent_events = self.recent_events[-50:]
        
        # Verificar level up
        self._check_level_up()
        
        # Verificar badges
        self._check_badges(event)
        
        # Atualizar streak
        self._update_streak(event)
    
    def _check_level_up(self):
        """Verifica se o usuário subiu de nível"""
        while self.total_xp >= self._calculate_xp_for_level(self.current_level + 1):
            self.current_level += 1
            logger.info(f"User {self.user_id} reached level {self.current_level}")
    
    def _check_badges(self, event: GamificationEvent):
        """Verifica se o usuário ganhou novos badges"""
        # Implementação seria baseada em regras dinâmicas
        pass
    
    def _update_streak(self, event: GamificationEvent):
        """Atualiza streak do usuário"""
        if event.event_type == EventType.LOGIN:
            if self.is_streak_active:
                self.current_streak += 1
                if self.current_streak > self.longest_streak:
                    self.longest_streak = self.current_streak
            else:
                # Verificar se foi ontem para manter streak
                if (self.last_activity and 
                    (event.timestamp.date() - self.last_activity.date()).days == 1):
                    self.current_streak += 1
                else:
                    self.current_streak = 1

class DynamicBadgeRule(BaseModel):
    """Regra dinâmica para badges"""
    rule_id: str
    name: str
    condition: str  # Expressão lógica
    badge_id: str
    priority: int = Field(0, ge=0)
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.now)
    
    class Config:
        schema_extra = {
            "example": {
                "rule_id": "fast_learner",
                "name": "Aprendiz Rápido",
                "condition": "activities_completed >= 10 and avg_score >= 90",
                "badge_id": "fast_learner_badge",
                "priority": 1
            }
        }

class GamificationConfig(BaseModel):
    """Configuração do sistema de gamificação"""
    xp_multiplier: float = Field(1.0, ge=0.1, le=5.0)
    streak_bonus: float = Field(0.1, ge=0, le=1.0)
    level_bonus: float = Field(0.05, ge=0, le=1.0)
    perfect_score_bonus: float = Field(0.2, ge=0, le=1.0)
    help_others_bonus: float = Field(0.15, ge=0, le=1.0)
    max_streak_bonus: int = Field(7, ge=1, le=30)
    
    # Configurações dinâmicas
    adaptive_difficulty: bool = True
    dynamic_badges: bool = True
    personalized_rewards: bool = True
    
    class Config:
        schema_extra = {
            "example": {
                "xp_multiplier": 1.2,
                "streak_bonus": 0.15,
                "level_bonus": 0.1,
                "perfect_score_bonus": 0.25,
                "help_others_bonus": 0.2,
                "max_streak_bonus": 10
            }
        }

class LeaderboardEntry(BaseModel):
    """Entrada no leaderboard"""
    user_id: str
    username: str
    total_xp: int
    level: int
    rank: int
    badges_count: int
    streak: int
    
    class Config:
        schema_extra = {
            "example": {
                "user_id": "user123",
                "username": "João Silva",
                "total_xp": 15000,
                "level": 15,
                "rank": 3,
                "badges_count": 8,
                "streak": 12
            }
        }

class Achievement(BaseModel):
    """Conquista especial com progresso"""
    id: str
    name: str
    description: str
    category: str
    progress: float = Field(0, ge=0, le=1)
    target_value: int
    current_value: int = Field(0, ge=0)
    is_completed: bool = False
    completed_at: Optional[datetime] = None
    xp_reward: int = Field(0, ge=0)
    
    def update_progress(self, value: int):
        """Atualiza progresso da conquista"""
        self.current_value = min(value, self.target_value)
        self.progress = self.current_value / self.target_value
        
        if self.progress >= 1.0 and not self.is_completed:
            self.is_completed = True
            self.completed_at = datetime.now()
    
    class Config:
        schema_extra = {
            "example": {
                "id": "math_100",
                "name": "Centenário da Matemática",
                "description": "Complete 100 exercícios de matemática",
                "category": "mathematics",
                "target_value": 100,
                "current_value": 75,
                "progress": 0.75,
                "xp_reward": 1000
            }
        }