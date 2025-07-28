"""
Motor de Machine Learning Adaptativo com treinamento contínuo
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
import structlog
import pickle
import json
import os
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline

import torch
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, 
    GPT2LMHeadModel, GPT2Tokenizer,
    pipeline
)

from app.core.config import settings
from app.core.circuit_breaker import ml_model_circuit_breaker
from app.core.metrics import track_ml_metrics

logger = structlog.get_logger(__name__)

class TrainingDataPoint:
    """Ponto de dados para treinamento"""
    
    def __init__(
        self,
        user_id: str,
        text: str,
        label: str,
        domain: str,
        confidence: float,
        metadata: Dict[str, Any],
        timestamp: datetime = None
    ):
        self.user_id = user_id
        self.text = text
        self.label = label
        self.domain = domain
        self.confidence = confidence
        self.metadata = metadata
        self.timestamp = timestamp or datetime.now()
        self.is_validated = "pending"

class AdaptiveMLEngine:
    """Motor de ML adaptativo com múltiplos modelos"""
    
    def __init__(self):
        self.models = {}
        self.vectorizers = {}
        self.scalers = {}
        self.label_encoders = {}
        self.training_data = []
        self.model_performance = {}
        self.last_training = {}
        
        self.models_dir = Path(settings.model_cache_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self._initialize_models()
    
    def _initialize_models(self):
        """Inicializa modelos para diferentes domínios"""
        try:
            domains = ['mathematics', 'language', 'science', 'general']
            
            for domain in domains:
                # Modelos tradicionais
                self.models[domain] = {
                    'traditional': self._create_traditional_model(),
                    'ensemble': self._create_ensemble_model(),
                    'deep_learning': None  # Será inicializado se disponível
                }
                
                # Vectorizers
                self.vectorizers[domain] = TfidfVectorizer(
                    max_features=10000,
                    ngram_range=(1, 2),
                    stop_words='english'
                )
                
                # Scalers
                self.scalers[domain] = StandardScaler()
                
                # Label encoders
                self.label_encoders[domain] = LabelEncoder()
                
                # Performance tracking
                self.model_performance[domain] = {
                    'accuracy': 0.0,
                    'last_updated': None,
                    'training_samples': 0
                }
                
                # Carregar modelos salvos se existirem
                self._load_saved_models(domain)
            
            # Inicializar modelos de deep learning se disponível
            if settings.ml_model_type in ['transformer', 'hybrid']:
                self._initialize_deep_learning_models()
            
            logger.info(f"Initialized ML models for {len(domains)} domains")
            
        except Exception as e:
            logger.error(f"Error initializing ML models: {e}")
    
    def _create_traditional_model(self):
        """Cria modelo tradicional (Random Forest)"""
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
    
    def _create_ensemble_model(self):
        """Cria modelo ensemble com múltiplos classificadores"""
        estimators = [
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
            ('lr', LogisticRegression(random_state=42, max_iter=1000)),
            ('nb', MultinomialNB()),
            ('svc', SVC(probability=True, random_state=42))
        ]
        
        return VotingClassifier(
            estimators=estimators,
            voting='soft'
        )
    
    def _initialize_deep_learning_models(self):
        """Inicializa modelos de deep learning"""
        try:
            # Verificar se CUDA está disponível
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            logger.info(f"Using device: {device}")
            
            # Modelos para diferentes domínios
            for domain in ['mathematics', 'language', 'science']:
                try:
                    # Modelo de classificação para o domínio
                    model_name = f"distilbert-base-multilingual-cased"
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                    model = AutoModelForSequenceClassification.from_pretrained(
                        model_name,
                        num_labels=5  # 5 níveis de dificuldade
                    )
                    
                    self.models[domain]['deep_learning'] = {
                        'model': model,
                        'tokenizer': tokenizer,
                        'device': device
                    }
                    
                    logger.info(f"Initialized deep learning model for {domain}")
                    
                except Exception as e:
                    logger.warning(f"Could not initialize deep learning model for {domain}: {e}")
            
            # Modelo de geração de texto
            try:
                self.gpt_model = GPT2LMHeadModel.from_pretrained('gpt2')
                self.gpt_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
                self.gpt_tokenizer.pad_token = self.gpt_tokenizer.eos_token
                
                logger.info("Initialized GPT-2 model for text generation")
                
            except Exception as e:
                logger.warning(f"Could not initialize GPT-2 model: {e}")
                
        except Exception as e:
            logger.error(f"Error initializing deep learning models: {e}")
    
    @ml_model_circuit_breaker
    @track_ml_metrics
    def predict(
        self, 
        text: str, 
        domain: str = 'general',
        model_type: str = 'ensemble'
    ) -> Dict[str, Any]:
        """
        Faz predição usando modelo específico do domínio
        """
        try:
            if domain not in self.models:
                domain = 'general'
            
            if model_type not in self.models[domain]:
                model_type = 'ensemble'
            
            model = self.models[domain][model_type]
            
            if model_type == 'deep_learning' and model is not None:
                return self._predict_deep_learning(text, domain, model)
            else:
                return self._predict_traditional(text, domain, model_type)
                
        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            return {
                'prediction': 'unknown',
                'confidence': 0.0,
                'model_type': model_type,
                'domain': domain,
                'error': str(e)
            }
    
    def _predict_traditional(
        self, 
        text: str, 
        domain: str, 
        model_type: str
    ) -> Dict[str, Any]:
        """Predição usando modelos tradicionais"""
        try:
            # Vectorizar texto
            vectorizer = self.vectorizers[domain]
            features = vectorizer.transform([text])
            
            # Fazer predição
            model = self.models[domain][model_type]
            prediction = model.predict(features)[0]
            probabilities = model.predict_proba(features)[0]
            
            # Calcular confiança
            confidence = np.max(probabilities)
            
            return {
                'prediction': prediction,
                'confidence': float(confidence),
                'probabilities': probabilities.tolist(),
                'model_type': model_type,
                'domain': domain
            }
            
        except Exception as e:
            logger.error(f"Error in traditional prediction: {e}")
            raise
    
    def _predict_deep_learning(
        self, 
        text: str, 
        domain: str, 
        model_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Predição usando modelos de deep learning"""
        try:
            model = model_data['model']
            tokenizer = model_data['tokenizer']
            device = model_data['device']
            
            # Tokenizar texto
            inputs = tokenizer(
                text,
                return_tensors='pt',
                truncation=True,
                max_length=512,
                padding=True
            ).to(device)
            
            # Fazer predição
            with torch.no_grad():
                outputs = model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=1)
                prediction = torch.argmax(probabilities, dim=1)
                confidence = torch.max(probabilities, dim=1)[0]
            
            return {
                'prediction': prediction.item(),
                'confidence': float(confidence.item()),
                'probabilities': probabilities[0].cpu().numpy().tolist(),
                'model_type': 'deep_learning',
                'domain': domain
            }
            
        except Exception as e:
            logger.error(f"Error in deep learning prediction: {e}")
            raise
    
    def add_training_data(
        self, 
        user_id: str, 
        text: str, 
        label: str, 
        domain: str,
        confidence: float = 1.0,
        metadata: Dict[str, Any] = None
    ):
        """Adiciona dados de treinamento"""
        try:
            data_point = TrainingDataPoint(
                user_id=user_id,
                text=text,
                label=label,
                domain=domain,
                confidence=confidence,
                metadata=metadata or {}
            )
            
            self.training_data.append(data_point)
            
            # Verificar se há dados suficientes para treinamento
            domain_data = [d for d in self.training_data if d.domain == domain]
            if len(domain_data) >= 50:  # Mínimo de 50 amostras
                self._schedule_training(domain)
            
            logger.info(f"Added training data for domain {domain}")
            
        except Exception as e:
            logger.error(f"Error adding training data: {e}")
    
    def _schedule_training(self, domain: str):
        """Agenda treinamento para um domínio"""
        try:
            # Verificar se já treinou recentemente
            last_training = self.last_training.get(domain)
            if last_training and (datetime.now() - last_training).hours < 24:
                return  # Treinar no máximo uma vez por dia
            
            # Executar treinamento
            self._train_models(domain)
            
        except Exception as e:
            logger.error(f"Error scheduling training: {e}")
    
    def _train_models(self, domain: str):
        """Treina modelos para um domínio específico"""
        try:
            # Filtrar dados do domínio
            domain_data = [d for d in self.training_data if d.domain == domain]
            
            if len(domain_data) < 20:
                logger.warning(f"Insufficient data for training {domain}: {len(domain_data)} samples")
                return
            
            # Preparar dados
            texts = [d.text for d in domain_data]
            labels = [d.label for d in domain_data]
            
            # Dividir em treino e teste
            X_train, X_test, y_train, y_test = train_test_split(
                texts, labels, test_size=0.2, random_state=42
            )
            
            # Treinar vectorizer
            vectorizer = self.vectorizers[domain]
            X_train_vectorized = vectorizer.fit_transform(X_train)
            X_test_vectorized = vectorizer.transform(X_test)
            
            # Treinar label encoder
            label_encoder = self.label_encoders[domain]
            y_train_encoded = label_encoder.fit_transform(y_train)
            y_test_encoded = label_encoder.transform(y_test)
            
            # Treinar modelos tradicionais
            for model_type in ['traditional', 'ensemble']:
                model = self.models[domain][model_type]
                
                # Treinar modelo
                model.fit(X_train_vectorized, y_train_encoded)
                
                # Avaliar performance
                y_pred = model.predict(X_test_vectorized)
                accuracy = accuracy_score(y_test_encoded, y_pred)
                
                # Atualizar performance
                self.model_performance[domain]['accuracy'] = accuracy
                self.model_performance[domain]['last_updated'] = datetime.now()
                self.model_performance[domain]['training_samples'] = len(domain_data)
                
                logger.info(f"Trained {model_type} model for {domain}: accuracy={accuracy:.3f}")
            
            # Salvar modelos
            self._save_models(domain)
            
            # Atualizar timestamp de treinamento
            self.last_training[domain] = datetime.now()
            
        except Exception as e:
            logger.error(f"Error training models for {domain}: {e}")
    
    def _save_models(self, domain: str):
        """Salva modelos treinados"""
        try:
            domain_dir = self.models_dir / domain
            domain_dir.mkdir(exist_ok=True)
            
            # Salvar modelos tradicionais
            for model_type in ['traditional', 'ensemble']:
                model = self.models[domain][model_type]
                model_path = domain_dir / f"{model_type}_model.pkl"
                
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
            
            # Salvar vectorizer
            vectorizer_path = domain_dir / "vectorizer.pkl"
            with open(vectorizer_path, 'wb') as f:
                pickle.dump(self.vectorizers[domain], f)
            
            # Salvar label encoder
            encoder_path = domain_dir / "label_encoder.pkl"
            with open(encoder_path, 'wb') as f:
                pickle.dump(self.label_encoders[domain], f)
            
            # Salvar performance
            performance_path = domain_dir / "performance.json"
            with open(performance_path, 'w') as f:
                json.dump(self.model_performance[domain], f, default=str)
            
            logger.info(f"Saved models for domain {domain}")
            
        except Exception as e:
            logger.error(f"Error saving models for {domain}: {e}")
    
    def _load_saved_models(self, domain: str):
        """Carrega modelos salvos"""
        try:
            domain_dir = self.models_dir / domain
            
            if not domain_dir.exists():
                return
            
            # Carregar modelos tradicionais
            for model_type in ['traditional', 'ensemble']:
                model_path = domain_dir / f"{model_type}_model.pkl"
                if model_path.exists():
                    with open(model_path, 'rb') as f:
                        self.models[domain][model_type] = pickle.load(f)
            
            # Carregar vectorizer
            vectorizer_path = domain_dir / "vectorizer.pkl"
            if vectorizer_path.exists():
                with open(vectorizer_path, 'rb') as f:
                    self.vectorizers[domain] = pickle.load(f)
            
            # Carregar label encoder
            encoder_path = domain_dir / "label_encoder.pkl"
            if encoder_path.exists():
                with open(encoder_path, 'rb') as f:
                    self.label_encoders[domain] = pickle.load(f)
            
            # Carregar performance
            performance_path = domain_dir / "performance.json"
            if performance_path.exists():
                with open(performance_path, 'r') as f:
                    self.model_performance[domain] = json.load(f)
            
            logger.info(f"Loaded saved models for domain {domain}")
            
        except Exception as e:
            logger.error(f"Error loading saved models for {domain}: {e}")
    
    def generate_text(
        self, 
        prompt: str, 
        max_length: int = 100,
        temperature: float = 0.8
    ) -> str:
        """Gera texto usando modelo de linguagem"""
        try:
            if not hasattr(self, 'gpt_model') or not hasattr(self, 'gpt_tokenizer'):
                return "Modelo de geração de texto não disponível."
            
            # Tokenizar prompt
            inputs = self.gpt_tokenizer.encode(prompt, return_tensors='pt')
            
            # Gerar texto
            with torch.no_grad():
                outputs = self.gpt_model.generate(
                    inputs,
                    max_length=max_length,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.gpt_tokenizer.eos_token_id
                )
            
            # Decodificar resposta
            generated_text = self.gpt_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            return generated_text
            
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            return "Erro na geração de texto."
    
    def get_model_performance(self, domain: str = None) -> Dict[str, Any]:
        """Obtém performance dos modelos"""
        try:
            if domain:
                return self.model_performance.get(domain, {})
            else:
                return self.model_performance
                
        except Exception as e:
            logger.error(f"Error getting model performance: {e}")
            return {}
    
    def validate_training_data(self, user_id: str, data_id: int, is_correct: bool):
        """Valida dados de treinamento"""
        try:
            if data_id < len(self.training_data):
                data_point = self.training_data[data_id]
                if data_point.user_id == user_id:
                    data_point.is_validated = "correct" if is_correct else "incorrect"
                    logger.info(f"Validated training data {data_id} as {'correct' if is_correct else 'incorrect'}")
                    
        except Exception as e:
            logger.error(f"Error validating training data: {e}")
    
    def get_training_statistics(self) -> Dict[str, Any]:
        """Obtém estatísticas dos dados de treinamento"""
        try:
            stats = {
                'total_samples': len(self.training_data),
                'domains': {},
                'validation_status': {},
                'recent_additions': []
            }
            
            # Estatísticas por domínio
            for data_point in self.training_data:
                domain = data_point.domain
                if domain not in stats['domains']:
                    stats['domains'][domain] = 0
                stats['domains'][domain] += 1
            
            # Status de validação
            validation_counts = {}
            for data_point in self.training_data:
                status = data_point.is_validated
                if status not in validation_counts:
                    validation_counts[status] = 0
                validation_counts[status] += 1
            stats['validation_status'] = validation_counts
            
            # Adições recentes
            recent_data = sorted(self.training_data, key=lambda x: x.timestamp, reverse=True)
            stats['recent_additions'] = [
                {
                    'id': i,
                    'domain': d.domain,
                    'label': d.label,
                    'timestamp': d.timestamp.isoformat(),
                    'validated': d.is_validated
                }
                for i, d in enumerate(recent_data[:10])
            ]
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting training statistics: {e}")
            return {'error': str(e)}

class MLService:
    """Serviço principal de ML"""
    
    def __init__(self):
        self.engine = AdaptiveMLEngine()
    
    @ml_model_circuit_breaker
    async def predict_difficulty(
        self, 
        text: str, 
        domain: str = 'general',
        model_type: str = 'ensemble'
    ) -> Dict[str, Any]:
        """Prediz dificuldade de um texto"""
        try:
            result = self.engine.predict(text, domain, model_type)
            
            # Mapear predição para nível de dificuldade
            difficulty_mapping = {
                'beginner': 1.0,
                'elementary': 2.0,
                'intermediate': 3.0,
                'advanced': 4.0,
                'expert': 5.0
            }
            
            difficulty_level = difficulty_mapping.get(result['prediction'], 3.0)
            
            return {
                'text': text,
                'domain': domain,
                'difficulty_level': difficulty_level,
                'confidence': result['confidence'],
                'model_type': result['model_type'],
                'prediction': result['prediction']
            }
            
        except Exception as e:
            logger.error(f"Error predicting difficulty: {e}")
            return {
                'text': text,
                'domain': domain,
                'difficulty_level': 3.0,
                'confidence': 0.0,
                'error': str(e)
            }
    
    async def add_training_example(
        self, 
        user_id: str, 
        text: str, 
        difficulty_label: str, 
        domain: str,
        confidence: float = 1.0,
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Adiciona exemplo de treinamento"""
        try:
            self.engine.add_training_data(
                user_id, text, difficulty_label, domain, confidence, metadata
            )
            
            return {
                'status': 'success',
                'message': 'Exemplo de treinamento adicionado com sucesso'
            }
            
        except Exception as e:
            logger.error(f"Error adding training example: {e}")
            return {
                'status': 'error',
                'message': f'Erro ao adicionar exemplo: {str(e)}'
            }
    
    async def generate_educational_content(
        self, 
        topic: str, 
        difficulty_level: float,
        content_type: str = 'explanation'
    ) -> Dict[str, Any]:
        """Gera conteúdo educacional"""
        try:
            # Criar prompt baseado no tipo de conteúdo
            if content_type == 'explanation':
                prompt = f"Explique {topic} de forma clara e didática, adequada para nível {difficulty_level}:"
            elif content_type == 'exercise':
                prompt = f"Crie um exercício sobre {topic} com dificuldade {difficulty_level}:"
            elif content_type == 'example':
                prompt = f"Dê um exemplo prático de {topic} para nível {difficulty_level}:"
            else:
                prompt = f"Escreva sobre {topic} para nível {difficulty_level}:"
            
            # Gerar texto
            generated_text = self.engine.generate_text(prompt, max_length=200)
            
            return {
                'topic': topic,
                'difficulty_level': difficulty_level,
                'content_type': content_type,
                'generated_content': generated_text,
                'prompt_used': prompt
            }
            
        except Exception as e:
            logger.error(f"Error generating educational content: {e}")
            return {
                'topic': topic,
                'difficulty_level': difficulty_level,
                'content_type': content_type,
                'error': str(e)
            }
    
    async def get_ml_statistics(self) -> Dict[str, Any]:
        """Obtém estatísticas do sistema de ML"""
        try:
            performance = self.engine.get_model_performance()
            training_stats = self.engine.get_training_statistics()
            
            return {
                'model_performance': performance,
                'training_statistics': training_stats,
                'total_models': len(self.engine.models),
                'available_domains': list(self.engine.models.keys())
            }
            
        except Exception as e:
            logger.error(f"Error getting ML statistics: {e}")
            return {'error': str(e)}