"""
Sistema RAG (Retrieval-Augmented Generation) avançado
"""
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import structlog
import json
import os
from pathlib import Path
import hashlib

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import faiss

from app.core.config import settings
from app.core.circuit_breaker import external_api_circuit_breaker
from app.core.metrics import track_ml_metrics

logger = structlog.get_logger(__name__)

class KnowledgeDocument:
    """Documento da base de conhecimento"""
    
    def __init__(
        self,
        id: str,
        title: str,
        content: str,
        domain: str,
        difficulty_level: float,
        tags: List[str],
        metadata: Dict[str, Any]
    ):
        self.id = id
        self.title = title
        self.content = content
        self.domain = domain
        self.difficulty_level = difficulty_level
        self.tags = tags
        self.metadata = metadata
        self.embedding = None
        self.created_at = datetime.now()
        self.last_accessed = None
        self.access_count = 0

class AdvancedRAGSystem:
    """Sistema RAG avançado com múltiplas estratégias de recuperação"""
    
    def __init__(self):
        self.embedding_model = None
        self.tfidf_vectorizer = TfidfVectorizer(max_features=10000)
        self.faiss_index = None
        self.documents = {}
        self.domain_indexes = {}
        self.knowledge_base_path = Path(settings.knowledge_base_path)
        
        self._initialize_models()
        self._populate_knowledge_base()
    
    def _initialize_models(self):
        """Inicializa modelos de embedding e indexação"""
        try:
            # Modelo de embedding para português
            self.embedding_model = SentenceTransformer(settings.embedding_model)
            logger.info(f"Initialized embedding model: {settings.embedding_model}")
        except Exception as e:
            logger.error(f"Error initializing embedding model: {e}")
            # Fallback para modelo mais simples
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def _populate_knowledge_base(self):
        """Popula base de conhecimento com documentos diversificados"""
        try:
            # Criar diretório se não existir
            self.knowledge_base_path.mkdir(parents=True, exist_ok=True)
            
            # Documentos de matemática
            math_docs = [
                {
                    'id': 'math_algebra_basics',
                    'title': 'Fundamentos da Álgebra',
                    'content': '''
                    A álgebra é uma ramificação da matemática que utiliza símbolos e letras para representar números e quantidades desconhecidas. Os conceitos fundamentais incluem:
                    
                    1. Variáveis: Letras que representam valores desconhecidos (x, y, z)
                    2. Expressões algébricas: Combinações de variáveis, números e operações
                    3. Equações: Igualdades que contêm variáveis
                    4. Inequações: Desigualdades que contêm variáveis
                    
                    Exemplos práticos:
                    - 2x + 3 = 7 (equação linear)
                    - x² - 4 = 0 (equação quadrática)
                    - 3x + 2y = 10 (sistema de equações)
                    ''',
                    'domain': 'mathematics',
                    'difficulty_level': 4.0,
                    'tags': ['álgebra', 'equações', 'variáveis', 'fundamentos']
                },
                {
                    'id': 'math_geometry_concepts',
                    'title': 'Conceitos de Geometria',
                    'content': '''
                    A geometria estuda as propriedades e relações entre pontos, linhas, superfícies e sólidos. Principais conceitos:
                    
                    1. Figuras planas: triângulos, quadriláteros, círculos
                    2. Áreas e perímetros: fórmulas para cálculo
                    3. Teorema de Pitágoras: a² + b² = c²
                    4. Volume e área superficial de sólidos
                    
                    Aplicações práticas:
                    - Arquitetura e construção
                    - Design e arte
                    - Engenharia e tecnologia
                    ''',
                    'domain': 'mathematics',
                    'difficulty_level': 5.0,
                    'tags': ['geometria', 'área', 'perímetro', 'volume']
                },
                {
                    'id': 'math_calculus_intro',
                    'title': 'Introdução ao Cálculo',
                    'content': '''
                    O cálculo é uma ferramenta matemática poderosa para analisar mudanças e tendências. Conceitos principais:
                    
                    1. Derivadas: Taxa de variação instantânea
                    2. Integrais: Acúmulo de quantidades
                    3. Limites: Comportamento de funções
                    4. Aplicações: Otimização, movimento, crescimento
                    
                    Exemplos:
                    - Derivada de x² é 2x
                    - Integral de 2x é x² + C
                    - Limite de 1/x quando x→∞ é 0
                    ''',
                    'domain': 'mathematics',
                    'difficulty_level': 8.0,
                    'tags': ['cálculo', 'derivadas', 'integrais', 'limites']
                }
            ]
            
            # Documentos de linguagem
            language_docs = [
                {
                    'id': 'lang_grammar_basics',
                    'title': 'Gramática Básica',
                    'content': '''
                    A gramática é o conjunto de regras que organizam a estrutura da língua. Elementos fundamentais:
                    
                    1. Classes gramaticais: substantivo, verbo, adjetivo, etc.
                    2. Concordância: entre sujeito e verbo, substantivo e adjetivo
                    3. Regência: relação entre verbos e preposições
                    4. Colocação: ordem das palavras na frase
                    
                    Exemplos:
                    - "O menino estuda" (concordância)
                    - "Gosto de música" (regência)
                    - "A casa grande" (colocação)
                    ''',
                    'domain': 'language',
                    'difficulty_level': 3.0,
                    'tags': ['gramática', 'concordância', 'regência', 'colocação']
                },
                {
                    'id': 'lang_literature_analysis',
                    'title': 'Análise Literária',
                    'content': '''
                    A análise literária examina textos para compreender seus significados e técnicas. Aspectos importantes:
                    
                    1. Gêneros literários: narrativo, lírico, dramático
                    2. Figuras de linguagem: metáfora, metonímia, hipérbole
                    3. Elementos narrativos: enredo, personagens, tempo, espaço
                    4. Contexto histórico e social
                    
                    Técnicas de análise:
                    - Leitura atenta e repetida
                    - Identificação de temas e símbolos
                    - Relação com contexto histórico
                    ''',
                    'domain': 'language',
                    'difficulty_level': 6.0,
                    'tags': ['literatura', 'análise', 'figuras de linguagem', 'gêneros']
                }
            ]
            
            # Documentos de ciências
            science_docs = [
                {
                    'id': 'sci_physics_motion',
                    'title': 'Movimento e Forças',
                    'content': '''
                    A física estuda os fenômenos naturais através de leis e princípios. Conceitos fundamentais:
                    
                    1. Movimento: posição, velocidade, aceleração
                    2. Forças: gravidade, atrito, força normal
                    3. Leis de Newton: inércia, força, ação e reação
                    4. Energia: cinética, potencial, conservação
                    
                    Aplicações:
                    - Engenharia e tecnologia
                    - Medicina e saúde
                    - Esportes e atividades físicas
                    ''',
                    'domain': 'science',
                    'difficulty_level': 6.0,
                    'tags': ['física', 'movimento', 'forças', 'energia']
                },
                {
                    'id': 'sci_chemistry_reactions',
                    'title': 'Reações Químicas',
                    'content': '''
                    As reações químicas transformam substâncias em outras. Elementos importantes:
                    
                    1. Reagentes e produtos
                    2. Balanceamento de equações
                    3. Tipos de reação: síntese, decomposição, substituição
                    4. Fatores que influenciam: temperatura, pressão, catalisadores
                    
                    Exemplos:
                    - 2H₂ + O₂ → 2H₂O (síntese)
                    - CaCO₃ → CaO + CO₂ (decomposição)
                    ''',
                    'domain': 'science',
                    'difficulty_level': 7.0,
                    'tags': ['química', 'reações', 'equações', 'balanceamento']
                }
            ]
            
            # Combinar todos os documentos
            all_docs = math_docs + language_docs + science_docs
            
            # Criar documentos e adicionar à base
            for doc_data in all_docs:
                doc = KnowledgeDocument(
                    id=doc_data['id'],
                    title=doc_data['title'],
                    content=doc_data['content'],
                    domain=doc_data['domain'],
                    difficulty_level=doc_data['difficulty_level'],
                    tags=doc_data['tags'],
                    metadata={'source': 'curated', 'language': 'pt-BR'}
                )
                self.documents[doc.id] = doc
            
            # Gerar embeddings e criar índices
            self._generate_embeddings()
            self._create_indexes()
            
            logger.info(f"Knowledge base populated with {len(self.documents)} documents")
            
        except Exception as e:
            logger.error(f"Error populating knowledge base: {e}")
    
    def _generate_embeddings(self):
        """Gera embeddings para todos os documentos"""
        try:
            documents = list(self.documents.values())
            texts = [doc.content for doc in documents]
            
            # Gerar embeddings
            embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
            
            # Atribuir embeddings aos documentos
            for doc, embedding in zip(documents, embeddings):
                doc.embedding = embedding
            
            logger.info(f"Generated embeddings for {len(documents)} documents")
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
    
    def _create_indexes(self):
        """Cria índices FAISS para busca eficiente"""
        try:
            documents = list(self.documents.values())
            embeddings = np.array([doc.embedding for doc in documents])
            
            # Criar índice FAISS
            dimension = embeddings.shape[1]
            self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner Product para similaridade de cosseno
            self.faiss_index.add(embeddings.astype('float32'))
            
            # Criar índices por domínio
            for domain in set(doc.domain for doc in documents):
                domain_docs = [doc for doc in documents if doc.domain == domain]
                domain_embeddings = np.array([doc.embedding for doc in domain_docs])
                
                domain_index = faiss.IndexFlatIP(dimension)
                domain_index.add(domain_embeddings.astype('float32'))
                self.domain_indexes[domain] = {
                    'index': domain_index,
                    'documents': domain_docs
                }
            
            logger.info(f"Created FAISS indexes for {len(self.domain_indexes)} domains")
            
        except Exception as e:
            logger.error(f"Error creating indexes: {e}")
    
    @external_api_circuit_breaker
    @track_ml_metrics
    def retrieve_relevant_documents(
        self, 
        query: str, 
        domain: Optional[str] = None,
        max_results: int = 5,
        similarity_threshold: float = 0.7
    ) -> List[Tuple[KnowledgeDocument, float]]:
        """
        Recupera documentos relevantes usando múltiplas estratégias
        """
        try:
            # Embedding da query
            query_embedding = self.embedding_model.encode([query])
            
            # Busca por similaridade de embedding
            embedding_results = self._search_by_embedding(query_embedding, domain, max_results)
            
            # Busca por TF-IDF
            tfidf_results = self._search_by_tfidf(query, domain, max_results)
            
            # Busca por tags e metadados
            metadata_results = self._search_by_metadata(query, domain, max_results)
            
            # Combinar e rankear resultados
            combined_results = self._combine_search_results(
                embedding_results, tfidf_results, metadata_results
            )
            
            # Filtrar por threshold de similaridade
            filtered_results = [
                (doc, score) for doc, score in combined_results 
                if score >= similarity_threshold
            ]
            
            # Atualizar estatísticas de acesso
            for doc, _ in filtered_results:
                doc.last_accessed = datetime.now()
                doc.access_count += 1
            
            return filtered_results[:max_results]
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return []
    
    def _search_by_embedding(
        self, 
        query_embedding: np.ndarray, 
        domain: Optional[str] = None,
        max_results: int = 5
    ) -> List[Tuple[KnowledgeDocument, float]]:
        """Busca por similaridade de embedding"""
        try:
            if domain and domain in self.domain_indexes:
                # Busca no domínio específico
                domain_data = self.domain_indexes[domain]
                scores, indices = domain_data['index'].search(
                    query_embedding.astype('float32'), max_results
                )
                documents = domain_data['documents']
            else:
                # Busca global
                scores, indices = self.faiss_index.search(
                    query_embedding.astype('float32'), max_results
                )
                documents = list(self.documents.values())
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(documents):
                    results.append((documents[idx], float(score)))
            
            return results
            
        except Exception as e:
            logger.error(f"Error in embedding search: {e}")
            return []
    
    def _search_by_tfidf(
        self, 
        query: str, 
        domain: Optional[str] = None,
        max_results: int = 5
    ) -> List[Tuple[KnowledgeDocument, float]]:
        """Busca por TF-IDF"""
        try:
            # Filtrar documentos por domínio se especificado
            if domain:
                documents = [doc for doc in self.documents.values() if doc.domain == domain]
            else:
                documents = list(self.documents.values())
            
            if not documents:
                return []
            
            # Preparar textos para TF-IDF
            texts = [doc.content for doc in documents]
            
            # Ajustar TF-IDF com os textos
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
            query_vector = self.tfidf_vectorizer.transform([query])
            
            # Calcular similaridade
            similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
            
            # Rankear resultados
            results = []
            for doc, similarity in zip(documents, similarities):
                results.append((doc, float(similarity)))
            
            # Ordenar por similaridade
            results.sort(key=lambda x: x[1], reverse=True)
            
            return results[:max_results]
            
        except Exception as e:
            logger.error(f"Error in TF-IDF search: {e}")
            return []
    
    def _search_by_metadata(
        self, 
        query: str, 
        domain: Optional[str] = None,
        max_results: int = 5
    ) -> List[Tuple[KnowledgeDocument, float]]:
        """Busca por metadados (tags, título, domínio)"""
        try:
            query_lower = query.lower()
            results = []
            
            for doc in self.documents.values():
                if domain and doc.domain != domain:
                    continue
                
                score = 0.0
                
                # Busca no título
                if query_lower in doc.title.lower():
                    score += 0.8
                
                # Busca nas tags
                for tag in doc.tags:
                    if query_lower in tag.lower():
                        score += 0.6
                
                # Busca no domínio
                if query_lower in doc.domain.lower():
                    score += 0.4
                
                if score > 0:
                    results.append((doc, score))
            
            # Ordenar por score
            results.sort(key=lambda x: x[1], reverse=True)
            
            return results[:max_results]
            
        except Exception as e:
            logger.error(f"Error in metadata search: {e}")
            return []
    
    def _combine_search_results(
        self,
        embedding_results: List[Tuple[KnowledgeDocument, float]],
        tfidf_results: List[Tuple[KnowledgeDocument, float]],
        metadata_results: List[Tuple[KnowledgeDocument, float]]
    ) -> List[Tuple[KnowledgeDocument, float]]:
        """Combina resultados de diferentes estratégias de busca"""
        try:
            # Criar dicionário para combinar scores
            combined_scores = {}
            
            # Processar resultados de embedding (peso 0.5)
            for doc, score in embedding_results:
                combined_scores[doc.id] = combined_scores.get(doc.id, 0) + score * 0.5
            
            # Processar resultados TF-IDF (peso 0.3)
            for doc, score in tfidf_results:
                combined_scores[doc.id] = combined_scores.get(doc.id, 0) + score * 0.3
            
            # Processar resultados de metadados (peso 0.2)
            for doc, score in metadata_results:
                combined_scores[doc.id] = combined_scores.get(doc.id, 0) + score * 0.2
            
            # Criar lista final ordenada
            final_results = []
            for doc_id, combined_score in combined_scores.items():
                doc = self.documents[doc_id]
                final_results.append((doc, combined_score))
            
            # Ordenar por score combinado
            final_results.sort(key=lambda x: x[1], reverse=True)
            
            return final_results
            
        except Exception as e:
            logger.error(f"Error combining search results: {e}")
            return embedding_results  # Fallback para resultados de embedding
    
    def extract_relevant_info(
        self, 
        documents: List[Tuple[KnowledgeDocument, float]],
        query: str,
        max_length: int = 1000
    ) -> str:
        """
        Extrai informações relevantes dos documentos recuperados
        """
        try:
            if not documents:
                return "Não encontrei informações específicas sobre sua pergunta."
            
            # Ordenar por relevância
            sorted_docs = sorted(documents, key=lambda x: x[1], reverse=True)
            
            # Extrair informações dos documentos mais relevantes
            relevant_info = []
            current_length = 0
            
            for doc, score in sorted_docs:
                if current_length >= max_length:
                    break
                
                # Extrair trechos relevantes do documento
                doc_info = self._extract_relevant_sections(doc, query, max_length - current_length)
                
                if doc_info:
                    relevant_info.append(f"**{doc.title}** (Relevância: {score:.2f})\n{doc_info}")
                    current_length += len(doc_info)
            
            if not relevant_info:
                return "Encontrei documentos relacionados, mas não consegui extrair informações específicas para sua pergunta."
            
            return "\n\n".join(relevant_info)
            
        except Exception as e:
            logger.error(f"Error extracting relevant info: {e}")
            return "Ocorreu um erro ao processar as informações."
    
    def _extract_relevant_sections(
        self, 
        document: KnowledgeDocument, 
        query: str, 
        max_length: int
    ) -> str:
        """Extrai seções relevantes de um documento"""
        try:
            # Dividir conteúdo em seções
            sections = document.content.split('\n\n')
            
            relevant_sections = []
            current_length = 0
            
            query_terms = query.lower().split()
            
            for section in sections:
                if current_length >= max_length:
                    break
                
                # Calcular relevância da seção
                section_lower = section.lower()
                relevance_score = sum(1 for term in query_terms if term in section_lower)
                
                if relevance_score > 0:
                    # Limitar tamanho da seção
                    if len(section) > max_length - current_length:
                        section = section[:max_length - current_length] + "..."
                    
                    relevant_sections.append(section)
                    current_length += len(section)
            
            return "\n\n".join(relevant_sections)
            
        except Exception as e:
            logger.error(f"Error extracting sections: {e}")
            return document.content[:max_length] if len(document.content) > max_length else document.content
    
    def add_document(
        self, 
        title: str, 
        content: str, 
        domain: str, 
        difficulty_level: float,
        tags: List[str],
        metadata: Dict[str, Any] = None
    ) -> str:
        """Adiciona novo documento à base de conhecimento"""
        try:
            # Gerar ID único
            doc_id = hashlib.md5(f"{title}{content[:100]}".encode()).hexdigest()[:12]
            
            # Criar documento
            doc = KnowledgeDocument(
                id=doc_id,
                title=title,
                content=content,
                domain=domain,
                difficulty_level=difficulty_level,
                tags=tags,
                metadata=metadata or {}
            )
            
            # Gerar embedding
            doc.embedding = self.embedding_model.encode([content])[0]
            
            # Adicionar à base
            self.documents[doc_id] = doc
            
            # Atualizar índices
            self._update_indexes()
            
            logger.info(f"Added document {doc_id} to knowledge base")
            return doc_id
            
        except Exception as e:
            logger.error(f"Error adding document: {e}")
            raise
    
    def _update_indexes(self):
        """Atualiza índices após adição de documentos"""
        try:
            # Recriar índice principal
            documents = list(self.documents.values())
            embeddings = np.array([doc.embedding for doc in documents])
            
            dimension = embeddings.shape[1]
            self.faiss_index = faiss.IndexFlatIP(dimension)
            self.faiss_index.add(embeddings.astype('float32'))
            
            # Atualizar índices por domínio
            self.domain_indexes.clear()
            for domain in set(doc.domain for doc in documents):
                domain_docs = [doc for doc in documents if doc.domain == domain]
                domain_embeddings = np.array([doc.embedding for doc in domain_docs])
                
                domain_index = faiss.IndexFlatIP(dimension)
                domain_index.add(domain_embeddings.astype('float32'))
                self.domain_indexes[domain] = {
                    'index': domain_index,
                    'documents': domain_docs
                }
            
            logger.info("Updated FAISS indexes")
            
        except Exception as e:
            logger.error(f"Error updating indexes: {e}")

class RAGService:
    """Serviço principal RAG"""
    
    def __init__(self):
        self.rag_system = AdvancedRAGSystem()
    
    @external_api_circuit_breaker
    async def query_knowledge_base(
        self, 
        query: str, 
        domain: Optional[str] = None,
        max_results: int = 5
    ) -> Dict[str, Any]:
        """
        Consulta a base de conhecimento e retorna informações relevantes
        """
        try:
            # Recuperar documentos relevantes
            relevant_docs = self.rag_system.retrieve_relevant_documents(
                query, domain, max_results
            )
            
            # Extrair informações relevantes
            relevant_info = self.rag_system.extract_relevant_info(relevant_docs, query)
            
            # Estatísticas da consulta
            stats = {
                'total_documents_found': len(relevant_docs),
                'average_relevance_score': np.mean([score for _, score in relevant_docs]) if relevant_docs else 0,
                'domains_covered': list(set(doc.domain for doc, _ in relevant_docs)),
                'query_processed_at': datetime.now().isoformat()
            }
            
            return {
                'query': query,
                'relevant_information': relevant_info,
                'source_documents': [
                    {
                        'id': doc.id,
                        'title': doc.title,
                        'domain': doc.domain,
                        'relevance_score': score,
                        'tags': doc.tags
                    }
                    for doc, score in relevant_docs
                ],
                'statistics': stats
            }
            
        except Exception as e:
            logger.error(f"Error querying knowledge base: {e}")
            return {
                'query': query,
                'relevant_information': "Desculpe, ocorreu um erro ao processar sua consulta.",
                'source_documents': [],
                'statistics': {'error': str(e)}
            }
    
    async def add_knowledge_document(
        self, 
        title: str, 
        content: str, 
        domain: str, 
        difficulty_level: float,
        tags: List[str],
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Adiciona novo documento à base de conhecimento"""
        try:
            doc_id = self.rag_system.add_document(
                title, content, domain, difficulty_level, tags, metadata
            )
            
            return {
                'document_id': doc_id,
                'status': 'success',
                'message': 'Documento adicionado com sucesso'
            }
            
        except Exception as e:
            logger.error(f"Error adding knowledge document: {e}")
            return {
                'status': 'error',
                'message': f'Erro ao adicionar documento: {str(e)}'
            }
    
    async def get_knowledge_statistics(self) -> Dict[str, Any]:
        """Obtém estatísticas da base de conhecimento"""
        try:
            documents = list(self.rag_system.documents.values())
            
            stats = {
                'total_documents': len(documents),
                'domains': {},
                'average_difficulty': np.mean([doc.difficulty_level for doc in documents]),
                'most_accessed_documents': [],
                'recent_additions': []
            }
            
            # Estatísticas por domínio
            for doc in documents:
                domain = doc.domain
                if domain not in stats['domains']:
                    stats['domains'][domain] = 0
                stats['domains'][domain] += 1
            
            # Documentos mais acessados
            accessed_docs = [doc for doc in documents if doc.access_count > 0]
            accessed_docs.sort(key=lambda x: x.access_count, reverse=True)
            stats['most_accessed_documents'] = [
                {
                    'id': doc.id,
                    'title': doc.title,
                    'access_count': doc.access_count
                }
                for doc in accessed_docs[:5]
            ]
            
            # Adições recentes
            recent_docs = sorted(documents, key=lambda x: x.created_at, reverse=True)
            stats['recent_additions'] = [
                {
                    'id': doc.id,
                    'title': doc.title,
                    'domain': doc.domain,
                    'created_at': doc.created_at.isoformat()
                }
                for doc in recent_docs[:5]
            ]
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting knowledge statistics: {e}")
            return {'error': str(e)}