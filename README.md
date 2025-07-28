# Vygotea - Sistema de Tutoria Inteligente

Sistema avançado de tutoria baseado em Zona de Desenvolvimento Proximal (ZDP) com gamificação e aprendizado de máquina.

## Características Principais

- **Análise ZDP Avançada**: Modelos adaptativos para cálculo de níveis e zonas proximais
- **Motor de Gamificação**: Sistema de XP, badges e streaks dinâmicos
- **Base de Conhecimento RAG**: Sistema robusto de recuperação e geração de conhecimento
- **Modelos ML Adaptativos**: Treinamento contínuo com feedback do usuário
- **Geração de Respostas Inteligente**: Integração com LLMs para respostas contextuais
- **Monitoramento e Observabilidade**: Métricas, logs estruturados e circuit breakers

## Estrutura do Projeto

```
vygotea/
├── app/
│   ├── core/           # Configurações e utilitários
│   ├── models/         # Modelos de dados Pydantic
│   ├── services/       # Lógica de negócio
│   ├── api/           # Rotas da API
│   ├── ml/            # Modelos de ML
│   └── gamification/  # Sistema de gamificação
├── tests/             # Testes
├── docs/              # Documentação
└── scripts/           # Scripts utilitários
```

## Instalação

```bash
pip install -r requirements.txt
```

## Configuração

Copie `.env.example` para `.env` e configure as variáveis:

```bash
cp .env.example .env
```

## Execução

```bash
uvicorn app.main:app --reload
```

## Documentação da API

Acesse `http://localhost:8000/docs` para a documentação interativa da API.
