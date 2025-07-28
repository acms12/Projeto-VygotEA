#!/usr/bin/env python3
"""
Script de teste básico para verificar se o sistema Vygotea está funcionando
"""
import asyncio
import json
import httpx
import time
from typing import Dict, Any

async def test_system_health():
    """Testa a saúde básica do sistema"""
    print("🔍 Testando saúde do sistema...")
    
    async with httpx.AsyncClient() as client:
        try:
            # Teste do endpoint raiz
            response = await client.get("http://localhost:8000/")
            print(f"✅ Endpoint raiz: {response.status_code}")
            print(f"   Resposta: {response.json()}")
            
            # Teste do endpoint de saúde
            response = await client.get("http://localhost:8000/health")
            print(f"✅ Endpoint de saúde: {response.status_code}")
            print(f"   Resposta: {response.json()}")
            
            # Teste das informações do sistema
            response = await client.get("http://localhost:8000/api/v1/system/info")
            print(f"✅ Informações do sistema: {response.status_code}")
            print(f"   Resposta: {response.json()}")
            
            return True
            
        except httpx.ConnectError:
            print("❌ Erro: Não foi possível conectar ao servidor")
            print("   Certifique-se de que o servidor está rodando em http://localhost:8000")
            return False
        except Exception as e:
            print(f"❌ Erro inesperado: {e}")
            return False

async def test_zdp_assessment():
    """Testa a funcionalidade de avaliação ZDP"""
    print("\n🧠 Testando avaliação ZDP...")
    
    test_data = {
        "user_id": "test_user_001",
        "domain_assessments": [
            {
                "domain": "mathematics",
                "score": 0.75,
                "confidence": 0.8,
                "difficulty": "medium"
            },
            {
                "domain": "language",
                "score": 0.6,
                "confidence": 0.7,
                "difficulty": "medium"
            },
            {
                "domain": "science",
                "score": 0.85,
                "confidence": 0.9,
                "difficulty": "hard"
            }
        ],
        "learning_style": "visual",
        "recent_activity": "completed_math_exercise"
    }
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                "http://localhost:8000/api/v1/zdp/assess",
                json=test_data
            )
            print(f"✅ Avaliação ZDP: {response.status_code}")
            result = response.json()
            print(f"   Nível atual: {result.get('current_level', 'N/A')}")
            print(f"   Zona proximal: {result.get('proximal_zone', 'N/A')}")
            print(f"   Potencial de desenvolvimento: {result.get('development_potential', 'N/A')}")
            print(f"   Nível de suporte: {result.get('support_level', 'N/A')}")
            
            return True
            
        except Exception as e:
            print(f"❌ Erro na avaliação ZDP: {e}")
            return False

async def test_gamification():
    """Testa a funcionalidade de gamificação"""
    print("\n🎮 Testando gamificação...")
    
    test_event = {
        "user_id": "test_user_001",
        "event_type": "completed_exercise",
        "domain": "mathematics",
        "difficulty": "medium",
        "score": 0.8,
        "time_spent": 300,
        "metadata": {
            "exercise_type": "algebra",
            "questions_count": 10
        }
    }
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                "http://localhost:8000/api/v1/gamification/event",
                json=test_event
            )
            print(f"✅ Evento de gamificação: {response.status_code}")
            result = response.json()
            print(f"   XP ganho: {result.get('xp_earned', 'N/A')}")
            print(f"   Novos badges: {result.get('new_badges', [])}")
            print(f"   Streak atual: {result.get('current_streak', 'N/A')}")
            
            return True
            
        except Exception as e:
            print(f"❌ Erro na gamificação: {e}")
            return False

async def test_rag_query():
    """Testa a funcionalidade RAG"""
    print("\n📚 Testando consulta RAG...")
    
    test_query = {
        "query": "Como resolver equações de segundo grau?",
        "domain": "mathematics",
        "max_results": 5
    }
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                "http://localhost:8000/api/v1/rag/query",
                json=test_query
            )
            print(f"✅ Consulta RAG: {response.status_code}")
            result = response.json()
            print(f"   Documentos encontrados: {len(result.get('documents', []))}")
            print(f"   Informação relevante: {result.get('relevant_info', 'N/A')[:100]}...")
            
            return True
            
        except Exception as e:
            print(f"❌ Erro na consulta RAG: {e}")
            return False

async def test_intervention():
    """Testa a funcionalidade de intervenção"""
    print("\n💬 Testando geração de intervenção...")
    
    test_context = {
        "user_id": "test_user_001",
        "message": "Estou tendo dificuldade com equações de segundo grau",
        "sentiment": "frustrated",
        "user_state": "struggling",
        "recent_activity": "failed_exercise",
        "zdp_assessment": {
            "current_level": 0.6,
            "support_level": "high"
        }
    }
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                "http://localhost:8000/api/v1/intervention/generate",
                json=test_context
            )
            print(f"✅ Geração de intervenção: {response.status_code}")
            result = response.json()
            print(f"   Resposta: {result.get('response_text', 'N/A')[:100]}...")
            print(f"   Urgência: {result.get('urgency_level', 'N/A')}")
            print(f"   Sugestões: {result.get('adaptive_suggestions', [])}")
            
            return True
            
        except Exception as e:
            print(f"❌ Erro na geração de intervenção: {e}")
            return False

async def test_integrated_chat():
    """Testa o chat integrado"""
    print("\n💭 Testando chat integrado...")
    
    test_message = {
        "user_id": "test_user_001",
        "message": "Preciso de ajuda com matemática",
        "context": {
            "current_domain": "mathematics",
            "recent_performance": 0.6
        }
    }
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                "http://localhost:8000/api/v1/chat",
                json=test_message
            )
            print(f"✅ Chat integrado: {response.status_code}")
            result = response.json()
            print(f"   Resposta: {result.get('response', 'N/A')[:100]}...")
            print(f"   ZDP atualizada: {result.get('zdp_updated', False)}")
            print(f"   Gamificação: {result.get('gamification_events', [])}")
            
            return True
            
        except Exception as e:
            print(f"❌ Erro no chat integrado: {e}")
            return False

async def main():
    """Função principal de teste"""
    print("🚀 Iniciando testes do sistema Vygotea...")
    print("=" * 50)
    
    # Aguardar um pouco para o servidor inicializar
    print("⏳ Aguardando inicialização do servidor...")
    await asyncio.sleep(2)
    
    tests = [
        ("Saúde do Sistema", test_system_health),
        ("Avaliação ZDP", test_zdp_assessment),
        ("Gamificação", test_gamification),
        ("Consulta RAG", test_rag_query),
        ("Intervenção", test_intervention),
        ("Chat Integrado", test_integrated_chat)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            success = await test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ Erro no teste {test_name}: {e}")
            results.append((test_name, False))
    
    # Resumo dos resultados
    print("\n" + "=" * 50)
    print("📊 RESUMO DOS TESTES")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASSOU" if success else "❌ FALHOU"
        print(f"{test_name}: {status}")
        if success:
            passed += 1
    
    print(f"\nResultado: {passed}/{total} testes passaram")
    
    if passed == total:
        print("🎉 Todos os testes passaram! Sistema funcionando corretamente.")
    else:
        print("⚠️  Alguns testes falharam. Verifique os logs do servidor.")

if __name__ == "__main__":
    asyncio.run(main())