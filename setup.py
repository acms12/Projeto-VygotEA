#!/usr/bin/env python3
"""
Script de setup para o sistema Vygotea
"""
import os
import sys
import subprocess
import shutil
from pathlib import Path

def print_banner():
    """Imprime o banner do sistema"""
    print("=" * 60)
    print("🚀 VYGOTEA - Sistema de Tutoria Inteligente")
    print("=" * 60)
    print("Setup e configuração do sistema")
    print("=" * 60)

def check_python_version():
    """Verifica a versão do Python"""
    print("🐍 Verificando versão do Python...")
    
    if sys.version_info < (3, 8):
        print("❌ Erro: Python 3.8 ou superior é necessário")
        print(f"   Versão atual: {sys.version}")
        return False
    
    print(f"✅ Python {sys.version.split()[0]} - OK")
    return True

def create_directories():
    """Cria diretórios necessários"""
    print("\n📁 Criando diretórios...")
    
    directories = [
        "models",
        "data",
        "data/training",
        "data/knowledge",
        "logs",
        "tests",
        "docs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"   ✅ {directory}/")

def create_env_file():
    """Cria arquivo .env se não existir"""
    print("\n⚙️  Configurando variáveis de ambiente...")
    
    if not os.path.exists(".env"):
        if os.path.exists(".env.example"):
            shutil.copy(".env.example", ".env")
            print("✅ Arquivo .env criado a partir de .env.example")
        else:
            print("⚠️  Arquivo .env.example não encontrado")
            print("   Crie manualmente o arquivo .env com as configurações necessárias")
    else:
        print("✅ Arquivo .env já existe")

def install_dependencies():
    """Instala as dependências"""
    print("\n📦 Instalando dependências...")
    
    try:
        # Verificar se pip está disponível
        subprocess.run([sys.executable, "-m", "pip", "--version"], 
                      check=True, capture_output=True)
        
        # Instalar dependências
        print("   Instalando pacotes do requirements.txt...")
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Dependências instaladas com sucesso")
        else:
            print("❌ Erro ao instalar dependências:")
            print(result.stderr)
            return False
            
    except subprocess.CalledProcessError:
        print("❌ Erro: pip não está disponível")
        return False
    except FileNotFoundError:
        print("❌ Erro: requirements.txt não encontrado")
        return False
    
    return True

def check_optional_dependencies():
    """Verifica dependências opcionais"""
    print("\n🔍 Verificando dependências opcionais...")
    
    optional_packages = [
        ("torch", "PyTorch para deep learning"),
        ("transformers", "Hugging Face Transformers"),
        ("sentence-transformers", "Modelos de embedding"),
        ("faiss-cpu", "FAISS para busca vetorial"),
        ("redis", "Redis para cache"),
        ("elasticsearch", "Elasticsearch"),
        ("prometheus-client", "Métricas Prometheus")
    ]
    
    for package, description in optional_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"   ✅ {package} - {description}")
        except ImportError:
            print(f"   ⚠️  {package} - {description} (opcional)")

def create_initial_data():
    """Cria dados iniciais para teste"""
    print("\n📊 Criando dados iniciais...")
    
    # Criar diretório de dados se não existir
    Path("data").mkdir(exist_ok=True)
    
    # Arquivo de exemplo de dados de treinamento
    training_data = {
        "examples": [
            {
                "text": "Como resolver equações de primeiro grau?",
                "domain": "mathematics",
                "difficulty": "easy",
                "label": "algebra_basic"
            },
            {
                "text": "Explicação sobre fotossíntese",
                "domain": "science",
                "difficulty": "medium",
                "label": "biology_photosynthesis"
            }
        ]
    }
    
    import json
    with open("data/training/initial_data.json", "w") as f:
        json.dump(training_data, f, indent=2)
    
    print("✅ Dados iniciais criados")

def run_basic_tests():
    """Executa testes básicos"""
    print("\n🧪 Executando testes básicos...")
    
    try:
        # Teste de importação dos módulos principais
        import app.core.config
        import app.models.zdp
        import app.services.zdp_service
        print("✅ Importações básicas - OK")
        
        # Teste de configuração
        from app.core.config import settings
        print(f"✅ Configuração carregada - Versão: {settings.version}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro nos testes básicos: {e}")
        return False

def print_next_steps():
    """Imprime próximos passos"""
    print("\n" + "=" * 60)
    print("🎉 SETUP CONCLUÍDO!")
    print("=" * 60)
    print("\n📋 Próximos passos:")
    print("1. Configure o arquivo .env com suas configurações")
    print("2. Execute o servidor: python -m app.main")
    print("3. Acesse a documentação: http://localhost:8000/docs")
    print("4. Execute os testes: python test_basic.py")
    print("\n🔗 URLs importantes:")
    print("   • API: http://localhost:8000")
    print("   • Documentação: http://localhost:8000/docs")
    print("   • Métricas: http://localhost:8000/metrics")
    print("   • Saúde: http://localhost:8000/health")
    print("\n📚 Recursos:")
    print("   • README.md - Documentação completa")
    print("   • .env.example - Exemplo de configuração")
    print("   • test_basic.py - Testes básicos")

def main():
    """Função principal do setup"""
    print_banner()
    
    # Verificar Python
    if not check_python_version():
        sys.exit(1)
    
    # Criar diretórios
    create_directories()
    
    # Criar arquivo .env
    create_env_file()
    
    # Instalar dependências
    if not install_dependencies():
        print("\n❌ Falha na instalação das dependências")
        print("   Verifique se você tem permissões adequadas")
        sys.exit(1)
    
    # Verificar dependências opcionais
    check_optional_dependencies()
    
    # Criar dados iniciais
    create_initial_data()
    
    # Executar testes básicos
    if not run_basic_tests():
        print("\n⚠️  Alguns testes falharam, mas o setup pode continuar")
    
    # Próximos passos
    print_next_steps()

if __name__ == "__main__":
    main()