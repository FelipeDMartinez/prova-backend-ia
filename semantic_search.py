"""
Questão 3: Sistema de Busca Semântica com Embeddings e Vector Store
Tecnologias: FAISS, Sentence-Transformers (Hugging Face) e LangChain
"""

from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# ==========================================================
# 1. Base de Conhecimento (Corpus de Artigos Técnicos)
# ==========================================================
# Conjunto de documentos representando posts de um blog de tecnologia
corpus_artigos = [
    {
        "id": "doc-01",
        "title": "Otimização de Índices e Performance em SQL",
        "content": (
            "Para otimizar bancos relacionais em alta escala, o uso adequado de índices B-Tree "
            "e a análise de planos de execução (EXPLAIN ANALYZE) evitam varreduras completas (Full Table Scans)."
        ),
        "category": "Banco de Dados"
    },
    {
        "id": "doc-02",
        "title": "Arquitetura Transformer e Mecanismos de Atenção",
        "content": (
            "A arquitetura Transformer utiliza o mecanismo de Self-Attention para calcular o contexto "
            "global entre tokens em paralelo, servindo como base para grandes modelos de linguagem (LLMs)."
        ),
        "category": "Inteligência Artificial"
    },
    {
        "id": "doc-03",
        "title": "Deploy de APIs com FastAPI e Docker",
        "content": (
            "Empacotar APIs FastAPI em containers Docker padroniza o ambiente de desenvolvimento e produção. "
            "O uso de múltiplos workers com Uvicorn permite processar requisições concorrentes com baixo consumo de memória."
        ),
        "category": "Backend & DevOps"
    },
    {
        "id": "doc-04",
        "title": "Técnicas de RAG (Retrieval-Augmented Generation)",
        "content": (
            "Sistemas RAG integram modelos de linguagem a bancos vetoriais. Ao buscar documentos semanticamente "
            "relevantes para compor o contexto do prompt, mitigam-se alucinações e permite-se consultar dados privados."
        ),
        "category": "Inteligência Artificial"
    },
    {
        "id": "doc-05",
        "title": "Gerenciamento de Memória e Concorrência em Python",
        "content": (
            "Python gerencia memória via contagem de referências e coletor de lixo cíclico (Garbage Collector). "
            "Para operações I/O-bound, o módulo asyncio oferece concorrência cooperativa de alto desempenho."
        ),
        "category": "Python"
    }
]

# ==========================================================
# 2. Inicialização do Modelo de Embeddings
# ==========================================================
# Utiliza o modelo 'all-MiniLM-L6-v2' (gera vetores densos de 384 dimensões).
# Roda localmente na CPU, sem necessidade de chaves de API pagas.
print("1. Carregando o modelo de embeddings...")
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)

# ==========================================================
# 3. Criação dos Documentos e Indexação no FAISS
# ==========================================================
# Converte a base de artigos em objetos Document estruturados do LangChain
documents = [
    Document(
        page_content=artigo["content"],
        metadata={
            "id": artigo["id"],
            "title": artigo["title"],
            "category": artigo["category"]
        }
    )
    for artigo in corpus_artigos
]

print("2. Gerando embeddings e indexando documentos no FAISS...")
# FAISS cria o índice vetorial em memória para busca aproximada de vizinhos mais próximos (ANN)
vector_store = FAISS.from_documents(documents, embedding_model)
print("-> Índice vetorial criado com sucesso!\n")

# ==========================================================
# 4. Função de Busca Semântica
# ==========================================================
def semantic_search(query: str, top_k: int = 2) -> List[Dict[str, Any]]:
    """
    Executa a busca semântica por similaridade vetorial.
    
    Parâmetros:
    - query: Frase ou pergunta em linguagem natural.
    - top_k: Quantidade de documentos mais relevantes a retornar.
    
    Retorno:
    - Lista de dicionários contendo os dados do documento e o score de distância L2.
      (No FAISS padrão com distância L2, quanto menor a distância, mais similar é o texto).
    """
    # similarity_search_with_score retorna pares: (Document, distance_score)
    results_with_scores = vector_store.similarity_search_with_score(query, k=top_k)
    
    ranked_results = []
    for doc, score in results_with_scores:
        ranked_results.append({
            "id": doc.metadata.get("id"),
            "title": doc.metadata.get("title"),
            "category": doc.metadata.get("category"),
            "content": doc.page_content,
            "distance_score": round(float(score), 4)
        })
    return ranked_results

# ==========================================================
# 5. Demonstração com Exemplos Práticos
# ==========================================================
if __name__ == "__main__":
    consultas_teste = [
        "Como evitar que o modelo invente informações usando dados internos?",
        "Qual é a melhor estratégia para deixar as consultas ao banco mais rápidas?",
        "Como publicar e rodar uma API moderna em ambiente de produção?"
    ]

    print("=" * 70)
    print("🔍 DEMONSTRAÇÃO DO SISTEMA DE BUSCA SEMÂNTICA")
    print("=" * 70)

    for i, consulta in enumerate(consultas_teste, 1):
        print(f"\n[Consulta #{i}]: '{consulta}'")
        resultados = semantic_search(consulta, top_k=2)

        for pos, item in enumerate(resultados, 1):
            print(f"  └─ Rank #{pos} (Distância L2: {item['distance_score']})")
            print(f"     Título: {item['title']} | Categoria: {item['category']}")
            print(f"     Trecho: \"{item['content']}\"")
        print("-" * 70)

    # Modo interativo via terminal
    print("\nModo interativo (digite 'sair' para encerrar):")
    while True:
        try:
            user_query = input("\nDigite sua busca: ").strip()
            if not user_query or user_query.lower() in ["sair", "exit", "quit"]:
                print("Encerrando busca semântica.")
                break
            
            matches = semantic_search(user_query, top_k=2)
            print("\nResultados mais relevantes:")
            for pos, item in enumerate(matches, 1):
                print(f" [{pos}] {item['title']} (Score: {item['distance_score']})")
                print(f"     {item['content']}")
        except KeyboardInterrupt:
            print("\nEncerrando...")
            break