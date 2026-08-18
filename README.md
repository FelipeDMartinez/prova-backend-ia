📚 Questão 1: API de Biblioteca Virtual (FastAPI + SQLite)
O que faz:
Cria uma API RESTful para cadastrar e consultar livros (título, autor, data de publicação e resumo).

Permite filtrar livros por título e/ou autor com busca parcial e paginação.

Persiste os dados em um banco de dados SQLite local via SQLAlchemy.

Inclui testes unitários automatizados cobrindo criação, filtros, erros de validação e rotas inexistentes com banco em memória.

Como rodar:
Iniciar a API:

Bash
uvicorn main:app --reload
Acesse a documentação interativa em: http://127.0.0.1:8000/docs

Executar os testes unitários:

Bash
pytest -v test_api.py
🤖 Questão 2: Chatbot Especialista em Python (LangChain + OpenAI)
O que faz:
Implementa um assistente conversacional tutor em linguagem Python utilizando o modelo GPT-4o da OpenAI e LangChain (LCEL).

Mantém memória do histórico de mensagens por sessão para responder perguntas em sequência.

Possui integração nativa com o LangSmith para rastreamento de traces, latência e custo de tokens.

Como rodar:
Configure sua chave da OpenAI no terminal:

Windows (PowerShell): $env:OPENAI_API_KEY="sua_chave_aqui"

Linux/macOS: export OPENAI_API_KEY="sua_chave_aqui"

Execute o chatbot:

Bash
python chatbot.py
🔍 Questão 3: Busca Semântica de Documentos (FAISS + Embeddings)
O que faz:
Converte um conjunto de artigos técnicos em embeddings densos locais utilizando o modelo sentence-transformers/all-MiniLM-L6-v2.

Armazena e indexa os vetores no banco vetorial FAISS.

Realiza busca semântica aproximada (ANN) retornando os documentos mais relevantes por similaridade/distância L2 com base no texto consultado.

Como rodar:
Executar a busca semântica:

Bash
python semantic_search.py
