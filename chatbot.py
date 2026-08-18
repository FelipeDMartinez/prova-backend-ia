"""
Questão 2: Chatbot Especialista em Python
Tecnologias: LangChain (LCEL), OpenAI (GPT-4o) e LangSmith Tracing
"""

import os
from typing import Dict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

# ==========================================================
# 1. Configurações de Ambiente (OpenAI & LangSmith)
# ==========================================================
# O LangSmith permite monitorar latência, custo de tokens e inspecionar prompts/respostas
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "true")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "python-tutor-bot")

# Certifique-se de exportar sua chave da OpenAI antes de rodar:
# export OPENAI_API_KEY="sk-..."
if not os.getenv("OPENAI_API_KEY"):
    print("Aviso: Variável de ambiente OPENAI_API_KEY não foi encontrada. Configure-a para executar o modelo.")

# ==========================================================
# 2. Definição do Prompt do Sistema (Persona e Diretrizes)
# ==========================================================
system_instructions = (
    "Você é um tutor sênior e mentor especialista em programação na linguagem Python.\n"
    "Suas diretrizes de resposta são:\n"
    "1. Fornecer explicações didáticas, diretas e com rigor técnico.\n"
    "2. Incluir exemplos práticos de código seguindo as convenções da PEP 8.\n"
    "3. Adicionar comentários explicativos no código para facilitar o aprendizado.\n"
    "4. Explicar boas práticas e cuidados comuns relacionados ao tema perguntado."
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_instructions),
    MessagesPlaceholder(variable_name="history"),  # Injeção dinâmica do histórico da conversa
    ("human", "{input}")
])

# ==========================================================
# 3. Inicialização do Modelo (LLM) e Construção da Chain
# ==========================================================
# Instanciação do modelo GPT-4o com baixa temperatura para respostas precisas
llm = ChatOpenAI(
    model_name="gpt-4o",
    temperature=0.2
)

# Construção da Chain via LangChain Expression Language (LCEL)
chain = prompt | llm | StrOutputParser()

# ==========================================================
# 4. Gerenciamento de Memória Conversacional (Por Sessão)
# ==========================================================
session_store: Dict[str, ChatMessageHistory] = {}

def get_session_history(session_id: str) -> ChatMessageHistory:
    """Recupera ou instancia um histórico em memória para o identificador da sessão."""
    if session_id not in session_store:
        session_store[session_id] = ChatMessageHistory()
    return session_store[session_id]

# Envolve a chain com suporte a histórico por sessão
conversational_bot = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)

# ==========================================================
# 5. Função de Interação
# ==========================================================
def ask_tutor(question: str, session_id: str = "default_session") -> str:
    """Envia uma mensagem ao assistente e retorna a resposta gerada."""
    return conversational_bot.invoke(
        {"input": question},
        config={"configurable": {"session_id": session_id}}
    )

# ==========================================================
# 6. Execução Interativa / Demonstração
# ==========================================================
if __name__ == "__main__":
    print("=" * 60)
    print("🤖 Tutor de Python Inicializado (digite 'sair' para encerrar)")
    print("=" * 60)

    # Exemplo 1: Pergunta padrão solicitada no enunciado
    exemplo_pergunta = "Como criar uma lista em Python?"
    print(f"\n[Usuário]: {exemplo_pergunta}")
    print(f"[Tutor Python]:\n{ask_tutor(exemplo_pergunta, session_id='demo_session')}\n")

    # Exemplo 2: Pergunta em sequência para testar retenção de contexto
    exemplo_contexto = "E como adiciono um novo elemento no final dessa lista?"
    print(f"[Usuário]: {exemplo_contexto}")
    print(f"[Tutor Python]:\n{ask_tutor(exemplo_contexto, session_id='demo_session')}\n")

    # Modo interativo no terminal
    print("-" * 60)
    print("Agora você pode fazer suas próprias perguntas:")
    while True:
        try:
            user_input = input("\nVocê: ").strip()
            if not user_input or user_input.lower() in ["sair", "exit", "quit"]:
                print("Encerrando o assistente. Até mais!")
                break
            
            resposta = ask_tutor(user_input, session_id="terminal_user")
            print(f"\nTutor Python:\n{resposta}")
        except KeyboardInterrupt:
            print("\nEncerrando...")
            break