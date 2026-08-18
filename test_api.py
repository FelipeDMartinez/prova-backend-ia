"""
Testes Unitários dos Endpoints da API
Executa sobre um banco SQLite em memória para garantir isolamento.
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from main import app, get_db, Base

# Banco em memória isolado para os testes
SQLALCHEMY_TEST_DATABASE_URL = "sqlite:///:memory:"

engine_test = create_engine(
    SQLALCHEMY_TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine_test)

@pytest.fixture(scope="function")
def db_session():
    """Recria as tabelas a cada teste para garantir estado limpo."""
    Base.metadata.create_all(bind=engine_test)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine_test)

@pytest.fixture(scope="function")
def client(db_session):
    """Sobrescreve a dependência get_db da aplicação para usar o banco de teste."""
    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


# ==========================================
# Casos de Teste
# ==========================================

def test_create_book_success(client):
    """Testa o cadastro bem-sucedido de um livro."""
    payload = {
        "title": "Clean Code",
        "author": "Robert C. Martin",
        "published_date": "2008-08-01",
        "summary": "Boas práticas para desenvolvimento de software limpo e manutenível."
    }
    response = client.post("/books/", json=payload)
    assert response.status_code == 201
    data = response.json()
    assert data["id"] == 1
    assert data["title"] == payload["title"]
    assert data["author"] == payload["author"]


def test_create_book_missing_fields(client):
    """Testa erro de validação (HTTP 422) ao enviar campos incompletos."""
    invalid_payload = {
        "title": "Incomplete Book"
    }
    response = client.post("/books/", json=invalid_payload)
    assert response.status_code == 422


def test_list_books_with_filters(client):
    """Testa a consulta com filtros por título e por autor."""
    # Inserção de dados de teste
    client.post("/books/", json={
        "title": "Arquitetura Limpa",
        "author": "Robert C. Martin",
        "published_date": "2017-09-20",
        "summary": "Padrões arquiteturais para sistemas robustos."
    })
    client.post("/books/", json={
        "title": "Design Patterns",
        "author": "Erich Gamma",
        "published_date": "1994-10-31",
        "summary": "Soluções reutilizáveis para problemas comuns de software."
    })

    # 1. Filtro por título
    res_title = client.get("/books/?title=Limpa")
    assert res_title.status_code == 200
    assert len(res_title.json()) == 1
    assert res_title.json()[0]["title"] == "Arquitetura Limpa"

    # 2. Filtro por autor
    res_author = client.get("/books/?author=Gamma")
    assert res_author.status_code == 200
    assert len(res_author.json()) == 1
    assert res_author.json()[0]["author"] == "Erich Gamma"


def test_get_book_not_found(client):
    """Testa resposta HTTP 404 para ID inexistente."""
    response = client.get("/books/999")
    assert response.status_code == 404
    assert "não foi encontrado" in response.json()["detail"]