"""
API de Biblioteca Virtual
Framework: FastAPI
ORM: SQLAlchemy
Banco de Dados: SQLite
"""

from datetime import date
from typing import List, Optional
from fastapi import FastAPI, Depends, HTTPException, Query, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import create_engine, Column, Integer, String, Date, Text
from sqlalchemy.orm import declarative_base, sessionmaker, Session

# ==========================================
# 1. Configuração do Banco de Dados (SQLite)
# ==========================================
DATABASE_URL = "sqlite:///./library.db"

# connect_args={"check_same_thread": False} é necessário para o SQLite no FastAPI
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# ==========================================
# 2. Modelo de Banco de Dados (SQLAlchemy)
# ==========================================
class BookModel(Base):
    __tablename__ = "books"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    title = Column(String(255), nullable=False, index=True)
    author = Column(String(255), nullable=False, index=True)
    published_date = Column(Date, nullable=False)
    summary = Column(Text, nullable=False)

# Cria as tabelas no banco de dados SQLite caso não existam
Base.metadata.create_all(bind=engine)

# ==========================================
# 3. Schemas de Validação e Serialização (Pydantic)
# ==========================================
class BookBase(BaseModel):
    title: str = Field(..., min_length=1, max_length=255, description="Título do livro")
    author: str = Field(..., min_length=1, max_length=255, description="Nome do autor")
    published_date: date = Field(..., description="Data de publicação no formato AAAA-MM-DD")
    summary: str = Field(..., min_length=5, description="Resumo da obra")

class BookCreate(BookBase):
    """Schema utilizado para criação de novos livros."""
    pass

class BookResponse(BookBase):
    """Schema retornado pela API, incluindo o ID gerado pelo banco."""
    id: int

    model_config = ConfigDict(from_attributes=True)

# ==========================================
# 4. Injeção de Dependência da Sessão
# ==========================================
def get_db():
    """Garante abertura e fechamento seguro da sessão do banco de dados por requisição."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ==========================================
# 5. Instância do FastAPI e Endpoints
# ==========================================
app = FastAPI(
    title="API de Biblioteca Virtual",
    description="API RESTful para cadastro e consulta de livros em SQLite.",
    version="1.0.0"
)

@app.post(
    "/books/",
    response_model=BookResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Cadastrar um novo livro",
    tags=["Livros"]
)
def create_book(book_in: BookCreate, db: Session = Depends(get_db)):
    """
    Cadastra um novo livro na biblioteca virtual.
    Valida campos obrigatórios: título, autor, data de publicação e resumo.
    """
    db_book = BookModel(
        title=book_in.title,
        author=book_in.author,
        published_date=book_in.published_date,
        summary=book_in.summary
    )
    db.add(db_book)
    db.commit()
    db.refresh(db_book)
    return db_book


@app.get(
    "/books/",
    response_model=List[BookResponse],
    status_code=status.HTTP_200_OK,
    summary="Consultar livros com filtros",
    tags=["Livros"]
)
def list_books(
    title: Optional[str] = Query(None, description="Filtro por título (busca parcial e insensível a maiúsculas/minúsculas)"),
    author: Optional[str] = Query(None, description="Filtro por autor (busca parcial e insensível a maiúsculas/minúsculas)"),
    skip: int = Query(0, ge=0, description="Número de registros a pular (paginação)"),
    limit: int = Query(20, ge=1, le=100, description="Número máximo de registros a retornar"),
    db: Session = Depends(get_db)
):
    """
    Retorna a lista de livros cadastrados com suporte a paginação e
    filtros opcionais por título e/ou autor.
    """
    query = db.query(BookModel)
    
    if title:
        query = query.filter(BookModel.title.ilike(f"%{title}%"))
    if author:
        query = query.filter(BookModel.author.ilike(f"%{author}%"))
        
    books = query.offset(skip).limit(limit).all()
    return books


@app.get(
    "/books/{book_id}",
    response_model=BookResponse,
    status_code=status.HTTP_200_OK,
    summary="Buscar livro por ID",
    tags=["Livros"]
)
def get_book_by_id(book_id: int, db: Session = Depends(get_db)):
    """Busca um livro específico pelo seu identificador numérico (ID)."""
    book = db.query(BookModel).filter(BookModel.id == book_id).first()
    if not book:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Livro com ID {book_id} não foi encontrado."
        )
    return book