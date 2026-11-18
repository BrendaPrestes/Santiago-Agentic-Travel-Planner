# Planejador Inteligente de Viagens para Santiago — Módulo de Criação do RAG
# Baseado na arquitetura de referência (Data Science Academy — Projeto 3)

import os
import json
import shutil
from typing import Literal
from pydantic import BaseModel, Field, ValidationError
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Diretório onde os dados de origem (JSONs) estão armazenados
DIRETORIO_DADOS = "dados_origem"

# Diretório onde o banco de dados vetorial será armazenado
DIRETORIO_VECTORDB = "rag/chroma_db"


# Contrato Pydantic para validação estrita dos itens curados (Slide 09 / Spec Seção 3)
class ChunkMeta(BaseModel):
    id_item: str
    nome: str
    categoria: Literal["hospedagem", "passeio", "restaurante"]
    bairro: str
    perfil: list[Literal["casal", "familia", "individual"]]
    faixa_preco: Literal["baixo", "medio", "alto"]
    clima: Literal["ar_livre_sol", "coberto_qualquer_clima", "neve_inverno"]
    preco_brl: float = Field(gt=0)
    preco_clp_base: int = Field(gt=0)
    parceiro_cupom: bool = False
    descricao: str
    ultima_atualizacao: str


def carregar_e_validar_documentos(diretorio: str = DIRETORIO_DADOS) -> list[Document]:
    """Lê os arquivos JSON de dados_origem/, valida cada item com ChunkMeta e gera objetos Document."""
    documentos: list[Document] = []

    if not os.path.exists(diretorio):
        print(f"Diretório '{diretorio}' não encontrado.")
        return documentos

    arquivos_json = sorted(f for f in os.listdir(diretorio) if f.endswith(".json"))
    if not arquivos_json:
        print("Nenhum arquivo JSON encontrado.")
        return documentos

    for nome_arquivo in arquivos_json:
        caminho = os.path.join(diretorio, nome_arquivo)
        with open(caminho, "r", encoding="utf-8") as f:
            itens = json.load(f)

        for item in itens:
            try:
                validado = ChunkMeta(**item)
            except ValidationError as e:
                id_ref = item.get("id_item", "desconhecido")
                raise ValueError(
                    f"Erro de validação Pydantic no item '{id_ref}' do arquivo '{nome_arquivo}': {e}"
                ) from e

            perfis_str = ", ".join(validado.perfil)
            parceiro_txt = "Sim (Cupom de Desconto da Agência)" if validado.parceiro_cupom else "Não"

            conteudo_textual = (
                f"ID: {validado.id_item}\n"
                f"Nome: {validado.nome}\n"
                f"Categoria: {validado.categoria}\n"
                f"Bairro/Localização: {validado.bairro}\n"
                f"Perfil recomendado: {perfis_str}\n"
                f"Faixa de preço: {validado.faixa_preco}\n"
                f"Condição climática ideal: {validado.clima}\n"
                f"Preço em Reais (BRL): R$ {validado.preco_brl:.2f}\n"
                f"Preço base em Pesos Chilenos (CLP): {validado.preco_clp_base} CLP\n"
                f"Parceiro com cupom da agência: {parceiro_txt}\n"
                f"Descrição: {validado.descricao}\n"
                f"Última atualização: {validado.ultima_atualizacao}"
            )

            # ChromaDB aceita apenas str, int, float ou bool nos metadados
            metadata = {
                "source": caminho,
                "id_item": validado.id_item,
                "nome": validado.nome,
                "categoria": validado.categoria,
                "bairro": validado.bairro,
                "perfil": ",".join(validado.perfil),
                "faixa_preco": validado.faixa_preco,
                "clima": validado.clima,
                "preco_brl": float(validado.preco_brl),
                "preco_clp_base": int(validado.preco_clp_base),
                "parceiro_cupom": bool(validado.parceiro_cupom),
                "ultima_atualizacao": validado.ultima_atualizacao,
            }

            documentos.append(Document(page_content=conteudo_textual, metadata=metadata))

    return documentos


def cria_vectordb():
    """Valida os JSONs, divide em chunks, gera embeddings BAAI/bge-base-en e cria o ChromaDB."""
    print("\nValidando documentos com Pydantic (ChunkMeta) e gerando as Embeddings. Aguarde...")

    documents = carregar_e_validar_documentos(DIRETORIO_DADOS)
    if not documents:
        print("Nenhum documento válido encontrado.")
        return None

    print(f"Total de itens validados com sucesso: {len(documents)}")

    # Divisor de texto (mesma configuração do professor, garantindo que cada item caiba inteiro no chunk)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=50,
    )
    chunks = text_splitter.split_documents(documents)

    # Modelo de embeddings (mesmo do professor)
    model_name = "BAAI/bge-base-en"
    encode_kwargs = {"normalize_embeddings": True}

    embedding_model = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs=encode_kwargs,
    )

    # Limpa diretório anterior se existir para evitar duplicatas ao reindexar (ex.: via worker)
    if os.path.exists(DIRETORIO_VECTORDB):
        shutil.rmtree(DIRETORIO_VECTORDB)

    vectordb = Chroma.from_documents(
        chunks,
        embedding_model,
        persist_directory=DIRETORIO_VECTORDB,
    )

    print("\nBanco de Dados Vetorial do RAG Criado com Sucesso em 'rag/chroma_db'.\n")
    return vectordb


if __name__ == "__main__":
    cria_vectordb()
