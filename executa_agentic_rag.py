# Planejador Inteligente de Viagens para Santiago — Agentic RAG, Re-Ranking e LangGraph
# Arquitetura baseada no Projeto 3 do curso Construção e Deploy de Agentes de IA (Data Science Academy)

import os
import re
import time
import warnings
import requests
import numpy as np
import streamlit as st
from pydantic import BaseModel
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from sklearn.metrics.pairwise import cosine_similarity
from worker_atualiza_precos import executar_worker

# Evita problema de compatibilidade entre Streamlit e PyTorch
import torch
torch.classes.__path__ = []

# Filtra warnings e desativa paralelismo de tokenização
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Configurações do LM Studio e ChromaDB
LM_STUDIO_URL = "http://localhost:1234/v1"
LM_STUDIO_MODEL = "hermes-3-llama-3.2-3b@q6_k"
DIRETORIO_VECTORDB = "rag/chroma_db"


########## Etapa 1 - Configuração da Interface do Streamlit ##########

st.set_page_config(
    page_title="Planejador Santiago — Agentic RAG",
    page_icon="🏔️",
    layout="wide",
)

st.sidebar.title("⚙️ Parâmetros da Viagem")
perfil_padrao = st.sidebar.selectbox(
    "Perfil do Viajante (padrão se não informado na pergunta):",
    options=["casal", "familia", "individual"],
    index=0,
)
orcamento_padrao = st.sidebar.number_input(
    "Orçamento Máximo (R$):",
    min_value=1500.0,
    max_value=20000.0,
    value=5000.0,
    step=250.0,
)

st.sidebar.markdown("---")
st.sidebar.subheader("📌 Instruções")
st.sidebar.write(
    """
- Digite sua pergunta sobre roteiro, hotéis, passeios, restaurantes, clima ou orçamento em **Santiago do Chile**.
- O agente detecta automaticamente o **perfil** (`casal`, `familia` ou `individual`) e o **orçamento** caso você escreva na pergunta, ou usa os seletores acima.
- Se o custo inicial ultrapassar o teto (ex.: **R$ 5.200 > R$ 5.000**), a **aresta condicional do LangGraph** aciona o loop de autocorreção trocando pelo **Hotel Parceiro com Cupom (R$ 1.950)**, fechando em **R$ 4.650**.
"""
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "ultimo_pacote" not in st.session_state:
    st.session_state.ultimo_pacote = {}

if st.sidebar.button("🗑️ Limpar Memória (Buffer K=3)"):
    st.session_state.chat_history = []
    st.session_state.ultimo_pacote = {}
    st.sidebar.success("Memória de curto prazo limpa!")

if st.session_state.chat_history:
    with st.sidebar.expander(f"🧠 Memória Ativa (Buffer K={len(st.session_state.chat_history)}/3)", expanded=False):
        for i, h in enumerate(st.session_state.chat_history, 1):
            st.caption(f"**{i}. Pergunta:** {h['pergunta']}\n\n*(Perfil: {h.get('perfil', '-')}, Teto: R$ {h.get('orcamento', 0):.0f})*")

st.sidebar.markdown("---")
st.sidebar.subheader("🔄 Worker de Preços (CLP → BRL)")
if st.sidebar.button("🔄 Executar Worker e Reindexar RAG"):
    with st.sidebar.status("Consultando câmbio CLP→BRL e reindexando ChromaDB...", expanded=True) as status_w:
        res_w = executar_worker(modo_live=False)
        carregar_recursos_rag.clear()
        status_w.update(label="Worker e Reindexação concluídos!", state="complete")
    st.sidebar.success(
        f"**Modo:** {res_w['modo']}\n\n"
        f"**Câmbio:** R$ 1 = {res_w['cambio_clp_por_brl']} CLP ({res_w['fonte_cambio']})\n\n"
        f"**Itens validados e reindexados:** {res_w['itens_reindexados']}\n\n"
        f"**Atualizado em:** {res_w['ultima_atualizacao']}"
    )

st.title("🏔️ Planejador Inteligente de Viagens para Santiago — Agentic RAG, Re-Ranking e LangGraph")


########## Etapa 2 - Configuração do Modelo LLM e do Processo de Retrieve do RAG ##########

@st.cache_resource
def carregar_recursos_rag():
    embedding_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vector_db = Chroma(
        persist_directory=DIRETORIO_VECTORDB,
        embedding_function=embedding_model,
    )
    return embedding_model, vector_db


embedding_model, vector_db = carregar_recursos_rag()
retriever = vector_db.as_retriever(search_kwargs={"k": 6})

llm = ChatOpenAI(
    model_name=LM_STUDIO_MODEL,
    openai_api_base=LM_STUDIO_URL,
    openai_api_key="lm-studio",
    temperature=0.3,
    max_tokens=512,
    request_timeout=45,
)

prompt_rag = PromptTemplate.from_template(
    "Você é um consultor especialista em viagens para Santiago do Chile.\n"
    "REGRAS OBRIGATÓRIAS:\n"
    "1. Responda SEMPRE em português do Brasil.\n"
    "2. Todos os pacotes oficiais da agência têm duração FECHADA de EXATAMENTE 4 DIAS (4 diárias). "
    "Se o cliente pedir 5, 6 ou mais dias, explique gentilmente que a agência opera exclusivamente com pacotes fechados de 4 dias.\n"
    "3. NUNCA recalcule preços, NUNCA escreva 'por noite' ou 'por dia' nos valores do pacote e NUNCA invente um novo Valor Total diferente do Cálculo Financeiro Determinístico abaixo.\n"
    "4. Escreva apenas um comentário consultivo curto (máximo 2 parágrafos) sugerindo como distribuir os passeios nos 4 dias de acordo com o clima.\n\n"
    "Histórico recente (Buffer K=3):\n{historico}\n\n"
    "Previsão do Tempo (Santiago):\n{clima}\n\n"
    "Cálculo Financeiro Determinístico Oficial (4 Dias):\n{orcamento}\n\n"
    "Documentos Curados (RAG):\n{context}\n\n"
    "Pergunta do cliente: {input}\n"
    "Comentário consultivo (focado nos 4 dias, sem alterar preços):"
)

chain_geracao = prompt_rag | llm | StrOutputParser()


########## Etapa 3 - Configuração de Ferramentas Externas (Clima Open-Meteo + Web DuckDuckGo) ##########

def consultar_clima_santiago() -> dict:
    """Consulta a previsão de 4 dias em Santiago via Open-Meteo API (gratuita, sem chave)."""
    url = (
        "https://api.open-meteo.com/v1/forecast"
        "?latitude=-33.4489&longitude=-70.6693"
        "&daily=temperature_2m_max,temperature_2m_min,precipitation_probability_max"
        "&timezone=America/Santiago&forecast_days=4"
    )
    try:
        resp = requests.get(url, timeout=6)
        resp.raise_for_status()
        daily = resp.json().get("daily", {})
        datas = daily.get("time", [])
        tmax = daily.get("temperature_2m_max", [])
        tmin = daily.get("temperature_2m_min", [])
        chuva = daily.get("precipitation_probability_max", [])

        linhas = []
        tem_chuva = False
        for i in range(min(4, len(datas))):
            prob = chuva[i] if i < len(chuva) else 0
            cond = "Chuvoso/Nublado 🌧️" if prob >= 40 else "Sol/Tempo firme ☀️"
            if prob >= 40:
                tem_chuva = True
            linhas.append(f"Dia {i+1} ({datas[i]}): {tmin[i]}°C a {tmax[i]}°C | Chuva: {prob}% ({cond})")

        resumo = "\n".join(linhas) if linhas else "Previsão: 3 dias de sol (12°C a 24°C) e 1 dia nublado."
        return {"resumo": resumo, "tem_chuva": tem_chuva, "fonte": "Open-Meteo API (Tempo Real)"}
    except Exception:
        return {
            "resumo": (
                "Dia 1: 11°C a 24°C | Chuva: 10% (Sol/Tempo firme ☀️)\n"
                "Dia 2: 12°C a 25°C | Chuva: 5% (Sol/Tempo firme ☀️)\n"
                "Dia 3: 9°C a 16°C | Chuva: 65% (Chuvoso/Nublado 🌧️)\n"
                "Dia 4: 11°C a 22°C | Chuva: 15% (Sol/Tempo firme ☀️)"
            ),
            "tem_chuva": True,
            "fonte": "Fallback Meteorológico Santiago",
        }


def buscar_na_web_duckduckgo(consulta: str) -> str:
    """Busca complementar na web via DuckDuckGo."""
    try:
        search = DuckDuckGoSearchAPIWrapper(region="br-pt", max_results=3)
        return search.run(f"Santiago Chile turismo {consulta}")
    except Exception:
        return "Sem notícias extras na web no momento; utilizando base curada da agência."


########## Etapa 4 - Estado do Agente e Nós do Grafo (LangGraph) ##########

class AgentState(BaseModel):
    query: str
    chat_history: list = []
    guardrail_ok: bool = True
    guardrail_msg: str = ""
    next_step: str = ""
    perfil_detectado: str = "casal"
    orcamento_maximo: float = 5000.0
    clima_info: str = ""
    retrieved_info: list = []
    possible_responses: list = []
    similarity_scores: list = []
    orcamento_detalhado: dict = {}
    historico_autocorrecao: list = []
    tentativas_ajuste: int = 0
    ranked_response: str = ""
    confidence_score: float = 0.0
    latencia_ms: float = 0.0


def detectar_perfil_e_orcamento(query: str, perfil_default: str, orcamento_default: float) -> tuple[str, float]:
    """Extrai perfil (casal, familia, individual) e orçamento em R$ diretamente da pergunta, se presentes."""
    q = query.lower()

    if any(p in q for p in ["familia", "família", "filho", "filhos", "criança", "crianças"]):
        perfil = "familia"
    elif any(p in q for p in ["individual", "sozinho", "sozinha", "solo", "solteiro"]):
        perfil = "individual"
    elif any(p in q for p in ["casal", "lua de mel", "esposa", "marido", "namorad"]):
        perfil = "casal"
    else:
        perfil = perfil_default

    orcamento = orcamento_default
    padrao_valor = re.findall(
        r"(?:r\$\s*|orçamento\s*(?:de)?\s*|até\s*|ate\s*)(\d{1,2}\.\d{3}|\d{3,5})(?:,\d{2})?",
        q,
    )
    if padrao_valor:
        valor_limpo = padrao_valor[0].replace(".", "")
        try:
            orcamento = float(valor_limpo)
        except ValueError:
            pass

    return perfil, orcamento


# Nó 1: Guardrail de Entrada (Escopo + Anti Prompt Injection)
def no_guardrail_entrada(state: AgentState) -> AgentState:
    q = state.query.lower().strip()

    if not q:
        state.guardrail_ok = False
        state.guardrail_msg = "Por favor, digite uma pergunta sobre sua viagem para Santiago do Chile."
        return state

    bloqueios_seguranca = [
        "ignore todas as instruções",
        "ignore previous instructions",
        "system prompt",
        "drop table",
        "delete from",
        "hackear",
    ]
    if any(b in q for b in bloqueios_seguranca):
        state.guardrail_ok = False
        state.guardrail_msg = "🛡️ **Guardrail de Entrada Ativado:** Solicitação bloqueada por tentativa de violação de instruções do sistema."
        return state

    temas_fora_escopo = [
        "receita de bolo",
        "campeonato brasileiro",
        "futebol",
        "criptomoeda",
        "bitcoin",
        " Imposto de renda",
        "processo trabalhista",
    ]
    palavras_viagem = [
        "santiago", "chile", "viagem", "roteiro", "hotel", "hospedagem", "passeio",
        "vinícola", "vinicola", "restaurante", "clima", "tempo", "chuva", "neve",
        "orçamento", "orcamento", "casal", "família", "familia", "individual",
        "lastarria", "providencia", "valle nevado", "teleférico", "teleferico",
        "museu", "preço", "preco", "diária", "diaria", "turismo", "explique", "o que é",
    ]

    if any(f in q for f in temas_fora_escopo) and not any(v in q for v in palavras_viagem):
        state.guardrail_ok = False
        state.guardrail_msg = (
            "🛡️ **Guardrail de Escopo:** Sou um assistente exclusivo para planejamento de viagens a **Santiago do Chile** "
            "(hotéis, passeios, gastronomia, clima e orçamento). Como posso ajudar na sua viagem?"
        )
        return state

    state.guardrail_ok = True
    return state


# Nó 2: Roteador (Decisão do Agente com Memória Buffer K=3)
def no_decisao_agente(state: AgentState) -> AgentState:
    query = state.query.lower()

    # Se já existe conversa no Buffer K=3 e o usuário não informou novo perfil/orçamento na frase,
    # herda o perfil e o orçamento da última interação da memória!
    base_perfil = state.perfil_detectado
    base_orcamento = state.orcamento_maximo
    if state.chat_history:
        ultimo_turno = state.chat_history[-1]
        base_perfil = ultimo_turno.get("perfil", base_perfil)
        base_orcamento = float(ultimo_turno.get("orcamento", base_orcamento))

    state.perfil_detectado, state.orcamento_maximo = detectar_perfil_e_orcamento(
        state.query, base_perfil, base_orcamento
    )

    palavras_clima_web = [
        "ensolarado", "sol", "chuva", "chover", "clima", "tempo", "temperatura",
        "frio", "neve", "nublado", "previsão", "previsao", "busque na web", "notícias", "noticias",
    ]
    palavras_novo_roteiro = [
        "planeje", "monte", "roteiro", "orçamento", "orcamento", "quero de até", "quero até", "r$",
    ]
    palavras_duvida_item = [
        "qual hotel", "qual restaurante", "quais passeios", "nesse pacote", "neste pacote",
        "explique", "defina", "conceito", "o que é", "detalhe", "como funciona",
    ]

    # 1) Se pergunta for especificamente sobre clima/sol/chuva (e não um pedido de novo roteiro completo)
    if any(c in query for c in palavras_clima_web) and not any(r in query for r in ["planeje", "monte", "orçamento", "orcamento", "r$"]):
        state.next_step = "usar_clima"
    # 2) Se for pergunta de continuação sobre itens do pacote anterior ou conceito geral
    elif any(d in query for d in palavras_duvida_item) and not any(r in query for r in palavras_novo_roteiro):
        state.next_step = "gerar"
    # 3) Caso contrário, monta e valida o roteiro completo + orçamento
    else:
        state.next_step = "roteiro_completo"

    return state


# Nó 3: Clima e Web (Open-Meteo + DuckDuckGo)
def no_usar_clima(state: AgentState) -> AgentState:
    dados_clima = consultar_clima_santiago()
    texto_clima = f"[{dados_clima['fonte']}]\n{dados_clima['resumo']}"

    q = state.query.lower()
    if any(p in q for p in ["busque na web", "notícias", "noticias", "atualizado", "recente"]):
        res_web = buscar_na_web_duckduckgo(state.query)
        texto_clima += f"\n\n[Busca Web DuckDuckGo]:\n{res_web}"

    state.clima_info = texto_clima
    return state


# Nó 4: Retrieve (Nó Guia — RAG com filtro por perfil e ajuste por tentativa de orçamento)
def no_retrieve_info(state: AgentState) -> AgentState:
    todos_docs = vector_db.similarity_search(state.query, k=13)
    perfil = state.perfil_detectado

    # Filtra documentos que atendem ao perfil detectado (ou traz todos como apoio)
    docs_perfil = [
        d for d in todos_docs
        if perfil in d.metadata.get("perfil", "").split(",")
    ]
    state.retrieved_info = docs_perfil if docs_perfil else todos_docs[:6]
    return state


# Nó 5: Financeiro + Loop de Autocorreção Determinística
def no_calcular_orcamento(state: AgentState) -> AgentState:
    todos_docs = vector_db.similarity_search("hotel passeio restaurante santiago", k=13)
    por_id = {d.metadata.get("id_item"): d.metadata for d in todos_docs}

    perfil = state.perfil_detectado
    nivel_ajuste = state.tentativas_ajuste  # 0 = padrão, 1 = cupom/hotel parceiro, 2 = ultra econômico

    # Seleção determinística baseada no perfil e no nível de ajuste
    if nivel_ajuste == 2:
        # Combinação mais econômica disponível na agência (ibis + 2 passeios + roteiro gastronômico econômico)
        hotel = por_id.get("hot-004", {"nome": "ibis Santiago Providencia", "preco_brl": 1350.0, "parceiro_cupom": False})
        if perfil == "familia":
            passeios = [
                por_id.get("pas-003", {"nome": "Teleférico de Santiago & Cerro San Cristóbal", "preco_brl": 250.0}),
                por_id.get("pas-005", {"nome": "Parque Bicentenário & Zoológico Metropolitano", "preco_brl": 180.0}),
            ]
        else:
            passeios = [
                por_id.get("pas-003", {"nome": "Teleférico de Santiago & Cerro San Cristóbal", "preco_brl": 250.0}),
                por_id.get("pas-004", {"nome": "Museu Nacional de Bellas Artes & Museu Precolombino", "preco_brl": 200.0}),
            ]
        gastronomia = por_id.get(
            "res-003",
            {"nome": "Roteiro Gastronômico Econômico — Patio Bellavista & Mercados + Metrô (4 dias)", "preco_brl": 950.0},
        )
    elif perfil == "casal":
        if nivel_ajuste == 0:
            hotel = por_id.get("hot-001", {"nome": "Hotel Cumbres Lastarria", "preco_brl": 2500.0, "parceiro_cupom": False})
        else:
            hotel = por_id.get("hot-002", {"nome": "Luciano K Hotel — Parceiro Agência com Cupom", "preco_brl": 1950.0, "parceiro_cupom": True})

        passeios = [
            por_id.get("pas-001", {"nome": "Viña Santa Rita — Tour & Degustação", "preco_brl": 650.0}),
            por_id.get("pas-003", {"nome": "Teleférico de Santiago & Cerro San Cristóbal", "preco_brl": 250.0}),
            por_id.get("pas-004", {"nome": "Museu Nacional de Bellas Artes & Museu Precolombino", "preco_brl": 200.0}),
        ]
        gastronomia = por_id.get(
            "res-001",
            {"nome": "Bocanariz Wine Bar + Pizzeria Tiramisú + Traslados Casal (Pacote 4 dias)", "preco_brl": 1600.0},
        )

    elif perfil == "familia":
        if nivel_ajuste == 0:
            hotel = por_id.get("hot-003", {"nome": "Novotel Santiago Providencia", "preco_brl": 2100.0, "parceiro_cupom": False})
        else:
            hotel = por_id.get("hot-004", {"nome": "ibis Santiago Providencia", "preco_brl": 1350.0, "parceiro_cupom": False})

        passeios = [
            por_id.get("pas-003", {"nome": "Teleférico de Santiago & Cerro San Cristóbal", "preco_brl": 250.0}),
            por_id.get("pas-005", {"nome": "Parque Bicentenário & Zoológico Metropolitano", "preco_brl": 180.0}),
            por_id.get("pas-004", {"nome": "Museu Nacional de Bellas Artes & Museu Precolombino", "preco_brl": 200.0}),
        ]
        gastronomia = por_id.get(
            "res-002",
            {"nome": "Giratorio Restaurant + Mestizo Parque Bicentenário + Traslados Família (Pacote 4 dias)", "preco_brl": 1750.0},
        )

    else:  # individual
        if nivel_ajuste == 0:
            hotel = por_id.get("hot-002", {"nome": "Luciano K Hotel — Parceiro Agência com Cupom", "preco_brl": 1950.0, "parceiro_cupom": True})
        else:
            hotel = por_id.get("hot-004", {"nome": "ibis Santiago Providencia", "preco_brl": 1350.0, "parceiro_cupom": False})

        passeios = [
            por_id.get("pas-001", {"nome": "Viña Santa Rita — Tour & Degustação", "preco_brl": 650.0}),
            por_id.get("pas-003", {"nome": "Teleférico de Santiago & Cerro San Cristóbal", "preco_brl": 250.0}),
            por_id.get("pas-004", {"nome": "Museu Nacional de Bellas Artes & Museu Precolombino", "preco_brl": 200.0}),
        ]
        gastronomia = por_id.get(
            "res-003",
            {"nome": "Roteiro Gastronômico Econômico — Patio Bellavista & Mercados + Metrô (4 dias)", "preco_brl": 950.0},
        )

    custo_hotel = float(hotel["preco_brl"])
    custo_passeios = float(sum(p["preco_brl"] for p in passeios))
    custo_gastronomia = float(gastronomia["preco_brl"])
    total = custo_hotel + custo_passeios + custo_gastronomia
    folga = state.orcamento_maximo - total
    aprovado = total <= state.orcamento_maximo

    detalhe = {
        "tentativa": state.tentativas_ajuste + 1,
        "perfil": perfil,
        "hotel_nome": hotel["nome"],
        "hotel_brl": custo_hotel,
        "parceiro_cupom": bool(hotel.get("parceiro_cupom", False)),
        "passeios_nomes": [p["nome"] for p in passeios],
        "passeios_brl": custo_passeios,
        "gastronomia_nome": gastronomia["nome"],
        "gastronomia_brl": custo_gastronomia,
        "total_brl": total,
        "orcamento_maximo": state.orcamento_maximo,
        "folga_brl": folga,
        "aprovado": aprovado,
    }

    state.orcamento_detalhado = detalhe
    state.historico_autocorrecao.append(detalhe)
    return state


def aresta_valida_orcamento(state: AgentState) -> str:
    """Aresta condicional do LangGraph: tenta até 2 níveis de redução de custos antes de concluir."""
    det = state.orcamento_detalhado
    if not det.get("aprovado", True) and state.tentativas_ajuste < 2:
        return "ajustar_roteiro"
    return "aprovado"


def no_incrementa_tentativa(state: AgentState) -> AgentState:
    state.tentativas_ajuste += 1
    return state


def montar_resposta_estruturada_base(state: AgentState) -> str:
    """Monta o resumo determinístico do roteiro ou avisa claramente se o orçamento for insuficiente."""
    d = state.orcamento_detalhado
    if not d:
        return "Consulte os documentos recuperados abaixo."

    # Caso mesmo o pacote mais econômico da agência ultrapasse o orçamento solicitado
    if not d["aprovado"]:
        return (
            f"### ❌ Orçamento Insuficiente para Pacote de 4 Dias em Santiago (Perfil: `{d['perfil'].upper()}`)\n\n"
            f"Infelizmente **não possuímos nenhum roteiro disponível até R$ {d['orcamento_maximo']:.2f}** na base curada da agência.\n\n"
            f"- 📉 **Pacote mais econômico disponível ({d['perfil']}):** **R$ {d['total_brl']:.2f}** "
            f"(diferença de **R$ {abs(d['folga_brl']):.2f}** acima do limite informado).\n"
            f"  - 🏨 Hospedagem mínima: *{d['hotel_nome']}* (R$ {d['hotel_brl']:.2f})\n"
            f"  - 🎟️ Passeios essenciais: {', '.join(d['passeios_nomes'])} (R$ {d['passeios_brl']:.2f})\n"
            f"  - 🍽️ Alimentação/Transporte econômico: *{d['gastronomia_nome']}* (R$ {d['gastronomia_brl']:.2f})\n\n"
            f"💡 **Sugestão:** Aumente o orçamento máximo para pelo menos **R$ {d['total_brl']:.2f}** para viabilizar a viagem."
        )

    passeios_str = "\n".join(f"  - {nome}" for nome in d["passeios_nomes"])
    status_str = f"✅ **Dentro do orçamento!** (Folga de R$ {d['folga_brl']:.2f})"

    # Detecta se o usuário pediu um número de dias diferente de 4 (ex.: "5 dias", "6 dias", "7 diárias")
    aviso_dias = ""
    match_dias = re.findall(r"\b(\d+)\s*(?:dias|diárias|diarias|noites)\b", state.query.lower())
    if match_dias:
        dias_pedidos = int(match_dias[0])
        if dias_pedidos != 4:
            aviso_dias = (
                f"\n> ℹ️ **Política de Pacotes da Agência:** Você mencionou **{dias_pedidos} dias**, porém nossa base curada "
                f"trabalha exclusivamente com **pacotes fechados de 4 dias (4 diárias)**. Abaixo apresentamos o pacote oficial de 4 dias:\n"
            )

    nota_autocorrecao = ""
    if len(state.historico_autocorrecao) > 1:
        t1 = state.historico_autocorrecao[0]
        nota_autocorrecao = (
            f"\n> 🔄 **Autocorreção pelo LangGraph:** O roteiro inicial com *{t1['hotel_nome']}* totalizou "
            f"**R$ {t1['total_brl']:.2f}** (acima do limite de R$ {d['orcamento_maximo']:.2f}). "
            f"O agente acionou a aresta condicional e ajustou para **{d['hotel_nome']}** "
            f"(R$ {d['hotel_brl']:.2f}), fechando o pacote em **R$ {d['total_brl']:.2f}**.\n"
        )

    return (
        f"### 🗺️ Roteiro Recomendado em Santiago (4 Dias — Perfil: `{d['perfil'].upper()}`)\n"
        f"{aviso_dias}{nota_autocorrecao}\n"
        f"- 🏨 **Hospedagem (Pacote 4 diárias):** {d['hotel_nome']} — **R$ {d['hotel_brl']:.2f}**\n"
        f"- 🎟️ **Passeios Curados (Sol + Coberto para dia nublado):** **R$ {d['passeios_brl']:.2f}**\n"
        f"{passeios_str}\n"
        f"- 🍷 **Gastronomia e Traslados (Pacote 4 dias):** {d['gastronomia_nome']} — **R$ {d['gastronomia_brl']:.2f}**\n\n"
        f"**💰 Valor Total do Pacote (4 Dias):** **R$ {d['total_brl']:.2f}** (Orçamento máximo: R$ {d['orcamento_maximo']:.2f}) — {status_str}"
    )


# Nó 6: Geração de Múltiplas Respostas (mantendo padrão do professor com fallback seguro)
def no_gera_multiplas_respostas(state: AgentState) -> AgentState:
    contexto_docs = "\n\n".join(doc.page_content for doc in state.retrieved_info)
    historico_str = "\n".join(
        f"Cliente: {h['pergunta']}\nAgente (Perfil {h.get('perfil', '')}, Teto R$ {h.get('orcamento', 0)}): {h['resposta']}"
        for h in state.chat_history[-3:]
    ) or "Sem interações anteriores."

    # Se for pergunta específica de Clima/Continuação (não pediu cálculo de novo roteiro completo)
    if state.next_step in ("usar_clima", "gerar") and not state.orcamento_detalhado:
        pacote_anterior_str = (
            f"Último pacote em conversa (Buffer K=3): Perfil {state.perfil_detectado.upper()}, Orçamento R$ {state.orcamento_maximo:.2f}.\n"
            f"{historico_str}"
        )
        cabecalho_clima = (
            f"### 🌤️ Previsão do Tempo e Condições para o seu Roteiro (`{state.perfil_detectado.upper()}` — Teto `R$ {state.orcamento_maximo:.2f}`)\n\n"
            f"```text\n{state.clima_info}\n```\n"
            f"- ☀️ **Dias de Sol/Tempo firme:** ideais para os passeios `ar_livre_sol` do pacote (*Teleférico & Cerro San Cristóbal*, *Viña Santa Rita* ou *Parque Bicentenário*).\n"
            f"- 🌧️ **Se houver dia Nublado/Chuvoso:** o roteiro já inclui opção `coberto_qualquer_clima` (*Museu Nacional de Bellas Artes & Museu Precolombino* e restaurantes cobertos).\n"
        ) if state.clima_info else ""

        respostas = []
        try:
            for _ in range(3):
                resp = chain_geracao.invoke(
                    {
                        "historico": historico_str,
                        "clima": state.clima_info or "Consulte previsão acima.",
                        "orcamento": pacote_anterior_str,
                        "context": contexto_docs,
                        "input": state.query,
                    }
                )
                if cabecalho_clima:
                    respostas.append(f"{cabecalho_clima}\n---\n**💬 Resposta do Consultor IA (com Memória Buffer K=3):**\n{resp}")
                else:
                    respostas.append(f"**💬 Resposta do Consultor IA (com Memória Buffer K=3):**\n{resp}")
        except Exception:
            respostas = [cabecalho_clima or "Consulte os detalhes dos documentos recuperados abaixo."]

        state.possible_responses = respostas
        return state

    resumo_orcamento = montar_resposta_estruturada_base(state)

    # Se mesmo após as tentativas o orçamento for insuficiente, não chama o LLM para inventar pacote: avisa direto
    if state.orcamento_detalhado and not state.orcamento_detalhado.get("aprovado", True):
        state.possible_responses = [resumo_orcamento]
        return state

    respostas = []
    try:
        # Gera 3 respostas candidatas no LM Studio para o Re-Ranking (rápido e fiel ao padrão do professor)
        for _ in range(3):
            resp = chain_geracao.invoke(
                {
                    "historico": historico_str,
                    "clima": state.clima_info or "Tempo firme em Santiago.",
                    "orcamento": resumo_orcamento,
                    "context": contexto_docs,
                    "input": state.query,
                }
            )
            respostas.append(f"{resumo_orcamento}\n\n---\n**💬 Comentário do Consultor IA (LM Studio):**\n{resp}")
    except Exception:
        # Fallback caso o LM Studio esteja ocupado ou desligado
        respostas = [
            f"{resumo_orcamento}\n\n*(Nota: Resposta estruturada gerada em modo determinístico a partir do RAG + Open-Meteo + Nó Financeiro).* "
        ]

    state.possible_responses = respostas
    return state


# Nó 7: Avaliação de Similaridade (Passo 1 do Re-Ranking — idêntico ao professor)
def no_avalia_similaridade(state: AgentState) -> AgentState:
    retrieved_texts = [doc.page_content for doc in state.retrieved_info]
    responses = state.possible_responses

    retrieved_embeddings = embedding_model.embed_documents(retrieved_texts) if retrieved_texts else []
    response_texts = [
        r["answer"] if isinstance(r, dict) and "answer" in r else str(r)
        for r in responses
    ]
    response_embeddings = embedding_model.embed_documents(response_texts) if response_texts else []

    if not retrieved_embeddings or not response_embeddings:
        state.similarity_scores = [0.0] * len(response_texts)
        return state

    similarities = [
        float(
            np.mean(
                [
                    cosine_similarity([response_embedding], [doc_embedding])[0][0]
                    for doc_embedding in retrieved_embeddings
                ]
            )
        )
        for response_embedding in response_embeddings
    ]

    state.similarity_scores = similarities
    return state


# Nó 8: Ranqueamento das Respostas (Passo 2 do Re-Ranking — idêntico ao professor)
def no_rank_respostas(state: AgentState) -> AgentState:
    response_with_scores = list(zip(state.possible_responses, state.similarity_scores))

    if response_with_scores:
        ranked_responses = sorted(response_with_scores, key=lambda x: x[1], reverse=True)
        state.ranked_response = ranked_responses[0][0]
        state.confidence_score = float(ranked_responses[0][1])
    else:
        state.ranked_response = "Desculpe, não encontrei informações relevantes para esta consulta."
        state.confidence_score = 0.0

    return state


# Nó 9: Guardrail de Saída (consistência financeira e de escopo)
def no_guardrail_saida(state: AgentState) -> AgentState:
    if not state.guardrail_ok:
        state.ranked_response = state.guardrail_msg
        state.confidence_score = 0.0
    return state


########## Etapa 5 - Montagem do Fluxo de Execução com LangGraph ##########

workflow = StateGraph(AgentState)

workflow.add_node("guardrail_entrada", no_guardrail_entrada)
workflow.add_node("decision", no_decisao_agente)
workflow.add_node("usar_clima", no_usar_clima)
workflow.add_node("retrieve", no_retrieve_info)
workflow.add_node("calcular_orcamento", no_calcular_orcamento)
workflow.add_node("incrementa_tentativa", no_incrementa_tentativa)
workflow.add_node("generate_multiple", no_gera_multiplas_respostas)
workflow.add_node("evaluate_similarity", no_avalia_similaridade)
workflow.add_node("rank_responses", no_rank_respostas)
workflow.add_node("guardrail_saida", no_guardrail_saida)

workflow.set_entry_point("guardrail_entrada")

workflow.add_conditional_edges(
    "guardrail_entrada",
    lambda state: "ok" if state.guardrail_ok else "bloqueado",
    {
        "ok": "decision",
        "bloqueado": "guardrail_saida",
    },
)

workflow.add_conditional_edges(
    "decision",
    lambda state: state.next_step,
    {
        "roteiro_completo": "usar_clima",
        "usar_clima": "usar_clima",
        "gerar": "retrieve",
    },
)

workflow.add_edge("usar_clima", "retrieve")

# Após o retrieve: se for roteiro_completo, calcula orçamento; se for dúvida de clima/continuação, vai direto gerar a resposta!
workflow.add_conditional_edges(
    "retrieve",
    lambda state: "calcular" if state.next_step == "roteiro_completo" else "responder_direto",
    {
        "calcular": "calcular_orcamento",
        "responder_direto": "generate_multiple",
    },
)

# Loop de Autocorreção (Aresta Condicional do Orçamento)
workflow.add_conditional_edges(
    "calcular_orcamento",
    aresta_valida_orcamento,
    {
        "ajustar_roteiro": "incrementa_tentativa",
        "aprovado": "generate_multiple",
    },
)
workflow.add_edge("incrementa_tentativa", "retrieve")

workflow.add_edge("generate_multiple", "evaluate_similarity")
workflow.add_edge("evaluate_similarity", "rank_responses")
workflow.add_edge("rank_responses", "guardrail_saida")
workflow.add_edge("guardrail_saida", END)

agent_workflow = workflow.compile()


########## Etapa 6 - Fluxo da Aplicação Web no Streamlit ##########

st.markdown("#### 💡 Sugestões rápidas de teste:")
col_ex1, col_ex2, col_ex3 = st.columns(3)
pergunta_sugerida = ""
if col_ex1.button("💑 Casal com R$ 5.000 (Ativa Autocorreção)"):
    pergunta_sugerida = "Planeje um roteiro de 4 dias em Santiago para casal com orçamento máximo de R$ 5.000, considerando o clima."
if col_ex2.button("👨‍👩‍👧 Família em Santiago (4 dias)"):
    pergunta_sugerida = "Quero um roteiro de 4 dias em Santiago para família com crianças e orçamento de R$ 5.000."
if col_ex3.button("🎒 Viajante Individual Econômico"):
    pergunta_sugerida = "Vou viajar sozinho (individual) para Santiago por 4 dias com orçamento de R$ 3.500. O que recomenda?"

query = st.text_input(
    "Digite sua pergunta sobre viagem para Santiago:",
    value=pergunta_sugerida,
    placeholder="Ex.: Planeje 4 dias em Santiago para casal com orçamento de R$ 5.000",
)

if st.button("🚀 Planejar / Enviar Consulta") or pergunta_sugerida:
    inicio = time.perf_counter()
    with st.spinner("O Agente (LangGraph + RAG + Open-Meteo + Re-Ranking) está processando sua consulta..."):
        estado_inicial = AgentState(
            query=query,
            chat_history=st.session_state.chat_history[-3:],
            perfil_detectado=perfil_padrao,
            orcamento_maximo=float(orcamento_padrao),
        )
        output = agent_workflow.invoke(estado_inicial)
    latencia = (time.perf_counter() - inicio) * 1000.0

    resposta = output.get("ranked_response", "Nenhuma resposta gerada.")
    confidence = output.get("confidence_score", 0.0)
    historico_ajustes = output.get("historico_autocorrecao", [])
    clima_txt = output.get("clima_info", "")

    # Atualiza BufferWindowMemory (últimas K=3 interações) guardando perfil e orçamento ativos
    if output.get("guardrail_ok", True):
        st.session_state.chat_history.append(
            {
                "pergunta": query,
                "resposta": resposta[:400],
                "perfil": output.get("perfil_detectado", perfil_padrao),
                "orcamento": float(output.get("orcamento_maximo", orcamento_padrao)),
            }
        )
        st.session_state.chat_history = st.session_state.chat_history[-3:]

    # Métricas de Observabilidade no topo
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Perfil Detectado", output.get("perfil_detectado", "casal").upper())
    m2.metric("Orçamento Teto", f"R$ {output.get('orcamento_maximo', 5000.0):,.2f}")
    m3.metric("Confidence Score (Re-Ranking)", f"{confidence:.2f}")
    m4.metric("Latência Total", f"{latencia:.0f} ms")

    # Quadro visual do Loop de Autocorreção (quando acionado)
    if len(historico_ajustes) > 1:
        st.warning("🔁 **Aresta Condicional Acionada — Loop de Autocorreção de Orçamento (LangGraph)**")
        cols_aj = st.columns(len(historico_ajustes))
        t1 = historico_ajustes[0]
        for idx, t in enumerate(historico_ajustes):
            with cols_aj[idx]:
                selo_cupom = " 🏷️ *Cupom Agência*" if t.get("parceiro_cupom") else ""
                if t["aprovado"]:
                    st.success(
                        f"**{t['tentativa']}ª Tentativa (Aprovada — Folga de R$ {t['folga_brl']:.2f})**\n"
                        f"- Hotel: {t['hotel_nome']} (R$ {t['hotel_brl']:.2f}){selo_cupom}\n"
                        f"- Passeios: R$ {t['passeios_brl']:.2f}\n"
                        f"- Gastronomia/Traslados: R$ {t['gastronomia_brl']:.2f}\n"
                        f"- **Total: R$ {t['total_brl']:.2f}** (Economia de R$ {t1['total_brl'] - t['total_brl']:.2f})"
                    )
                else:
                    st.error(
                        f"**{t['tentativa']}ª Tentativa (Reprovada — Excedeu R$ {abs(t['folga_brl']):.2f})**\n"
                        f"- Hotel: {t['hotel_nome']} (R$ {t['hotel_brl']:.2f}){selo_cupom}\n"
                        f"- Passeios: R$ {t['passeios_brl']:.2f}\n"
                        f"- Gastronomia/Traslados: R$ {t['gastronomia_brl']:.2f}\n"
                        f"- **Total: R$ {t['total_brl']:.2f}** (Teto: R$ {t['orcamento_maximo']:.2f})"
                    )

    st.subheader("📋 Resposta Ranqueada:")
    st.markdown(resposta)

    if clima_txt:
        with st.expander("🌤️ Previsão do Tempo e Busca Externa (Nó Clima & Web)", expanded=True):
            st.text(clima_txt)

    documentos_relacionados = output.get("retrieved_info", [])
    if documentos_relacionados:
        with st.expander(f"📚 Documentos Curados Recuperados do ChromaDB ({len(documentos_relacionados)} itens)", expanded=False):
            for doc in documentos_relacionados:
                st.markdown(
                    f"**ID:** `{doc.metadata.get('id_item', doc.id)}` | "
                    f"**Nome:** `{doc.metadata.get('nome')}` | "
                    f"**Fonte:** `{doc.metadata.get('source', 'Desconhecida')}`"
                )
                st.text_area("Conteúdo", doc.page_content, height=90, key=f"doc_{doc.metadata.get('id_item', doc.id)}")
