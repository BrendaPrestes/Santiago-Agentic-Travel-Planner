# Especificação de Design — Planejador Inteligente de Viagens para Santiago (Agentic RAG + Re-Ranking + LangGraph)

**Data:** 2026-11-16  
**Autora:** Brenda Prestes  
**Diretório do Projeto:** `D:\Projetos\planejador-viagens-santiago-agentic-rag`

---

## 1. Objetivo e Contexto

Adaptar a arquitetura base dos Capítulos 11, 12 e 13 do curso (*Re-Ranking e Agentic RAG com LangGraph e LM Studio*) para um sistema próprio de portfólio (GitHub/Currículo) focado no **planejamento inteligente de roteiros de viagem para Santiago do Chile**, refletindo fielmente o fluxo apresentado nos slides de estudo de caso (`Planejador Inteligente de Viagens com IA`).

O sistema resolve três limitações de um LLM isolado:
1. **Conhecimento curado sem alucinação (Nó Guia — RAG + Re-Ranking):** Consulta documentos reais da agência sobre **Hotéis, Passeios e Restaurantes em Santiago**, validados com `Pydantic` (`ChunkMeta`), indexados no **ChromaDB** com embeddings `BAAI/bge-base-en` e ranqueados por similaridade de cosseno.
2. **Dados em tempo real (Nó Clima e Web — API gratuita):** Consulta a previsão do tempo real em Santiago via **Open-Meteo API** (gratuita, sem necessidade de chave) e busca complementar via **DuckDuckGoSearchAPIWrapper**, ajustando recomendações para dias de sol (passeios ao ar livre) ou dias nublados/chuvosos (museus e locais cobertos).
3. **Precisão matemática e controle de orçamento (Nó Financeiro + Loop de Autocorreção):** Calcula o valor total do roteiro (`hospedagem + passeios + alimentação/traslados`) usando ferramenta matemática determinística e, caso ultrapasse o orçamento do cliente (ex.: R$ 5.000), aciona a **aresta condicional do LangGraph** para substituir itens (como aplicar o hotel parceiro em Lastarria com cupom da agência por R$ 1.950) até caber no orçamento.
4. **Atualização contínua (Worker de Preços):** Script dedicado (`worker_atualiza_precos.py`) que consulta gratuitamente a cotação atual do Peso Chileno (`CLP` → `BRL`) e variações de preços na web para atualizar os arquivos `.json` de `dados_origem/` e reindexar o banco vetorial **ChromaDB**.

---

## 2. Estrutura de Diretórios e Arquivos

```text
D:\Projetos\planejador-viagens-santiago-agentic-rag\
├── dados_origem/
│   ├── hoteis_santiago.json         # Hotéis reais em Santiago (Lastarria, Providencia, Las Condes)
│   ├── passeios_santiago.json       # Passeios reais (Vinícolas, Teleférico, Museus, Parques, Neve)
│   └── restaurantes_santiago.json   # Restaurantes reais (Bocanariz, Tiramisú, Giratorio, Mestizo) e traslados
├── rag/
│   └── chroma_db/                   # Banco vetorial persistido pelo ChromaDB
├── docs/
│   └── superpowers/
│       ├── specs/
│       │   └── 2026-11-16-santiago-agentic-rag-design.md
│       └── plans/
├── cria_rag.py                      # Ingestão dos JSONs, validação Pydantic (ChunkMeta), chunking e criação do ChromaDB
├── worker_atualiza_precos.py        # Worker gratuito (Câmbio CLP->BRL + busca web) que atualiza os JSONs e reindexa o RAG
├── executa_agentic_rag.py           # Aplicação Streamlit + Grafo LangGraph (Guardrails, Roteador, RAG+Re-Ranking, Clima, Financeiro)
├── requirements.txt                 # Dependências Python baseadas no projeto original
└── README.md                        # Documentação completa para portfólio no GitHub
```

---

## 3. Modelo de Dados e Contrato Pydantic (`ChunkMeta`)

Em conformidade com o **Slide 09** da apresentação, cada item dos arquivos em `dados_origem/` é validado na ingestão pelo contrato `ChunkMeta`:

```python
from typing import Literal
from pydantic import BaseModel, Field

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
```

### Itens Curados nos 3 Documentos JSON (`dados_origem/`)

1. **`hoteis_santiago.json` (`categoria: "hospedagem"`):**
   - **Hotel Cumbres Lastarria** (Lastarria, perfil `["casal"]`, faixa `alto`, pacote 4 diárias: **R$ 2.500**)
   - **Luciano K Hotel — Parceiro Agência com Cupom** (Lastarria, perfil `["casal"]`, faixa `medio`, `parceiro_cupom: true`, pacote 4 diárias: **R$ 1.950**)
   - **Novotel Santiago Providencia** (Providencia, perfil `["familia"]`, piscina kids e quartos amplos, faixa `medio`, pacote 4 diárias: **R$ 2.100**)
   - **ibis Santiago Providencia** (Providencia, perfil `["casal", "familia"]`, faixa `baixo`, pacote 4 diárias: **R$ 1.350**)

2. **`passeios_santiago.json` (`categoria: "passeio"`):**
   - **Viña Santa Rita — Tour & Degustação** (Vale do Maipo, `["casal"]`, `ar_livre_sol`, **R$ 650** casal)
   - **Viña Concha y Toro — Tour Marques** (Pirque, `["casal"]`, `ar_livre_sol`, **R$ 620** casal)
   - **Teleférico de Santiago & Cerro San Cristóbal** (Providencia, `["casal", "familia"]`, `ar_livre_sol`, **R$ 250**)
   - **Museu Nacional de Bellas Artes & Museu Precolombino** (Centro/Lastarria, `["casal", "familia"]`, `coberto_qualquer_clima`, **R$ 200** com transporte)
   - **Parque Bicentenário & Zoológico Metropolitano** (Vitacura/Bellavista, `["familia"]`, `ar_livre_sol`, **R$ 180**)
   - **Valle Nevado & Farellones — Dia na Neve** (Cordilheira dos Andes, `["casal", "familia"]`, `neve_inverno`, **R$ 850**)

3. **`restaurantes_santiago.json` (`categoria: "restaurante"`):**
   - **Bocanariz Wine Bar + Pizzeria Tiramisú + Traslados Casal (Pacote 4 dias)** (Lastarria / Las Condes, `["casal"]`, `coberto_qualquer_clima`, **R$ 1.600**)
   - **Giratorio Restaurant + Mestizo Parque Bicentenário + Traslados Família (Pacote 4 dias)** (Providencia / Vitacura, `["familia", "casal"]`, `coberto_qualquer_clima`, **R$ 1.750**)
   - **Roteiro Gastronômico Econômico — Patio Bellavista & Mercados + Metrô (4 dias)** (`["casal", "familia"]`, `coberto_qualquer_clima`, **R$ 950**)

*(Nota: A soma `Hotel Cumbres R$ 2.500 + Passeios Casal R$ 1.100 + Gastronomia/Traslados R$ 1.600 = R$ 5.200` excede o orçamento de R$ 5.000, enquanto a troca pelo `Hotel Parceiro Lastarria R$ 1.950 + R$ 1.100 + R$ 1.600 = R$ 4.650` deixa margem de R$ 350 — reproduzindo com exatidão matemática o exemplo dos Slides 02 e 05 da apresentação).*

---

## 4. Arquitetura do Grafo (`LangGraph`) e Componentes (`executa_agentic_rag.py`)

Mantemos a configuração do modelo original do professor:
- **LLM:** `ChatOpenAI(model_name="hermes-3-llama-3.2-3b@q6_k", openai_api_base="http://localhost:1234/v1", openai_api_key="lm-studio", temperature=0.3, max_tokens=512)`
- **Embeddings:** `HuggingFaceEmbeddings(model_name="BAAI/bge-base-en")`
- **Vector Store:** `Chroma(persist_directory="rag/chroma_db", embedding_function=embedding_model)`

### Estado do Agente (`AgentState`)
```python
class AgentState(BaseModel):
    query: str
    chat_history: list = []           # BufferWindowMemory (últimas K=3 interações)
    guardrail_ok: bool = True
    guardrail_msg: str = ""
    next_step: str = ""               # "roteiro_completo", "retrieve", "usar_clima", "calcular_orcamento", "gerar"
    perfil_detectado: str = "casal"   # "casal", "familia" ou "individual"
    orcamento_maximo: float = 5000.0
    clima_info: str = ""
    retrieved_info: list = []
    possible_responses: list = []
    similarity_scores: list = []
    orcamento_detalhado: dict = {}
    tentativas_ajuste: int = 0
    ranked_response: str = ""
    confidence_score: float = 0.0
```

### Nós do Grafo (`StateGraph(AgentState)`)
1. **`guardrail_entrada`**: Verifica se a pergunta está dentro do escopo (viagens, turismo, Santiago, Chile, clima, orçamento, hospedagem, restaurantes) e sem comandos maliciosos.
2. **`decision` (Roteador)**: Analisa a consulta:
   - Se pedir apenas conceito geral (`"explique"`, `"defina"`, `"o que é"`) → `"gerar"`
   - Se pedir consulta de clima/chuva/notícias atualizadas → `"usar_clima"`
   - Se pedir consulta ao guia/roteiro/orçamento → `"retrieve"`
3. **`usar_clima` (Nó Clima + Web)**:
   - Consulta a API gratuita **Open-Meteo** para Santiago (`latitude=-33.4489, longitude=-70.6693`) retornando temperatura e condição de chuva/sol dos próximos 4 dias, além de permitir busca complementar via `DuckDuckGoSearchAPIWrapper`.
4. **`retrieve` (Nó Guia — RAG)**:
   - Recupera os documentos relevantes do `ChromaDB` e aplica filtro/priorização conforme o perfil (`casal`, `familia` ou `individual`) e o clima.
5. **`calcular_orcamento` (Nó Financeiro + Loop de Autocorreção)**:
   - Soma os custos selecionados (`hotel + passeios + gastronomia_traslados`).
   - **Aresta condicional (`valida_orcamento`)**: Se `total > orcamento_maximo` e `tentativas_ajuste < 1`, retorna ao Nó Guia trocando o hotel padrão pelo **Hotel Parceiro com cupom da agência** (reduzindo de R$ 5.200 para R$ 4.650) e registra o histórico da autocorreção para exibir na tela do Streamlit.
6. **`generate_multiple` → `evaluate_similarity` → `rank_responses` (Re-Ranking)**:
   - Mantém o pipeline de geração e avaliação de similaridade de cosseno (`cosine_similarity`) entre os embeddings das respostas e os documentos recuperados do ChromaDB, selecionando a resposta de maior `confidence_score`.
7. **`guardrail_saida`**: Garante que a resposta final contém valores consistentes e respeita o orçamento solicitado.

---

## 5. Worker de Atualização de Preços (`worker_atualiza_precos.py`)

Script independente (e também acionável por botão na barra lateral do Streamlit):
1. Consulta a cotação real do dia `CLP -> BRL` via API pública aberta (`https://open.er-api.com/v6/latest/CLP`), com fallback seguro caso esteja sem internet.
2. Opcionalmente consulta variações via `DuckDuckGoSearchAPIWrapper`.
3. Recalcula/atualiza `preco_brl` e o carimbo `ultima_atualizacao` nos arquivos JSON de `dados_origem/` (preservando os valores de demonstração do caso base quando executado em modo demonstração ou aplicando a cotação real quando executado em modo live).
4. Chama a rotina de `cria_rag.py` para atualizar o banco vetorial `rag/chroma_db`.

---

## 6. Tratamento de Erros e Modo Offline/Fallback

- **LM Studio desligado:** Caso o usuário teste a interface do Streamlit sem o servidor do LM Studio ativo em `localhost:1234`, o sistema captura o erro de conexão (`ConnectionError`) de forma amigável, avisa que o LM Studio não foi detectado na porta 1234 e ainda exibe uma resposta estruturada baseada diretamente nos documentos recuperados do RAG + Clima + Cálculo Financeiro para que a demonstração nunca quebre ao vivo.
- **Sem internet na hora do Clima/Worker:** Possui fallback automático com dados meteorológicos típicos de Santiago e taxa de câmbio padrão.
