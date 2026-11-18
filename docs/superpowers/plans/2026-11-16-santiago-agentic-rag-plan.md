# Plano de Implementação — Planejador de Viagens Santiago (Agentic RAG)

Spec: `docs/superpowers/specs/2026-11-16-santiago-agentic-rag-design.md`
Base de código: Cap13 do curso (`dsa_cria_rag.py`, `dsa_executa_agentic_rag.py`, `requirements.txt`), copiada em
`C:\Users\kaykb\.gemini\antigravity\brain\771b69b8-2ebd-4164-a310-84a67f0fd220\scratch\cap13\`.

## Decisões já tomadas
- Reaproveitar a estrutura do professor (imports, LLM, embeddings, Chroma, re-ranking, Streamlit) e adicionar os nós novos por cima.
- `requirements.txt` do professor (versões fixas) + pacotes novos.
- LM Studio em `localhost:1234` (constante no topo do arquivo, fácil de trocar; **a confirmar com o usuário**).
- Dados: 3 JSONs em `dados_origem/` (**já criados**, 13 itens, formato `ChunkMeta`).

## O que muda em relação ao código do professor
| Professor | Nosso projeto |
|---|---|
| JSONLoader com jq (`to_entries`) | Lê JSON, valida cada item com Pydantic `ChunkMeta`, cria `Document` com metadados |
| Retriever simples | Retriever + filtro/priorização por `perfil` e `clima` |
| Decisão: gerar / usar_web / retrieve | + guardrail de entrada, + `usar_clima` (Open-Meteo), + `calcular_orcamento` |
| Sem cálculo | Soma determinística + aresta condicional de autocorreção (troca pelo hotel parceiro) |
| Sem guardrail de saída | `guardrail_saida` valida valores e orçamento |
| LLM `max_tokens=256`, IP `10.0.0.20` | `max_tokens=512`, `localhost:1234` |

## Tarefas (na ordem combinada)

### Tarefa 1 — `requirements.txt`
- Copiar o do professor e acrescentar: `requests` (já existe, 2.31.0). Nenhum pacote novo obrigatório (Open-Meteo e câmbio via `requests`).
- Verificar: `pip install -r requirements.txt` em ambiente conda `python=3.12`.

### Tarefa 2 — `cria_rag.py`
1. Definir `ChunkMeta` (Pydantic) conforme spec seção 3.
2. Ler `dados_origem/*.json`, validar cada item; item inválido gera erro claro com `id_item`.
3. Montar `Document(page_content=texto, metadata=...)` com nome, bairro, perfil, faixa, clima, preço, cupom etc. (`perfil` vira string, pois Chroma só aceita tipos simples).
4. Embeddings `BAAI/bge-base-en` com `normalize_embeddings=True`, CPU.
5. `Chroma.from_documents(..., persist_directory="rag/chroma_db")`. Expor função `cria_vectordb()` (o worker a reutiliza) e `if __name__ == "__main__"`.
- Verificar: rodar o script; contar 13 documentos no Chroma; fazer uma busca de teste ("hotel para casal em Lastarria").

### Tarefa 3 — `executa_agentic_rag.py`
Subpassos, cada um testável isoladamente:
1. Interface Streamlit (título Santiago, barra lateral com instruções e botão de atualizar preços).
2. LLM, embeddings, Chroma, retriever (config do professor).
3. `AgentState` conforme spec seção 4.
4. Nós: `guardrail_entrada`, `decision`, `usar_clima` (Open-Meteo + DuckDuckGo, fallback offline), `retrieve` (filtro por perfil/clima), `calcular_orcamento` + aresta `valida_orcamento`, `generate_multiple`, `evaluate_similarity`, `rank_responses`, `guardrail_saida`.
5. Montagem do `StateGraph` e arestas condicionais.
6. Tela: resposta, confiança, clima, tabela do orçamento, histórico de autocorreção, documentos relacionados.
7. Fallback se o LM Studio estiver desligado (resposta estruturada só com RAG + clima + cálculo).
- Verificar: o cenário do spec — casal, orçamento R$ 5.000 → total inicial R$ 5.200 → troca pelo hotel parceiro → R$ 4.650.

### Tarefa 4 — `worker_atualiza_precos.py`
1. Câmbio CLP→BRL via `https://open.er-api.com/v6/latest/CLP`, fallback fixo (170 CLP = R$ 1).
2. Modo demonstração (mantém preços do spec, só atualiza data) e modo live (recalcula `preco_brl` a partir de `preco_clp_base`).
3. Chamar `cria_vectordb()` para reindexar.
- Verificar: rodar nos dois modos; conferir JSONs e reindexação.

### Tarefa 5 — `README.md`
Visão geral, arquitetura (diagrama do grafo), instalação (conda + pip), como rodar (`python cria_rag.py`, `streamlit run executa_agentic_rag.py`), exemplos de perguntas, créditos ao curso.

## Perguntas ainda em aberto (vou perguntar antes de implementar)
1. LM Studio roda na mesma máquina? (se sim, `localhost:1234`)
2. O marcador de rastreabilidade (`APP_WATERMARK`) do professor entra no nosso app ou não?
3. Nome do projeto/título exibido no Streamlit.
