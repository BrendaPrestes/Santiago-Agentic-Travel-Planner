# Planejador Inteligente de Viagens para Santiago — Worker de Atualização de Preços e Reindexação
# Consulta cotação CLP -> BRL em API pública gratuita, atualiza os JSONs e reindexa o ChromaDB

import os
import sys
import json
import argparse
from datetime import datetime
import requests
from cria_rag import cria_vectordb, DIRETORIO_DADOS

URL_CAMBIO_CLP = "https://open.er-api.com/v6/latest/CLP"
TAXA_CLP_POR_BRL_PADRAO = 170.0  # Fallback: ~170 CLP = R$ 1,00


def consultar_cambio_clp_brl() -> dict:
    """Consulta a cotação atual de CLP para BRL via API pública gratuita (sem chave)."""
    try:
        resp = requests.get(URL_CAMBIO_CLP, timeout=6)
        resp.raise_for_status()
        dados = resp.json()
        brl_por_clp = float(dados["rates"]["BRL"])
        clp_por_brl = round(1.0 / brl_por_clp, 2) if brl_por_clp > 0 else TAXA_CLP_POR_BRL_PADRAO
        return {
            "brl_por_clp": brl_por_clp,
            "clp_por_brl": clp_por_brl,
            "fonte": "API open.er-api.com (Tempo Real)",
        }
    except Exception:
        return {
            "brl_por_clp": round(1.0 / TAXA_CLP_POR_BRL_PADRAO, 6),
            "clp_por_brl": TAXA_CLP_POR_BRL_PADRAO,
            "fonte": "Fallback Câmbio Padrão (Offline)",
        }


def executar_worker(modo_live: bool = False) -> dict:
    """
    Atualiza os arquivos JSON em dados_origem/ e reindexa o ChromaDB.
    - modo_live=False (Padrão / Modo Demonstração): consulta o câmbio real e atualiza `ultima_atualizacao`,
      preservando os valores de `preco_brl` do caso base (R$ 5.200 -> R$ 4.650).
    - modo_live=True: recalcula `preco_brl` a partir de `preco_clp_base` usando o câmbio do dia.
    """
    cambio = consultar_cambio_clp_brl()
    data_hoje = datetime.now().strftime("%Y-%m-%d %H:%M")
    arquivos_atualizados = 0
    itens_atualizados = 0

    if os.path.exists(DIRETORIO_DADOS):
        for nome_arq in sorted(os.listdir(DIRETORIO_DADOS)):
            if not nome_arq.endswith(".json"):
                continue
            caminho = os.path.join(DIRETORIO_DADOS, nome_arq)
            with open(caminho, "r", encoding="utf-8") as f:
                itens = json.load(f)

            for item in itens:
                item["ultima_atualizacao"] = data_hoje
                if modo_live:
                    clp_base = float(item.get("preco_clp_base", 0))
                    if clp_base > 0:
                        item["preco_brl"] = round(clp_base * cambio["brl_por_clp"], 2)
                itens_atualizados += 1

            with open(caminho, "w", encoding="utf-8") as f:
                json.dump(itens, f, ensure_ascii=False, indent=2)
            arquivos_atualizados += 1

    # Reindexa o banco vetorial ChromaDB com os dados validados
    cria_vectordb()

    resumo = {
        "modo": "LIVE (Preços Recalculados)" if modo_live else "DEMONSTRAÇÃO (Preços Base Preservados)",
        "cambio_clp_por_brl": cambio["clp_por_brl"],
        "fonte_cambio": cambio["fonte"],
        "ultima_atualizacao": data_hoje,
        "arquivos_atualizados": arquivos_atualizados,
        "itens_reindexados": itens_atualizados,
    }
    print(
        f"\n[Worker Concluído] Modo: {resumo['modo']} | "
        f"Câmbio: 1 BRL = {resumo['cambio_clp_por_brl']} CLP ({resumo['fonte_cambio']}) | "
        f"Itens reindexados: {resumo['itens_reindexados']} em {resumo['ultima_atualizacao']}\n"
    )
    return resumo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Worker de atualização de preços e reindexação do RAG Santiago.")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Recalcula preco_brl usando a cotação CLP->BRL do dia (por padrão usa o Modo Demonstração).",
    )
    args = parser.parse_args()
    executar_worker(modo_live=args.live)
