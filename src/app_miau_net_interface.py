r"""
Miau-net | Interface de demonstração do modelo de Machine Learning (interface com cores equilibradas e cartões).
Carrega os arquivos treinados no notebook; não treina novamente ao abrir.

Execute na raiz do projeto:
    .\.venv\Scripts\python.exe -m streamlit run src/app.py

Arquivos usados (a partir da raiz do projeto):
    modelos/modelo_logistico_balanceado.joblib
    modelos/preprocessador_class.joblib
    dados/original/animal-shelter-intakes-and-outcomes.csv
    assets/gato.png ou assets/gato.jpg      (foto do cabeçalho, opcional)
    .streamlit/config.toml                  (tema visual do Streamlit)
"""
import base64
from io import BytesIO
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.colors import LinearSegmentedColormap

BASE_DIR = Path(__file__).resolve().parent.parent
PASTA_MODELOS = BASE_DIR / "modelos"
CAMINHO_MODELO = PASTA_MODELOS / "modelo_logistico_balanceado.joblib"
CAMINHO_PREPROCESSADOR = PASTA_MODELOS / "preprocessador_class.joblib"
CAMINHOS_CSV = [
    BASE_DIR / "dados" / "original" / "animal-shelter-intakes-and-outcomes.csv",
    BASE_DIR / "animal-shelter-intakes-and-outcomes.csv",
]
CAMINHOS_FOTO = [
    BASE_DIR / "assets" / "gato.jpg", BASE_DIR / "assets" / "gato.png",
    Path(__file__).resolve().parent / "assets" / "gato.jpg",
    Path(__file__).resolve().parent / "assets" / "gato.png",
]

NUM = ["idade_na_entrada"]
CAT = ["Sex", "Intake Condition", "Intake Type", "Intake Subtype", "Primary Color", "Jurisdiction"]
COLUNAS_ENTRADA = NUM + CAT

NOME_COLUNA = {
    "idade_na_entrada": "Idade", "Sex": "Sexo", "Intake Condition": "Condição",
    "Intake Type": "Tipo de entrada", "Intake Subtype": "Subtipo",
    "Primary Color": "Cor", "Jurisdiction": "Jurisdição",
}
TRADUCAO = {
    "Sex": {"Male": "Macho", "Female": "Fêmea", "Neutered": "Macho castrado",
            "Spayed": "Fêmea castrada", "Unknown": "Desconhecido"},
    "Intake Condition": {"NORMAL": "Normal", "UNDER AGE/WEIGHT": "Filhote / abaixo do peso",
                         "ILL MILD": "Doente (leve)", "ILL MODERATETE": "Doente (moderado)",
                         "ILL SEVERE": "Doente (grave)", "INJURED  MILD": "Ferido (leve)",
                         "INJURED  MODERATE": "Ferido (moderado)", "INJURED  SEVERE": "Ferido (grave)",
                         "FRACTIOUS": "Arisco", "FERAL": "Feral (selvagem)"},
    "Intake Type": {"STRAY": "Encontrado na rua", "OWNER SURRENDER": "Entregue pelo tutor",
                    "WELFARE SEIZED": "Apreendido (maus-tratos)", "CONFISCATE": "Confiscado",
                    "RETURN": "Devolvido", "TRAP, NEUTER, RETURN": "Captura-castração-devolução",
                    "SAFE KEEP": "Guarda temporária", "QUARANTINE": "Quarentena"},
    "Intake Subtype": {"FIELD": "Recolhido na rua",
                       "TRAP": "Capturado em armadilha", "ABAN FIELD": "Abandonado na rua",
                       "ABAN SHLTR": "Abandonado no abrigo", "OWNER DIED": "Tutor faleceu",
                       "POLICE": "Polícia"},
    "Primary Color": {"BLACK": "Preto", "BRN TABBY": "Rajado marrom", "GRAY": "Cinza",
                      "GRAY TABBY": "Rajado cinza", "WHITE": "Branco",
                      "ORG TABBY": "Rajado laranja", "CALICO": "Tricolor"},
}
PRESETS = {
    "castrado": dict(titulo=":material/home: Gato castrado e saudável", legenda="Exemplo com pontuação alta",
                     sexo="Neutered", cond="NORMAL", tipo="OWNER SURRENDER", sub="OTC",
                     cor="BLACK", jur="LONG BEACH", idade=24.0, idade_desc=False),
    "filhote": dict(titulo=":material/favorite: Filhote saudável", legenda="Exemplo com pontuação intermediária",
                    sexo="Female", cond="NORMAL", tipo="STRAY", sub="OTC",
                    cor="BRN TABBY", jur="LONG BEACH", idade=2.0, idade_desc=False),
    "ferido": dict(titulo=":material/shield: Adulto com ferimento grave", legenda="Exemplo com pontuação baixa",
                   sexo="Male", cond="INJURED  SEVERE", tipo="STRAY", sub="FIELD",
                   cor="BLACK", jur="LONG BEACH", idade=48.0, idade_desc=False),
}

# Métricas registradas no notebook; não são recalculadas ao abrir a interface.
METRICAS_TESTE = {"Acurácia": 0.7994, "Precisão": 0.5702, "Recall": 0.7875, "F1": 0.6615}
# Linhas = classe real; colunas = classe prevista. Ordem: Outro desfecho, Adoção.
MATRIZ_TESTE = np.array([[3265, 799], [286, 1060]])
METRICAS_VALIDACAO = pd.DataFrame({
    "Acurácia": [0.8257, 0.8102],
    "Precisão": [0.6418, 0.5885],
    "Recall": [0.6776, 0.7883],
    "F1": [0.6592, 0.6739],
}, index=["Regressão Logística (padrão)", "Regressão Logística balanceada"])

# Paleta (igual ao tema em .streamlit/config.toml)
BG = "#EDF2F8"
TEXTO = "#263449"
SUAVE = "#5E7188"

st.set_page_config(page_title="Miau-net | Adoção de gatos", page_icon="🐱", layout="wide")


@st.cache_resource(show_spinner="Carregando o modelo treinado...")
def carregar_modelos():
    if not CAMINHO_MODELO.exists():
        raise FileNotFoundError(f"Não encontrei: {CAMINHO_MODELO}")
    if not CAMINHO_PREPROCESSADOR.exists():
        raise FileNotFoundError(f"Não encontrei: {CAMINHO_PREPROCESSADOR}")
    return joblib.load(CAMINHO_MODELO), joblib.load(CAMINHO_PREPROCESSADOR)


@st.cache_data(show_spinner="Lendo a base para preencher as opções...")
def carregar_gatos(conteudo_ou_caminho):
    if isinstance(conteudo_ou_caminho, bytes):
        df = pd.read_csv(BytesIO(conteudo_ou_caminho), encoding="utf-8-sig")
    else:
        df = pd.read_csv(conteudo_ou_caminho, encoding="utf-8-sig")
    gatos = df[df["Animal Type"] == "CAT"].copy()
    dob = pd.to_datetime(gatos["DOB"], errors="coerce")
    entrada = pd.to_datetime(gatos["Intake Date"], errors="coerce")
    gatos["idade_na_entrada"] = (entrada - dob).dt.days / 365.25
    gatos.loc[gatos["idade_na_entrada"] < 0, "idade_na_entrada"] = np.nan
    gatos = gatos.dropna(subset=["Outcome Type"]).copy()
    gatos["adotado"] = (gatos["Outcome Type"] == "ADOPTION").astype(int)
    return gatos[COLUNAS_ENTRADA + ["adotado"]].reset_index(drop=True)


def pct(valor):
    return f"{valor:.1%}".replace(".", ",")


def milhar(n):
    return f"{n:,}".replace(",", ".")


def rotulo(coluna):
    mapa = TRADUCAO.get(coluna, {})
    return lambda valor: mapa.get(valor, valor)


def nome_amigavel(nome_saida):
    nome_saida = nome_saida.replace("numerico__", "").replace("categorico__", "")
    if nome_saida.startswith("idade_na_entrada"):
        return "Idade"
    for coluna in CAT:
        prefixo = coluna + "_"
        if nome_saida.startswith(prefixo):
            valor = nome_saida[len(prefixo):]
            valor = TRADUCAO.get(coluna, {}).get(valor, valor)
            return f"{NOME_COLUNA[coluna]}: {valor}"
    return nome_saida


def prever(modelo, preprocessador, valores):
    linha = pd.DataFrame([valores], columns=COLUNAS_ENTRADA)
    linha_preparada = preprocessador.transform(linha)
    indice_adocao = list(modelo.classes_).index(1)
    pontuacao = float(modelo.predict_proba(linha_preparada)[0, indice_adocao])
    classe = int(modelo.predict(linha_preparada)[0])
    return pontuacao, classe


def preparar_opcoes(dados):
    opcoes = {}
    for coluna in CAT:
        serie = dados[coluna].fillna("Não informado").astype(str)
        valores = serie.value_counts().index.tolist()
        if "Não informado" not in valores:
            valores.append("Não informado")
        opcoes[coluna] = valores
    return opcoes


def valor_valido(coluna, desejado, opcoes):
    return desejado if desejado in opcoes[coluna] else opcoes[coluna][0]


def aplicar_preset(chave, opcoes):
    p = PRESETS[chave]
    st.session_state.update({
        "sexo": valor_valido("Sex", p["sexo"], opcoes),
        "cond": valor_valido("Intake Condition", p["cond"], opcoes),
        "tipo": valor_valido("Intake Type", p["tipo"], opcoes),
        "sub": valor_valido("Intake Subtype", p["sub"], opcoes),
        "cor": valor_valido("Primary Color", p["cor"], opcoes),
        "jur": valor_valido("Jurisdiction", p["jur"], opcoes),
        "idade": p["idade"], "idade_desc": p["idade_desc"],
    })


# ----------------------------------------------------------------------------
# Visual: CSS, ícones e blocos HTML
# ----------------------------------------------------------------------------
def html(texto):
    """Envia o HTML numa linha só (o Markdown trataria linhas recuadas como código)."""
    st.markdown(" ".join(linha.strip() for linha in texto.strip().splitlines()), unsafe_allow_html=True)


ICONES = {
    "paw": ('<ellipse cx="12" cy="16.2" rx="4.6" ry="3.8"/><circle cx="5.6" cy="11.2" r="2.1"/>'
            '<circle cx="9.4" cy="6.8" r="2.1"/><circle cx="14.6" cy="6.8" r="2.1"/><circle cx="18.4" cy="11.2" r="2.1"/>'),
    "bulb": '<path d="M9 18h6M10 21h4M12 3a6 6 0 0 0-3.5 10.9c.6.5 1 1.2 1 2V16h5v-.1c0-.8.4-1.5 1-2A6 6 0 0 0 12 3z"/>',
    "info": '<circle cx="12" cy="12" r="9"/><path d="M12 11v6M12 7.5v.5"/>',
    "check": '<path d="M5 12.5l4.5 4.5L19 7.5"/>',
    "target": '<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="5"/><circle cx="12" cy="12" r="1.5"/>',
    "refresh": '<path d="M20 11a8 8 0 1 0-2.3 5.7M20 4v7h-7"/>',
    "star": '<path d="M12 3l2.7 5.6 6.1.9-4.4 4.3 1 6.1L12 17l-5.4 2.9 1-6.1L3.2 9.5l6.1-.9z"/>',
    "db": '<ellipse cx="12" cy="6" rx="7" ry="3"/><path d="M5 6v12c0 1.7 3.1 3 7 3s7-1.3 7-3V6M5 12c0 1.7 3.1 3 7 3s7-1.3 7-3"/>',
    "code": '<path d="M8 7l-5 5 5 5M16 7l5 5-5 5"/>',
    "grid": '<rect x="4" y="4" width="16" height="16" rx="2"/><path d="M4 12h16M12 4v16"/>',
    "bars": '<path d="M5 20V10M12 20V4M19 20v-7"/>',
    "spark": '<path d="M12 3v5M12 16v5M3 12h5M16 12h5M6 6l3 3M15 15l3 3M18 6l-3 3M9 15l-3 3"/>',
}


def icone(nome, tam=22):
    if nome == "paw":
        return f'<svg width="{tam}" height="{tam}" viewBox="0 0 24 24" fill="currentColor">{ICONES[nome]}</svg>'
    return (f'<svg width="{tam}" height="{tam}" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
            f'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">{ICONES[nome]}</svg>')


def foto_cabecalho():
    for caminho in CAMINHOS_FOTO:
        if caminho.exists():
            tipo = "jpeg" if caminho.suffix.lower() in (".jpg", ".jpeg") else "png"
            dados = base64.b64encode(caminho.read_bytes()).decode()
            return f"url(data:image/{tipo};base64,{dados})"
    return None


CSS = """
<style>
.stApp{background:#EDF2F8;color:#263449}.block-container{padding-top:1.4rem;max-width:1320px}
header[data-testid="stHeader"]{background:transparent}
#MainMenu,footer{visibility:hidden}
.mn-hero{display:flex;flex-wrap:wrap;align-items:stretch;justify-content:space-between;min-height:176px;border-radius:22px;overflow:hidden;border:1px solid rgba(139,124,246,.22);background:linear-gradient(90deg,#E8EEF7 0%,#DCE6F3 60%,#DCDCF2 100%)}
.mn-hero-left{display:flex;align-items:center;gap:24px;padding:26px 32px;flex:1 1 560px}
.mn-logo{flex:0 0 92px;height:92px;border-radius:24px;display:flex;align-items:center;justify-content:center;color:#fff;background:linear-gradient(135deg,#6C83BD,#8B5CF6);box-shadow:0 10px 30px rgba(91,108,240,.35)}
.mn-title{margin:0;font-size:42px;line-height:1.1;font-weight:800;color:#24364D;letter-spacing:-.01em}
.mn-title span,.mn-grad{background:linear-gradient(90deg,#7A72C6,#9B8BDD);-webkit-background-clip:text;background-clip:text;color:transparent}
.mn-sub{margin:10px 0 0;font-size:16px;line-height:1.5;color:#52657D;max-width:720px}
.mn-hero-right{display:flex;flex:0 1 560px;min-height:176px}
.mn-photo{flex:1 1 auto;background-size:cover;background-position:center;-webkit-mask-image:linear-gradient(to right,transparent,#000 45%);mask-image:linear-gradient(to right,transparent,#000 45%)}
.mn-tag{flex:0 0 190px;display:flex;align-items:center;justify-content:center;padding:0 18px;text-align:center;font-family:"Segoe Script","Bradley Hand","Brush Script MT",cursive;font-size:22px;line-height:1.25;color:#24364D;transform:rotate(-6deg);background:linear-gradient(135deg,rgba(139,92,246,.25),rgba(91,108,240,.05))}
@media (max-width:1100px){.mn-hero-right{display:none}}
.mn-head{display:flex;align-items:center;gap:14px;margin:2px 0 4px}
.mn-head-ico{width:36px;height:36px;border-radius:10px;display:flex;align-items:center;justify-content:center;color:#EDF2F8;background:linear-gradient(135deg,#7FE3F7,#7B70C2)}
.mn-head h3{margin:0;font-size:19px;font-weight:700;color:#24364D}
.mn-head p{margin:2px 0 0;font-size:14px;color:#5E7188}
.mn-result{border-radius:20px;padding:22px;border:1px solid rgba(45,212,168,.22);background:linear-gradient(160deg,#E4F3F0 0%,#F7F9FC 70%);display:flex;flex-direction:column;gap:18px}
.mn-result h2{margin:0;display:flex;align-items:center;gap:12px;font-size:26px;font-weight:700;color:#24364D}
.mn-result h2 svg{color:#2F9D8F}
.mn-score{border-radius:16px;padding:20px;border:1px solid rgba(45,212,168,.25);background:rgba(10,20,40,.45)}
.mn-score-top{display:flex;align-items:center;gap:18px}
.mn-score-ico{flex:0 0 68px;height:68px;border-radius:50%;display:flex;align-items:center;justify-content:center;color:#fff;background:linear-gradient(135deg,#278F83,#2F9D8F)}
.mn-score-lab{font-size:16px;color:#CFE0F5}
.mn-score-val{font-size:44px;line-height:1.1;font-weight:800;color:#fff}
.mn-bar{height:12px;border-radius:999px;margin-top:16px;background:#D7E1EE;overflow:hidden}
.mn-bar div{height:12px;border-radius:999px;background:linear-gradient(90deg,#2F9D8F,#72CBB9)}
.mn-verdict{display:flex;align-items:center;gap:16px;border-radius:16px;padding:18px 20px;font-size:17px;line-height:1.4;color:#B9CCF2}
.mn-verdict b{display:block;color:#fff;font-size:18px}
.mn-verdict-ico{flex:0 0 46px;height:46px;border-radius:50%;display:flex;align-items:center;justify-content:center}
.mn-warn{border:1px solid #B97B27;background:rgba(224,162,58,.10)}
.mn-warn .mn-verdict-ico{background:rgba(224,162,58,.30);color:#8D5D17}
.mn-ok{border:1px solid #2F9D8F;background:rgba(45,212,168,.10)}
.mn-ok .mn-verdict-ico{background:rgba(45,212,168,.30);color:#8FF0D5}
.mn-note{display:flex;gap:14px;align-items:flex-start;font-size:13.5px;line-height:1.6;color:#5E7188}
.mn-note svg{flex:0 0 22px;color:#5E7188}
.mn-title2{margin:6px 0 4px;font-size:36px;font-weight:800;color:#24364D;letter-spacing:-.01em}
.mn-sub2{margin:0 0 18px;font-size:16px;color:#5E7188}
.mn-metrics{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:18px;margin-bottom:16px}
@media (max-width:1000px){.mn-metrics{grid-template-columns:repeat(2,minmax(0,1fr))}}
.mn-metric{display:flex;align-items:center;gap:18px;border-radius:18px;padding:20px 22px;border:1px solid rgba(130,150,255,.18);background:linear-gradient(160deg,#FFFFFF,#F7F9FC)}
.mn-metric-ico{flex:0 0 62px;height:62px;border-radius:50%;display:flex;align-items:center;justify-content:center;color:#fff}
.mn-metric-lab{font-size:16px;color:#43566F}
.mn-metric-val{font-size:40px;line-height:1.1;font-weight:800;color:#fff}
.mn-info{display:flex;align-items:center;gap:14px;border-radius:14px;padding:14px 18px;margin-bottom:18px;font-size:15px;color:#43566F;border:1px solid rgba(76,125,255,.35);background:rgba(40,70,160,.18)}
.mn-info svg{flex:0 0 24px;color:#6FA0FF}
.mn-table{width:100%;border-collapse:separate;border-spacing:0;border:1px solid rgba(130,150,255,.2);border-radius:14px;overflow:hidden;font-size:15px;color:#263449}
.mn-table th{padding:14px 12px;text-align:center;font-weight:500;color:#52657D;background:rgba(20,30,60,.5);border-bottom:1px solid rgba(130,150,255,.2)}
.mn-table td{padding:16px 12px;text-align:center;border-bottom:1px solid rgba(130,150,255,.12)}
.mn-table td:first-child,.mn-table th:first-child{text-align:left;padding-left:18px}
.mn-table tr:last-child td{border-bottom:none}
.mn-table tr.mn-sel td{background:rgba(26,163,154,.18)}
.mn-table tr.mn-sel td:first-child{box-shadow:inset 4px 0 0 #2F9D8F}
.mn-ok-note{display:flex;align-items:center;gap:14px;margin-top:16px;border-radius:14px;padding:14px 18px;font-size:15px;color:#263449;border:1px solid rgba(45,212,168,.4);background:rgba(45,212,168,.10)}
.mn-ok-note svg{flex:0 0 30px;padding:5px;border-radius:50%;background:#2F9D8F;color:#EDF2F8;box-sizing:border-box}
.mn-rows{margin-top:14px;border:1px solid rgba(130,150,255,.2);border-radius:14px;overflow:hidden}
.mn-row{display:flex;align-items:center;gap:14px;padding:14px 18px;font-size:15px;color:#263449;border-bottom:1px solid rgba(130,150,255,.12)}
.mn-row:last-child{border-bottom:none}
.mn-row svg{color:#5E7188}
.mn-row b{color:#fff}
.mn-steps{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:16px;margin-bottom:18px}
@media (max-width:1000px){.mn-steps{grid-template-columns:repeat(2,minmax(0,1fr))}}
.mn-step{border-radius:18px;padding:20px;border:1px solid rgba(130,150,255,.18);background:linear-gradient(160deg,#FFFFFF,#F7F9FC)}
.mn-step .n{display:inline-flex;width:34px;height:34px;border-radius:50%;align-items:center;justify-content:center;font-weight:800;color:#EDF2F8;background:linear-gradient(135deg,#7FE3F7,#7B70C2)}
.mn-step h4{margin:12px 0 6px;font-size:18px;color:#fff}
.mn-step p{margin:0;font-size:14px;line-height:1.5;color:#5E7188}
.mn-chips{display:flex;flex-wrap:wrap;gap:8px;margin:12px 0 4px}
.mn-chip{padding:8px 16px;border-radius:999px;font-size:14px;color:#263449;border:1px solid rgba(139,124,246,.4);background:rgba(139,124,246,.14)}
.mn-card{border-radius:18px;padding:22px 24px;border:1px solid rgba(130,150,255,.18);background:linear-gradient(160deg,#FFFFFF,#F7F9FC);height:100%;box-sizing:border-box}
.mn-card h3{margin:0 0 8px;font-size:21px;color:#fff}
.mn-card p{margin:0 0 10px;font-size:15px;line-height:1.6;color:#52657D}
.mn-li{display:flex;gap:12px;align-items:flex-start;margin:0 0 12px;font-size:15px;line-height:1.55;color:#43566F}
.mn-li i{flex:0 0 9px;height:9px;margin-top:8px;border-radius:50%;background:#7B70C2}
button[data-baseweb="tab"]{font-size:16px;padding:14px 22px}
div[data-testid="stButton"] button{border-radius:14px;min-height:54px;font-weight:600}

/* Ajustes de legibilidade: manter a identidade visual e garantir contraste nos controles. */
.stApp { background: #EDF2F8 !important; color: #263449 !important; }
.block-container { max-width: 1380px; padding-top: 1.2rem; padding-bottom: 2.5rem; }
.mn-hero { background: linear-gradient(105deg, #E2EAF5 0%, #D9E5F3 60%, #E6E2F5 100%) !important; border: 1px solid #C8D6E7 !important; box-shadow: 0 8px 24px rgba(39,58,83,.08) !important; }
.mn-title, .mn-title2, .mn-head h3, .mn-result h2, .mn-card h3, .mn-step h4 { color: #263449 !important; }
.mn-title span, .mn-grad { background: none !important; color: #6559A8 !important; -webkit-text-fill-color: #6559A8 !important; }
.mn-sub, .mn-sub2, .mn-head p, .mn-step p, .mn-card p, .mn-li, .mn-note, .mn-row, .mn-metric-lab { color: #52657D !important; }
.mn-logo { background: linear-gradient(135deg, #627FB0, #8B7CC2) !important; }
.mn-head-ico { background: #DCE6F3 !important; color: #405A7D !important; }
.mn-result, .mn-score, .mn-metric, .mn-step, .mn-card { background: #FFFFFF !important; border: 1px solid #D3DEEA !important; box-shadow: 0 4px 14px rgba(39,58,83,.05); }
.mn-result h2, .mn-score-val, .mn-verdict b, .mn-metric-val { color: #263449 !important; }
.mn-score-lab, .mn-verdict, .mn-info, .mn-ok-note { color: #43566F !important; }
.mn-score { background: #F6F9FC !important; }
.mn-bar { background: #D9E3EF !important; }
.mn-verdict { background-color: #F7F9FC !important; }
.mn-ok { border-color: #8CCFC2 !important; background: #EAF7F3 !important; }
.mn-warn { border-color: #E2C18B !important; background: #FFF6E8 !important; }
.mn-note svg { color: #5E7188 !important; }
.mn-info { background: #E7F0FA !important; border-color: #C8D9EB !important; }
.mn-table { color: #263449 !important; border-color: #D3DEEA !important; }
.mn-table th { color: #43566F !important; background: #EDF2F8 !important; border-color: #D3DEEA !important; }
.mn-table td, .mn-row { border-color: #E2E8F0 !important; }
.mn-table tr.mn-sel td { background: #E5F4F0 !important; }
.mn-ok-note { background: #EAF7F3 !important; border-color: #B9DED5 !important; }
.mn-row b { color: #263449 !important; }
.mn-chip { color: #51478B !important; border-color: #D6D0EE !important; background: #F0EDFA !important; }
.mn-step .n { color: #FFFFFF !important; background: #6C83B5 !important; }

/* Menu de abas: seletores compatíveis com diferentes versões do Streamlit */
div[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: #F4F1FA !important;
    border-radius: 14px !important;
    padding: 6px 10px !important;
    gap: 6px !important;
}

/* Todas as abas começam com texto cinza-azulado legível */
div[data-testid="stTabs"] button[role="tab"],
div[data-testid="stTabs"] button[role="tab"] p,
div[data-testid="stTabs"] button[role="tab"] span,
div[data-testid="stTabs"] button[role="tab"] svg {
    color: #596579 !important;
    -webkit-text-fill-color: #596579 !important;
    fill: currentColor !important;
    opacity: 1 !important;
    font-weight: 600 !important;
    border-color: transparent !important;
}

/* Aba ativa: roxo e indicador inferior roxo */
div[data-testid="stTabs"] button[role="tab"][aria-selected="true"],
div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] p,
div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] span,
div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] svg {
    color: #6858A8 !important;
    -webkit-text-fill-color: #6858A8 !important;
    fill: currentColor !important;
    border-color: transparent !important;
}

/* Remove indicadores laterais e mantém somente a linha inferior */
div[data-testid="stTabs"] [data-baseweb="tab-border"],
div[data-testid="stTabs"] [data-baseweb="tab-highlight"] {
    background-color: #6858A8 !important;
    border-color: transparent !important;
}

div[data-testid="stTabs"] button[role="tab"]:hover,
div[data-testid="stTabs"] button[role="tab"]:hover p,
div[data-testid="stTabs"] button[role="tab"]:hover span,
div[data-testid="stTabs"] button[role="tab"]:hover svg {
    color: #6858A8 !important;
    -webkit-text-fill-color: #6858A8 !important;
    fill: currentColor !important;
    background-color: #EAE5F7 !important;
}
/* Botões de exemplo. */
div[data-testid="stButton"] button { color: #FFFFFF !important; -webkit-text-fill-color: #FFFFFF !important; background: #647FAE !important; border: 1px solid #647FAE !important; }
div[data-testid="stButton"] button:hover { background: #526E9D !important; border-color: #526E9D !important; }

/* Rótulos e campos de formulário: fundo claro e texto escuro, inclusive nos elementos internos do BaseWeb. */
div[data-testid="stSelectbox"] label,
div[data-testid="stNumberInput"] label,
div[data-testid="stCheckbox"] label,
div[data-testid="stSelectbox"] label *,
div[data-testid="stNumberInput"] label *,
div[data-testid="stCheckbox"] label * { color: #263449 !important; -webkit-text-fill-color: #263449 !important; }

div[data-testid="stSelectbox"] [data-baseweb="select"] > div,
div[data-testid="stMultiSelect"] [data-baseweb="select"] > div,
div[data-testid="stSelectbox"] [role="combobox"],
div[data-testid="stMultiSelect"] [role="combobox"] {
  background: #FFFFFF !important; border: 1px solid #C8D6E7 !important;
  box-shadow: none !important; color: #263449 !important;
}
div[data-testid="stSelectbox"] [data-baseweb="select"] *,
div[data-testid="stMultiSelect"] [data-baseweb="select"] * {
  color: #263449 !important; -webkit-text-fill-color: #263449 !important;
}
div[data-testid="stSelectbox"] [data-baseweb="select"] svg,
div[data-testid="stMultiSelect"] [data-baseweb="select"] svg { fill: #52657D !important; color: #52657D !important; }

div[data-testid="stNumberInput"] input { background: #FFFFFF !important; color: #263449 !important; -webkit-text-fill-color: #263449 !important; border-color: #C8D6E7 !important; }
div[data-testid="stNumberInput"] button { background: #E7EDF5 !important; color: #263449 !important; border-color: #C8D6E7 !important; }

/* Lista que aparece ao abrir um menu de seleção. */
div[data-baseweb="popover"], div[data-baseweb="popover"] *,
ul[role="listbox"], li[role="option"], li[role="option"] * {
  background-color: #FFFFFF !important; color: #263449 !important; -webkit-text-fill-color: #263449 !important;
}
li[role="option"]:hover { background-color: #E7F0FA !important; }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] * { color: #5E7188 !important; }
.stSelectbox, .stNumberInput, .stCheckbox { color: #263449 !important; }







/* Corrigir a cor dos nomes das abas */
div[data-testid="stTabs"] button[role="tab"] {
    color: #596579 !important;
    -webkit-text-fill-color: #596579 !important;
    opacity: 1 !important;
}

div[data-testid="stTabs"] button[role="tab"] * {
    color: #596579 !important;
    -webkit-text-fill-color: #596579 !important;
    opacity: 1 !important;
}

div[data-testid="stTabs"] button[role="tab"][aria-selected="true"],
div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] * {
    color: #6858A8 !important;
    -webkit-text-fill-color: #6858A8 !important;
}

/* Corrigir somente o botão da seta dos campos de seleção */
div[data-testid="stSelectbox"] button[aria-label="Open"],
div[data-testid="stMultiSelect"] button[aria-label="Open"] {
    background-color: #17213B !important;
    color: #FFFFFF !important;
}

div[data-testid="stSelectbox"] button[aria-label="Open"] svg,
div[data-testid="stMultiSelect"] button[aria-label="Open"] svg {
    fill: #FFFFFF !important;
    color: #FFFFFF !important;
}

</style>
"""

html(CSS)

# ----------------------------------------------------------------------------
# Cabeçalho
# ----------------------------------------------------------------------------
foto = foto_cabecalho()
estilo_foto = f' style="background-image:{foto}"' if foto else ""
html(f"""
<div class="mn-hero">
<div class="mn-hero-left">
<div class="mn-logo">{icone('paw', 52)}</div>
<div>
<h1 class="mn-title">Miau-net: <span>previsão de adoção de gatos</span></h1>
<p class="mn-sub">Sistema de Machine Learning para prever as chances de adoção de gatos com base em características do animal no momento da entrada no abrigo.</p>
</div>
</div>
<div class="mn-hero-right">
<div class="mn-photo"{estilo_foto}></div>
<div class="mn-tag">Mais adoções, mais histórias felizes!</div>
</div>
</div>
""")

# ----------------------------------------------------------------------------
# Modelo e dados
# ----------------------------------------------------------------------------
try:
    modelo, preprocessador = carregar_modelos()
except Exception as erro:
    st.error(f"Não foi possível carregar o modelo salvo: {erro}")
    st.info("Confira se os dois arquivos .joblib estão dentro da pasta modelos.")
    st.stop()

caminho_csv = next((c for c in CAMINHOS_CSV if c.exists()), None)
if caminho_csv is None:
    arquivo = st.file_uploader("Envie animal-shelter-intakes-and-outcomes.csv para preencher as opções.", type="csv")
    if arquivo is None:
        st.warning("O modelo foi carregado. Para usar o formulário, localize ou envie o CSV.")
        st.stop()
    dados = carregar_gatos(arquivo.getvalue())
else:
    dados = carregar_gatos(str(caminho_csv))
OPCOES = preparar_opcoes(dados)

if "sexo" not in st.session_state:
    aplicar_preset("filhote", OPCOES)

aba_prever, aba_desempenho, aba_fatores, aba_sobre = st.tabs([
    ":material/home: Previsão",
    ":material/bar_chart: Desempenho",
    ":material/lightbulb: Fatores que mais pesam",
    ":material/info: Sobre o modelo",
])

# ----------------------------------------------------------------------------
# Aba 1: previsão
# ----------------------------------------------------------------------------
with aba_prever:
    esquerda, direita = st.columns([3, 2], gap="large")

    with esquerda:
        with st.container(border=True):
            html(f"""
            <div class="mn-head"><div class="mn-head-ico">{icone('spark', 20)}</div>
            <div><h3>Informe as características do <span class="mn-grad">gato</span> no momento da entrada no abrigo</h3>
            <p>Escolha um exemplo ou preencha os dados abaixo para ver a previsão de adoção.</p></div></div>
            """)
            botoes = st.columns(len(PRESETS))
            for coluna_botao, chave in zip(botoes, PRESETS):
                with coluna_botao:
                    st.button(PRESETS[chave]["titulo"], on_click=aplicar_preset, args=(chave, OPCOES),
                              width="stretch", key=f"btn_{chave}")
                    st.caption(PRESETS[chave]["legenda"])

            c1, c2 = st.columns(2)
            sexo = c1.selectbox(":material/person: Sexo", OPCOES["Sex"], key="sexo", format_func=rotulo("Sex"))
            condicao = c2.selectbox(":material/favorite: Condição na entrada", OPCOES["Intake Condition"], key="cond",
                                    format_func=rotulo("Intake Condition"))
            tipo = c1.selectbox(":material/login: Tipo de entrada", OPCOES["Intake Type"], key="tipo",
                                format_func=rotulo("Intake Type"))
            subtipo = c2.selectbox(":material/sell: Subtipo de entrada", OPCOES["Intake Subtype"], key="sub",
                       format_func=rotulo("Intake Subtype"))
            cor = c1.selectbox(":material/palette: Cor principal", OPCOES["Primary Color"], key="cor",
                   format_func=rotulo("Primary Color"))
            
            jurisdicao = c2.selectbox(":material/location_on: Jurisdição", OPCOES["Jurisdiction"], key="jur")

            c3, c4 = st.columns(2)
            idade_meses = c3.number_input(":material/calendar_month: Idade na entrada (meses)", min_value=0.0,
                                          max_value=288.0, step=0.1, format="%.2f", key="idade",
                                          disabled=st.session_state.get("idade_desc", False))
            c4.write("")
            c4.write("")
            idade_desconhecida = c4.checkbox("Idade desconhecida", key="idade_desc")

    valores = {
        "idade_na_entrada": np.nan if idade_desconhecida else idade_meses / 12,
        "Sex": sexo, "Intake Condition": condicao, "Intake Type": tipo,
        "Intake Subtype": subtipo, "Primary Color": cor, "Jurisdiction": jurisdicao,
    }
    pontuacao, classe = prever(modelo, preprocessador, valores)
    largura = max(pontuacao * 100, 2.5)
    adocao = classe == 1
    estilo_veredito = "mn-ok" if adocao else "mn-warn"
    desfecho = "ADOÇÃO" if adocao else "OUTRO DESFECHO"
    perto = " (perto do limite)" if abs(pontuacao - 0.5) < 0.10 else ""

    with direita:
        html(f"""
        <div class="mn-result">
        <h2>{icone('spark', 30)} Resultado</h2>
        <div class="mn-score">
        <div class="mn-score-top"><div class="mn-score-ico">{icone('paw', 34)}</div>
        <div><div class="mn-score-lab">Pontuação de adoção segundo o modelo</div><div class="mn-score-val">{pontuacao:.0%}</div></div></div>
        <div class="mn-bar"><div style="width:{largura:.1f}%"></div></div>
        </div>
        <div class="mn-verdict {estilo_veredito}"><div class="mn-verdict-ico">{icone('bulb', 24)}</div>
        <div>O modelo classifica este gato como<b>{desfecho}{perto}</b></div></div>
        <div class="mn-note">{icone('info', 22)}
        <div>A pontuação é produzida por um modelo treinado com dados históricos. Como as classes foram balanceadas, ela não deve ser interpretada como uma probabilidade exata. O resultado não garante o desfecho de um animal.</div></div>
        </div>
        """)

# ----------------------------------------------------------------------------
# Aba 2: desempenho
# ----------------------------------------------------------------------------
with aba_desempenho:
    t = METRICAS_TESTE
    html(f"""
    <div class="mn-title2">Avaliação final <span class="mn-grad">(conjunto de teste)</span></div>
    <p class="mn-sub2">Resultados do modelo de Machine Learning para prever a adoção de gatos.</p>
    <div class="mn-metrics">
    <div class="mn-metric"><div class="mn-metric-ico" style="background:linear-gradient(135deg,#5B78B5,#7898CB)">{icone('target', 30)}</div>
    <div><div class="mn-metric-lab">Acurácia</div><div class="mn-metric-val">{pct(t['Acurácia'])}</div></div></div>
    <div class="mn-metric"><div class="mn-metric-ico" style="background:linear-gradient(135deg,#278F83,#2F9D8F)">{icone('target', 30)}</div>
    <div><div class="mn-metric-lab">Precisão (adoção)</div><div class="mn-metric-val">{pct(t['Precisão'])}</div></div></div>
    <div class="mn-metric"><div class="mn-metric-ico" style="background:linear-gradient(135deg,#8174C4,#7B70C2)">{icone('refresh', 30)}</div>
    <div><div class="mn-metric-lab">Recall (adoção)</div><div class="mn-metric-val">{pct(t['Recall'])}</div></div></div>
    <div class="mn-metric"><div class="mn-metric-ico" style="background:linear-gradient(135deg,#547FAF,#7DA8D5)">{icone('star', 30)}</div>
    <div><div class="mn-metric-lab">F1-score (adoção)</div><div class="mn-metric-val">{pct(t['F1'])}</div></div></div>
    </div>
    <div class="mn-info">{icone('info', 24)}<div>Um classificador que sempre previsse "Outro desfecho" acertaria aproximadamente 75,1% dos casos, mas não identificaria adoções. Por isso, recall e F1 da classe adoção também são importantes.</div></div>
    """)

    grafico, tabela_col = st.columns(2, gap="large")
    with grafico:
        with st.container(border=True):
            html(f'<div class="mn-head"><div class="mn-head-ico">{icone("grid", 20)}</div><div><h3>Matriz de confusão</h3></div></div>')
            cmap = LinearSegmentedColormap.from_list("mn", ["#D9EAF8", "#5B8FC4", "#315F9B"])
            fig, ax = plt.subplots(figsize=(5.6, 4.2))
            fig.patch.set_facecolor(BG)
            ax.set_facecolor(BG)
            im = ax.imshow(MATRIZ_TESTE, cmap=cmap, vmin=MATRIZ_TESTE.min(), vmax=MATRIZ_TESTE.max(), aspect="auto")
            classes = ["Outro desfecho", "Adoção"]
            ax.set_xticks([0, 1], classes, color=TEXTO)
            ax.set_yticks([0, 1], classes, color=TEXTO)
            ax.set_xlabel("Previsto pelo modelo", color="#526F9A", fontsize=12)
            ax.set_ylabel("Desfecho real", color="#526F9A", fontsize=12)
            ax.tick_params(length=0)
            for lado in ax.spines.values():
                lado.set_visible(False)
            meio = MATRIZ_TESTE.max() * 0.58
            for i in range(2):
                for j in range(2):
                    valor = int(MATRIZ_TESTE[i, j])
                    # Texto escuro nas células claras e branco apenas na célula mais escura.
                    cor_numero = "#24364D" if valor < meio else "#FFFFFF"
                    ax.text(j, i, milhar(valor), ha="center", va="center", fontsize=16,
                            fontweight="bold", color=cor_numero)
            barra = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.04)
            barra.outline.set_visible(False)
            barra.ax.tick_params(colors=SUAVE, length=0)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    with tabela_col:
        with st.container(border=True):
            html(f'<div class="mn-head"><div class="mn-head-ico">{icone("bars", 20)}</div><div><h3>Comparação na validação</h3></div></div>')
            linhas = ""
            for nome, linha in METRICAS_VALIDACAO.iterrows():
                sel = ' class="mn-sel"' if "balanceada" in nome else ""
                linhas += (f"<tr{sel}><td>{nome}</td><td>{pct(linha['Acurácia'])}</td><td>{pct(linha['Precisão'])}</td>"
                           f"<td>{pct(linha['Recall'])}</td><td>{pct(linha['F1'])}</td></tr>")
            html(f"""
            <table class="mn-table"><thead><tr><th></th><th>Acurácia</th><th>Precisão</th><th>Recall</th><th>F1</th></tr></thead>
            <tbody>{linhas}</tbody></table>
            <div class="mn-ok-note">{icone('check', 30)}<div>A versão balanceada teve o melhor F1 na validação e foi escolhida para a avaliação final.</div></div>
            <div class="mn-rows">
            <div class="mn-row">{icone('db', 22)}<div><b>Divisão dos dados:</b> &nbsp; treino 16.230 | validação 5.410 | teste 5.410</div></div>
            <div class="mn-row">{icone('code', 22)}<div><b>Variáveis após o One-Hot Encoding:</b> &nbsp; 142</div></div>
            </div>
            """)

# ----------------------------------------------------------------------------
# Aba 3: fatores
# ----------------------------------------------------------------------------
with aba_fatores:
    html("""
    <div class="mn-title2">Fatores que <span class="mn-grad">mais pesam</span></div>
    <p class="mn-sub2">Associações aprendidas pelo modelo; não demonstram causa e efeito.</p>
    """)
    nomes_features = preprocessador.get_feature_names_out()
    coeficientes = modelo.coef_[0]
    tabela_coef = pd.DataFrame({
        "feature_original": nomes_features,
        "fator": [nome_amigavel(n) for n in nomes_features],
        "coef": coeficientes,
    })
    contagens = {}
    for coluna in CAT:
        serie = dados[coluna].fillna("Não informado").astype(str).value_counts()
        for valor, quantidade in serie.items():
            contagens[f"{coluna}_{valor}"] = int(quantidade)

    def contar_feature(nome):
        nome = nome.replace("categorico__", "")
        if "idade_na_entrada" in nome:
            return len(dados)
        return contagens.get(nome, 0)

    tabela_coef["n"] = [contar_feature(n) for n in tabela_coef["feature_original"]]
    tabela_categorias = tabela_coef[(tabela_coef["fator"] != "Idade") & (tabela_coef["n"] >= 100)].copy()
    if tabela_categorias.empty:
        st.info("Não há categorias suficientes para exibir o gráfico.")
    else:
        destaque = pd.concat([
            tabela_categorias.nsmallest(8, "coef"),
            tabela_categorias.nlargest(8, "coef"),
        ]).drop_duplicates(subset=["feature_original"]).sort_values("coef")
        cores = ["#D58A54" if valor < 0 else "#2F9D8F" for valor in destaque["coef"]]
        with st.container(border=True):
            fig, ax = plt.subplots(figsize=(9, 5.6))
            fig.patch.set_facecolor(BG)
            ax.set_facecolor(BG)
            ax.barh(destaque["fator"], destaque["coef"], color=cores)
            ax.axvline(0, color=SUAVE, linewidth=0.8)
            ax.set_xlabel("Coeficiente da Regressão Logística", color=SUAVE)
            ax.tick_params(colors=TEXTO, length=0)
            for lado in ax.spines.values():
                lado.set_visible(False)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
    idade_rows = tabela_coef[tabela_coef["fator"] == "Idade"]
    if not idade_rows.empty:
        coef_idade = float(idade_rows.iloc[0]["coef"])
        sentido = "diminui" if coef_idade < 0 else "aumenta"
        html(f"""
        <div class="mn-info">{icone('info', 24)}<div>Verde aumenta e laranja diminui a pontuação de adoção. O coeficiente da idade é {coef_idade:.2f}; seu sinal indica associação com uma pontuação de adoção que {sentido}. Isso não significa que a idade cause o desfecho.</div></div>
        """)

# ----------------------------------------------------------------------------
# Aba 4: sobre
# ----------------------------------------------------------------------------
with aba_sobre:
    taxa_adocao = float(dados["adotado"].mean())
    html(f"""
    <div class="mn-title2">Sobre o <span class="mn-grad">modelo</span></div>
    <p class="mn-sub2">Do dado bruto à previsão de adoção de gatos.</p>
    <div class="mn-steps">
    <div class="mn-step"><span class="n">1</span><h4>Dados</h4><p>{milhar(len(dados))} gatos com desfecho registrado, de uma base com 57.562 registros de animais.</p></div>
    <div class="mn-step"><span class="n">2</span><h4>Preparação</h4><p>Valores ausentes preenchidos e One-Hot Encoding, gerando 142 variáveis.</p></div>
    <div class="mn-step"><span class="n">3</span><h4>Divisão</h4><p>60% treino, 20% validação e 20% teste, mantendo a proporção de adoções.</p></div>
    <div class="mn-step"><span class="n">4</span><h4>Modelo</h4><p>Regressão Logística com classes balanceadas, treinada no notebook.</p></div>
    <div class="mn-step"><span class="n">5</span><h4>Avaliação</h4><p>F1-score de {pct(METRICAS_TESTE['F1'])} para a classe adoção no conjunto de teste.</p></div>
    </div>
    """)
    a, b = st.columns(2, gap="large")
    with a:
        html(f"""
        <div class="mn-card"><h3>O que o modelo prevê</h3>
        <p>Adoção = 1. Qualquer outro desfecho conhecido = 0. Taxa histórica de adoção nesta base: {taxa_adocao:.1%}.</p>
        <div class="mn-chips"><span class="mn-chip">Idade na entrada</span><span class="mn-chip">Sexo</span><span class="mn-chip">Condição de entrada</span>
        <span class="mn-chip">Tipo de entrada</span><span class="mn-chip">Subtipo de entrada</span><span class="mn-chip">Cor principal</span><span class="mn-chip">Jurisdição</span></div>
        <h3 style="margin-top:22px">Por que a classificação</h3>
        <p>A Regressão Linear do tempo de permanência apresentou erros elevados e baixa capacidade de explicar a variação. Por isso o problema foi reformulado como classificação.</p></div>
        """)
    with b:
        html("""
        <div class="mn-card"><h3>Limitações</h3>
        <div class="mn-li"><i></i><div>A base é de um único abrigo e os padrões podem não valer para outros locais.</div></div>
        <div class="mn-li"><i></i><div>A precisão para adoção é limitada; parte dos casos classificados como adoção teve outro desfecho.</div></div>
        <div class="mn-li"><i></i><div>O modelo identifica associações, não causa e efeito, e não substitui avaliação humana.</div></div>
        <div class="mn-li"><i></i><div>A pontuação não garante o desfecho de um gato individual.</div></div></div>
        """)
