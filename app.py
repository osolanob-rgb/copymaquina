"""
Copymáquina · Generación agéntica de copies de performance a partir de un brief
Módulo Python para IA · Máster en Inteligencia Artificial · Oscar Solano Brenes

Pipeline de agentes (cada uno es una llamada al LLM con un rol y un contrato JSON):
  1. Analista de brief   -> brief estructurado (marca, objetivo, audiencia, insight, claims, tono...)
  2. Estratega           -> N ángulos creativos (dolor, beneficio, prueba social, urgencia...)
  3. Copywriter          -> por cada ángulo x canal, un lote de copies (en PARALELO) que respeta el formato
  4. QA / Compliance     -> un lote por canal: puntúa, marca claims prohibidos; lo que falla se regenera
  5. Exportación         -> tarjetas, tabla filtrable, CSV / Excel

Sin clave de API la app corre en MODO DEMO con un LLM simulado (versión 0 del reto).
"""

from __future__ import annotations

import html
import io
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import streamlit as st

APP_NOMBRE = "Copymáquina"
BASE_DIR = Path(__file__).parent
UMBRAL_QA = 7          # puntuación mínima (0-10) para aprobar un copy
RONDAS_MAX = 3         # intentos por (canal, ángulo) antes de rendirse
MODELOS_RESPALDO = ["gemini-3.8-flash", "gemini-2.0-flash"]

# ---------------------------------------------------------------------------
# Formatos y límites de caracteres por canal
# ---------------------------------------------------------------------------
CANALES: dict[str, dict] = {
    "Meta Ads (Facebook/Instagram)": {
        "corto": "Meta", "color": "#1877F2",
        "campos": {"texto_principal": 125, "titulo": 40, "descripcion": 30},
        "nota": "Texto principal ≤125 (visible sin 'ver más'), título ≤40, descripción ≤30.",
    },
    "Google Ads (RSA)": {
        "corto": "Google", "color": "#EA4335",
        "campos": {"titulo": 30, "descripcion": 90},
        "nota": "Anuncios de búsqueda responsivos: títulos ≤30, descripciones ≤90. Sin emojis.",
    },
    "TikTok Ads": {
        "corto": "TikTok", "color": "#111111",
        "campos": {"texto": 100, "cta": 20},
        "nota": "Texto del anuncio ≤100; tono nativo, hablado, sin lenguaje corporativo.",
    },
    "LinkedIn Ads": {
        "corto": "LinkedIn", "color": "#0A66C2",
        "campos": {"texto_intro": 150, "titulo": 70},
        "nota": "Introducción ≤150 (visible), título ≤70. Tono profesional, orientado a resultados.",
    },
    "Email (asunto + preheader)": {
        "corto": "Email", "color": "#7C3AED",
        "campos": {"asunto": 50, "preheader": 90},
        "nota": "Asunto ≤50, preheader ≤90. Nada de MAYÚSCULAS completas ni signos repetidos.",
    },
    "Display / Banner": {
        "corto": "Display", "color": "#F59E0B",
        "campos": {"titular": 35, "subtitular": 60, "cta": 15},
        "nota": "Titular ≤35, subtitular ≤60, botón ≤15.",
    },
}

ANGULOS_BASE = [
    "Dolor / problema", "Beneficio directo", "Prueba social", "Urgencia / escasez",
    "Curiosidad", "Comparación / antes-después", "Objeción resuelta", "Identidad / aspiración",
]


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def _extraer_json(texto: str):
    limpio = re.sub(r"```(?:json)?", "", texto).strip().rstrip("`").strip()
    try:
        return json.loads(limpio)
    except json.JSONDecodeError:
        inicio = min([i for i in (limpio.find("{"), limpio.find("[")) if i >= 0] or [0])
        fin = max(limpio.rfind("}"), limpio.rfind("]")) + 1
        return json.loads(limpio[inicio:fin])


def _primera_coincidencia(patron: str, texto: str) -> str | None:
    m = re.search(patron, texto)
    return m.group(1).strip() if m else None


def _lista(texto: str | None) -> list[str]:
    return [t.strip() for t in re.split(r"[,;]", texto) if t.strip()] if texto else []


# ---------------------------------------------------------------------------
# Cliente LLM: Gemini real o simulado (modo demo)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=3600)
def listar_modelos(api_key: str) -> list[str]:
    """Pregunta a la API qué modelos puede usar esta clave (los nombres cambian con el tiempo)."""
    try:
        from google import genai
        cliente = genai.Client(api_key=api_key)
        nombres = []
        for m in cliente.models.list():
            acciones = getattr(m, "supported_actions", None) or []
            if acciones and "generateContent" not in acciones:
                continue
            nombre = m.name.replace("models/", "")
            if "gemini" in nombre and not any(x in nombre for x in ("embedding", "tts", "image", "live", "audio")):
                nombres.append(nombre)
        flash = [n for n in nombres if "flash" in n]
        return list(dict.fromkeys(flash + nombres)) or MODELOS_RESPALDO
    except Exception:
        return MODELOS_RESPALDO


class GeminiLLM:
    """Llamadas a Gemini con salida JSON forzada y razonamiento interno desactivado (es más rápido
    y esta tarea no lo necesita). Si el modelo no acepta ese ajuste, se reintenta sin él.
    La clave nunca se escribe en el código: viene de st.secrets o de la barra lateral."""

    def __init__(self, api_key: str, modelo: str = MODELOS_RESPALDO[0]):
        from google import genai
        self.cliente = genai.Client(api_key=api_key)
        self.modelo = modelo
        self.sin_thinking = True
        self.llamadas = 0
        self._lock = threading.Lock()

    def _config(self, sistema: str, temperatura: float):
        from google.genai import types
        kw = dict(system_instruction=sistema, temperature=temperatura,
                  response_mime_type="application/json", max_output_tokens=8192)
        if self.sin_thinking:
            kw["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
        return types.GenerateContentConfig(**kw)

    def json(self, sistema: str, usuario: str, temperatura: float = 0.9):
        ultimo_error = None
        for intento in range(3):
            try:
                with self._lock:
                    self.llamadas += 1
                resp = self.cliente.models.generate_content(
                    model=self.modelo, contents=usuario, config=self._config(sistema, temperatura))
                return _extraer_json(resp.text)
            except Exception as e:
                ultimo_error = e
                msg = str(e).lower()
                if self.sin_thinking and ("thinking" in msg or "invalid" in msg or "400" in msg):
                    self.sin_thinking = False   # este modelo no acepta el ajuste: seguimos sin él
                    continue
                _ = time.sleep(2 * (intento + 1))    # cuota o red: espera y reintenta
        raise RuntimeError(f"Gemini no respondió tras 3 intentos: {ultimo_error}")


class DemoLLM:
    """Simula los 4 agentes sin red ni coste. Produce copies plausibles a partir del brief."""

    def __init__(self):
        self.llamadas = 0
        self._lock = threading.Lock()

    def json(self, sistema: str, usuario: str, temperatura: float = 0.9):
        with self._lock:
            self.llamadas += 1
        _ = time.sleep(0.15)  # simula latencia de red
        if "ANALISTA" in sistema:
            t = usuario
            return {
                "marca": (_primera_coincidencia(r"(?i)marca\s*[:\-]\s*(.+)", t) or "La marca").strip(),
                "producto": (_primera_coincidencia(r"(?i)producto\s*[:\-]\s*(.+)", t) or "el producto").strip(),
                "objetivo": _primera_coincidencia(r"(?i)objetivo\s*[:\-]\s*(.+)", t) or "Conversión",
                "audiencia": _primera_coincidencia(r"(?i)audiencia\s*[:\-]\s*(.+)", t) or "Adultos 25-45",
                "insight": _primera_coincidencia(r"(?i)insight\s*[:\-]\s*(.+)", t) or "Quieren resolverlo sin complicarse",
                "propuesta_de_valor": _primera_coincidencia(r"(?i)propuesta de valor\s*[:\-]\s*(.+)", t) or "Más resultado, menos esfuerzo",
                "beneficios": ["Ahorro de tiempo", "Resultado comprobable", "Fácil de empezar"],
                "tono": _primera_coincidencia(r"(?i)tono\s*[:\-]\s*(.+)", t) or "Directo y cercano",
                "cta_preferidos": _lista(_primera_coincidencia(r"(?i)cta\s*(?:preferidos?)?\s*[:\-]\s*(.+)", t)) or ["Comprar ahora", "Probar gratis", "Saber más"],
                "claims_prohibidos": _lista(_primera_coincidencia(r"(?i)prohibid[oa]s?\s*[:\-]\s*(.+)", t)) or ["el mejor del mercado", "garantizado"],
                "palabras_clave": ["oferta", "resultados", "hoy"],
                "oferta": _primera_coincidencia(r"(?i)oferta\s*[:\-]\s*(.+)", t) or "",
                "preguntas_abiertas": ["No se indica presupuesto por canal"],
            }
        if "ESTRATEGA" in sistema:
            n = int(_primera_coincidencia(r"Propón (\d+)", usuario) or 6)
            return {"angulos": [
                {"nombre": a, "hipotesis": f"La audiencia responde al enfoque de '{a.lower()}' porque conecta con el insight del brief.",
                 "hook": f"Enfoque {a.lower()}: por qué esto importa hoy"} for a in ANGULOS_BASE[:n]]}
        if "COPYWRITER" in sistema:
            d = _extraer_json(usuario.split("DATOS:")[-1])
            campos, n, angulo = d["campos"], d["cantidad"], d["angulo"]["nombre"]
            marca = d["brief"]["marca"]
            producto = " ".join(d["brief"]["producto"].split()[:4]).rstrip(",;(")
            ctas = d["brief"]["cta_preferidos"]; prohibidos = d["brief"].get("claims_prohibidos") or []
            semillas = {
                "texto_principal": "{marca}: {producto} pensado para quien no tiene tiempo que perder. {angulo}. {cta}.",
                "titulo": "{producto} que sí cumple", "descripcion": "{cta} con {marca}",
                "texto": "¿Todavía sin {producto}? {angulo}. {cta} 👇", "cta": "{cta}",
                "texto_intro": "Equipos que usan {producto} reportan mejores resultados en semanas. {angulo}. {cta}.",
                "asunto": "{producto}: {angulo}", "preheader": "{marca} te lo pone fácil. {cta}.",
                "titular": "{producto}, sin vueltas", "subtitular": "{angulo}. {cta} con {marca}.",
            }
            copies = []
            for i in range(n):
                cta = ctas[i % len(ctas)]
                c = {"variante": i + 1}
                for campo, limite in campos.items():
                    base = semillas.get(campo, "{producto}: {cta}").format(marca=marca, producto=producto, angulo=angulo, cta=cta)
                    if i % 3 == 1: base = base.replace("hoy", "esta semana").replace("sin vueltas", "al grano")
                    if i % 3 == 2: base = base.replace("¿Todavía", "¿Aún")
                    if i == 3 and n >= 4 and prohibidos and campo == list(campos)[0]:
                        base = f"{prohibidos[0].capitalize()}: " + base   # para enseñar el bucle de QA
                    c[campo] = base[:limite]
                copies.append(c)
            return {"copies": copies}
        if "QA" in sistema:
            d = _extraer_json(usuario.split("DATOS:")[-1])
            prohibidos = [p.lower() for p in d["brief"].get("claims_prohibidos", [])]
            out = []
            for c in d["copies"]:
                texto = " ".join(str(v) for k, v in c.items() if k != "id").lower()
                hallados = [p for p in prohibidos if p in texto]
                out.append({"id": c["id"], "puntuacion": 8 if not hallados else 4, "claims_prohibidos_detectados": hallados,
                            "comentario": "Claro y dentro del brief" if not hallados else f"Contiene claim prohibido: {hallados}"})
            return {"evaluaciones": out}
        return {}


# ---------------------------------------------------------------------------
# Agentes
# ---------------------------------------------------------------------------
PROMPT_ANALISTA = """Eres el ANALISTA DE BRIEFS de una agencia de publicidad de performance.
Lees un brief (puede venir desordenado o incompleto) y lo conviertes en un JSON con EXACTAMENTE estas claves:
marca, producto, objetivo, audiencia, insight, propuesta_de_valor, beneficios (lista), tono,
cta_preferidos (lista), claims_prohibidos (lista), palabras_clave (lista), oferta, preguntas_abiertas (lista de lo que falta en el brief).
No inventes claims ni datos que no estén en el brief; si falta algo, déjalo vacío y anótalo en preguntas_abiertas.
Responde solo con el JSON."""

PROMPT_ESTRATEGA = """Eres el ESTRATEGA creativo de performance. A partir del brief estructurado propones ángulos de mensaje
distintos entre sí, cada uno con una hipótesis de por qué puede convertir con esa audiencia.
Responde solo con JSON: {"angulos": [{"nombre": str, "hipotesis": str, "hook": str}]}."""

PROMPT_COPYWRITER = """Eres el COPYWRITER senior de performance. Escribes lotes de copies para un canal concreto a partir de un
brief estructurado y de UN ángulo creativo. Reglas:
- Respeta estrictamente el límite de caracteres de cada campo (cuenta espacios). Mejor corto que cortado.
- Cada variante debe ser claramente distinta: cambia el hook, la estructura o el CTA; no cambies solo una palabra.
- Nunca uses los claims prohibidos. No inventes cifras ni premios. Usa el tono y el idioma del brief.
- Nada de relleno genérico ("soluciones innovadoras", "líder del mercado").
Responde solo con JSON: {"copies": [{"variante": int, <un campo por cada campo pedido>}]}."""

PROMPT_QA = """Eres el revisor de QA y compliance de la agencia. Evalúas cada copy contra el brief: fidelidad al brief,
claridad, fuerza del CTA, ausencia de claims prohibidos o promesas no sustentadas, y adecuación al canal.
Puntúa de 0 a 10. Por debajo de 7 el copy se descarta. Devuelve una evaluación por cada id recibido.
Responde solo con JSON: {"evaluaciones": [{"id": str, "puntuacion": int, "claims_prohibidos_detectados": [str], "comentario": str}]}."""


@dataclass
class Resultado:
    brief: dict = field(default_factory=dict)
    angulos: list = field(default_factory=list)
    filas: list = field(default_factory=list)
    descartados: int = 0
    llamadas_llm: int = 0
    segundos: float = 0.0


def agente_analista(llm, brief_texto: str) -> dict:
    return llm.json(PROMPT_ANALISTA, f"BRIEF:\n{brief_texto}", temperatura=0.2)


def agente_estratega(llm, brief: dict, n: int) -> list[dict]:
    salida = llm.json(PROMPT_ESTRATEGA, f"Propón {n} ángulos. Lista de referencia (puedes adaptarla): {ANGULOS_BASE}.\n"
                                        f"BRIEF:\n{json.dumps(brief, ensure_ascii=False)}")
    return salida.get("angulos", [])[:n]


def agente_copywriter(llm, brief: dict, angulo: dict, canal: str, cantidad: int) -> list[dict]:
    spec = CANALES[canal]
    datos = {"canal": canal, "nota_canal": spec["nota"], "campos": spec["campos"], "cantidad": cantidad, "angulo": angulo, "brief": brief}
    salida = llm.json(PROMPT_COPYWRITER, f"Escribe {cantidad} variantes para el canal {canal}.\nDATOS:{json.dumps(datos, ensure_ascii=False)}")
    return salida.get("copies", [])


def agente_qa(llm, brief: dict, canal: str, copies: list[dict]) -> dict[str, dict]:
    datos = {"canal": canal, "brief": brief, "copies": copies}
    salida = llm.json(PROMPT_QA, f"Evalúa estos copies.\nDATOS:{json.dumps(datos, ensure_ascii=False)}", temperatura=0.1)
    return {str(e["id"]): e for e in salida.get("evaluaciones", [])}


def verificar_limites(copy: dict, campos: dict[str, int]) -> list[str]:
    """Control determinista de longitud: se ejecuta siempre, aunque el modelo prometa haber cumplido."""
    return [f"{campo} excede {limite} ({len(str(copy.get(campo, '')))})"
            for campo, limite in campos.items() if len(str(copy.get(campo, ""))) > limite]


def ejecutar_pipeline(llm, brief_texto: str, canales: list[str], n_angulos: int, variantes: int,
                      paralelo: int = 4, avisar=None) -> Resultado:
    """Orquestador: analista -> estratega -> [copywriter en paralelo] -> [QA por canal] -> regenerar lo que falla."""
    t0 = time.time()
    r = Resultado()
    if avisar: avisar("analista", "Leyendo y estructurando el brief…")
    r.brief = agente_analista(llm, brief_texto)
    if avisar: avisar("estratega", f"Proponiendo {n_angulos} ángulos creativos…")
    r.angulos = agente_estratega(llm, r.brief, n_angulos)

    pendientes = {(c, i): variantes for c in canales for i in range(len(r.angulos))}
    contador = 0
    for ronda in range(1, RONDAS_MAX + 1):
        tareas = [(c, i, n) for (c, i), n in pendientes.items() if n > 0]
        if not tareas:
            break
        if avisar: avisar("copywriter", f"Ronda {ronda}: escribiendo {len(tareas)} lotes en paralelo…")
        lotes: dict[tuple, list] = {}
        with ThreadPoolExecutor(max_workers=max(1, paralelo)) as pool:
            futuros = {pool.submit(agente_copywriter, llm, r.brief, r.angulos[i], c, n): (c, i) for c, i, n in tareas}
            for k, fut in enumerate(as_completed(futuros), 1):
                c, i = futuros[fut]
                try:
                    lotes[(c, i)] = fut.result()
                except Exception:
                    lotes[(c, i)] = []
                if avisar: avisar("copywriter", f"Ronda {ronda}: {k}/{len(tareas)} lotes listos", k / len(tareas))

        # QA: un lote por canal con todos los copies de esa ronda
        if avisar: avisar("qa", f"Ronda {ronda}: revisando calidad y compliance por canal…")
        por_canal: dict[str, list] = {}
        meta: dict[str, tuple] = {}
        for (c, i), copies in lotes.items():
            for copy in copies:
                contador += 1
                cid = f"c{contador}"
                por_canal.setdefault(c, []).append({"id": cid, **{k: v for k, v in copy.items() if k != "variante"}})
                meta[cid] = (c, i)
        evaluaciones: dict[str, dict] = {}
        with ThreadPoolExecutor(max_workers=max(1, paralelo)) as pool:
            futs = [pool.submit(agente_qa, llm, r.brief, c, copies) for c, copies in por_canal.items()]
            for fut in as_completed(futs):
                try:
                    evaluaciones |= fut.result()
                except Exception:
                    pass

        for c, copies in por_canal.items():
            campos = CANALES[c]["campos"]
            for copy in copies:
                cid = copy["id"]; _, i = meta[cid]
                ev = evaluaciones.get(cid, {})
                errores = verificar_limites(copy, campos)
                puntuacion = int(ev.get("puntuacion", 0) or 0)
                claims = ev.get("claims_prohibidos_detectados") or []
                aprobado = puntuacion >= UMBRAL_QA and not errores and not claims and pendientes[(c, i)] > 0
                if aprobado:
                    pendientes[(c, i)] -= 1
                else:
                    r.descartados += 1
                r.filas.append({"id": cid, "canal": c, "angulo": r.angulos[i]["nombre"], "hook": r.angulos[i].get("hook", ""),
                                **{k: v for k, v in copy.items() if k != "id"},
                                "qa_puntuacion": puntuacion, "qa_comentario": ev.get("comentario", ""),
                                "claims_prohibidos": ", ".join(claims), "errores_formato": "; ".join(errores),
                                "estado": "aprobado" if aprobado else "descartado", "ronda": ronda})
    r.llamadas_llm = llm.llamadas
    r.segundos = time.time() - t0
    return r


# ---------------------------------------------------------------------------
# Lectura del brief
# ---------------------------------------------------------------------------
def leer_brief(archivo) -> str:
    nombre = archivo.name.lower(); datos = archivo.read()
    if nombre.endswith(".pdf"):
        from pypdf import PdfReader
        return "\n".join((p.extract_text() or "") for p in PdfReader(io.BytesIO(datos)).pages)
    if nombre.endswith(".docx"):
        import docx
        return "\n".join(p.text for p in docx.Document(io.BytesIO(datos)).paragraphs)
    return datos.decode("utf-8", errors="ignore")


@st.cache_data
def brief_ejemplo() -> str:
    ruta = BASE_DIR / "ejemplos" / "brief_ejemplo.md"
    return ruta.read_text(encoding="utf-8") if ruta.exists() else ""


# ---------------------------------------------------------------------------
# Estilo "app"
# ---------------------------------------------------------------------------
CSS = """
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
:root { --cm-ink:#0f172a; --cm-muted:#64748b; --cm-line:#e2e8f0; --cm-bg:#f8fafc; --cm-brand:#0f766e; --cm-brand2:#14b8a6; }
html, body, [class*="css"], .stApp { font-family: 'Inter', -apple-system, 'Segoe UI', sans-serif; }
#MainMenu, footer { visibility: hidden; height: 0; }
.block-container { padding-top: 1.2rem; padding-bottom: 3rem; max-width: 1280px; }
section[data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid var(--cm-line); }
section[data-testid="stSidebar"] h2 { font-size: .78rem; letter-spacing: .08em; text-transform: uppercase; color: var(--cm-muted); margin-top: 1.2rem; }
.cm-hero { background: linear-gradient(120deg, #0f766e 0%, #115e59 55%, #1e293b 100%); color: #fff; border-radius: 18px;
           padding: 26px 30px; margin-bottom: 18px; position: relative; overflow: hidden; }
.cm-hero:after { content:""; position:absolute; right:-60px; top:-60px; width:260px; height:260px; border-radius:50%;
                 background: radial-gradient(circle, rgba(20,184,166,.45), transparent 65%); }
.cm-hero h1 { margin: 0; font-size: 2rem; font-weight: 800; letter-spacing: -.02em; color: #fff; }
.cm-hero p { margin: 6px 0 0; opacity: .85; font-size: .95rem; }
.cm-pills { margin-top: 12px; display: flex; gap: 8px; flex-wrap: wrap; }
.cm-pill { background: rgba(255,255,255,.14); border: 1px solid rgba(255,255,255,.25); padding: 4px 10px; border-radius: 999px; font-size: .78rem; font-weight: 600; }
.cm-kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin: 6px 0 14px; }
.cm-kpi { background: #fff; border: 1px solid var(--cm-line); border-radius: 14px; padding: 14px 16px; box-shadow: 0 1px 2px rgba(15,23,42,.04); }
.cm-kpi .l { font-size: .72rem; color: var(--cm-muted); text-transform: uppercase; letter-spacing: .06em; font-weight: 600; }
.cm-kpi .v { font-size: 1.65rem; font-weight: 800; color: var(--cm-ink); line-height: 1.15; margin-top: 2px; }
.cm-kpi .s { font-size: .78rem; color: var(--cm-muted); }
.cm-agents { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin: 4px 0 12px; }
.cm-agent { background: #fff; border: 1px solid var(--cm-line); border-radius: 14px; padding: 12px 14px; }
.cm-agent .n { font-weight: 700; font-size: .9rem; color: var(--cm-ink); }
.cm-agent .d { font-size: .76rem; color: var(--cm-muted); margin-top: 2px; }
.cm-agent.on { border-color: var(--cm-brand2); box-shadow: 0 0 0 3px rgba(20,184,166,.15); }
.cm-agent.ok { border-color: #bbf7d0; background: #f0fdf4; }
.cm-agent .st { font-size: .72rem; font-weight: 700; margin-top: 8px; }
.cm-agent.on .st { color: var(--cm-brand); } .cm-agent.ok .st { color: #15803d; } .cm-agent.wait .st { color: #94a3b8; }
.cm-card { background: #fff; border: 1px solid var(--cm-line); border-radius: 14px; padding: 14px 16px; margin-bottom: 12px;
           box-shadow: 0 1px 2px rgba(15,23,42,.04); min-height: 150px; }
.cm-card .top { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; gap: 8px; }
.cm-chip { color: #fff; font-size: .7rem; font-weight: 700; padding: 3px 9px; border-radius: 999px; }
.cm-score { font-size: .72rem; font-weight: 700; padding: 3px 9px; border-radius: 999px; background: #ecfdf5; color: #047857; }
.cm-score.bad { background: #fef2f2; color: #b91c1c; }
.cm-field { margin: 6px 0; }
.cm-field .k { font-size: .68rem; color: var(--cm-muted); text-transform: uppercase; letter-spacing: .06em; font-weight: 600; }
.cm-field .t { font-size: .92rem; color: var(--cm-ink); line-height: 1.35; }
.cm-field .c { font-size: .68rem; color: #94a3b8; margin-left: 6px; }
.cm-angle { font-size: .74rem; color: var(--cm-muted); margin-top: 6px; }
.cm-note { font-size: .74rem; color: #b91c1c; margin-top: 6px; }
div[data-testid="stTextArea"] textarea { border-radius: 12px; }
button[kind="primary"] { border-radius: 12px; font-weight: 700; padding: .6rem 1rem; }
</style>
"""

AGENTES = [("analista", "Analista", "Lee y estructura el brief"),
           ("estratega", "Estratega", "Propone ángulos con hipótesis"),
           ("copywriter", "Copywriter", "Escribe los lotes por canal"),
           ("qa", "QA / Compliance", "Puntúa, bloquea claims, regenera")]


def html_agentes(activo: str | None, hechos: set[str]) -> str:
    cards = []
    for clave, nombre, desc in AGENTES:
        if clave in hechos: cls, estado = "ok", "✓ Completado"
        elif clave == activo: cls, estado = "on", "● Trabajando"
        else: cls, estado = "wait", "○ En espera"
        cards.append(f'<div class="cm-agent {cls}"><div class="n">{nombre}</div><div class="d">{desc}</div><div class="st">{estado}</div></div>')
    return f'<div class="cm-agents">{"".join(cards)}</div>'


def html_kpis(items: list[tuple[str, str, str]]) -> str:
    return '<div class="cm-kpis">' + "".join(
        f'<div class="cm-kpi"><div class="l">{l}</div><div class="v">{v}</div><div class="s">{s}</div></div>' for l, v, s in items) + "</div>"


def html_copy_card(fila: dict) -> str:
    spec = CANALES[fila["canal"]]
    campos = "".join(
        f'<div class="cm-field"><div class="k">{campo.replace("_", " ")}<span class="c">{len(str(fila.get(campo, "")))}/{lim}</span></div>'
        f'<div class="t">{html.escape(str(fila.get(campo, "")))}</div></div>' for campo, lim in spec["campos"].items())
    malo = fila["estado"] != "aprobado"
    nota = ""
    if malo:
        motivo = fila.get("claims_prohibidos") or fila.get("errores_formato") or fila.get("qa_comentario") or "QA bajo"
        nota = f'<div class="cm-note">Descartado: {html.escape(str(motivo))}</div>'
    return (f'<div class="cm-card"><div class="top"><span class="cm-chip" style="background:{spec["color"]}">{spec["corto"]}</span>'
            f'<span class="cm-score {"bad" if malo else ""}">QA {fila["qa_puntuacion"]}/10</span></div>'
            f'{campos}<div class="cm-angle">Ángulo · {html.escape(str(fila["angulo"]))}</div>{nota}</div>')


# ---------------------------------------------------------------------------
# Interfaz
# ---------------------------------------------------------------------------
st.set_page_config(page_title=f"{APP_NOMBRE} · copies de performance", page_icon="⚙️", layout="wide",
                   initial_sidebar_state="expanded")
st.markdown(CSS, unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## Motor de IA")
    try:
        clave_secreta = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        clave_secreta = ""
    clave = clave_secreta or st.text_input("Clave de Gemini (opcional, solo esta sesión)", type="password",
                                           help="Gratis en aistudio.google.com. No se guarda.")
    modo_demo = not clave
    if modo_demo:
        modelo = MODELOS_RESPALDO[0]
        st.info("Sin clave: **modo demo** con un LLM simulado. El flujo es el real; los textos, de muestra.")
    else:
        modelos = listar_modelos(clave)
        modelo = st.selectbox("Modelo", modelos, index=0, help="Lista obtenida de tu clave. Los nombres cambian con el tiempo.")
        otro = st.text_input("…u otro nombre de modelo (opcional)", placeholder="p. ej. gemini-3.8-flash")
        if otro.strip():
            modelo = otro.strip()
        st.success(f"Conectado · {modelo}")

    st.markdown("## Volumen")
    canales = st.multiselect("Canales", list(CANALES), default=list(CANALES)[:2],
                             format_func=lambda c: CANALES[c]["corto"] + " · " + c.split(" (")[0])
    n_angulos = st.slider("Ángulos creativos", 2, 8, 4)
    variantes = st.slider("Variantes por ángulo y canal", 2, 15, 5)

    st.markdown("## Velocidad")
    paralelo = st.slider("Llamadas en paralelo", 1, 8, 4,
                         help="Más rápido con más paralelo, pero el nivel gratuito de Gemini limita las peticiones por minuto. Si ves errores de cuota, bajalo.")
    objetivo = len(canales) * n_angulos * variantes
    llamadas_est = 2 + len(canales) * n_angulos + len(canales)
    st.markdown(html_kpis([("Copies objetivo", f"{objetivo}", "aprobados por QA"),
                           ("Llamadas", f"≈{llamadas_est}", "mínimo, más regeneraciones")]), unsafe_allow_html=True)

st.markdown(f"""
<div class="cm-hero">
  <h1>⚙️ {APP_NOMBRE}</h1>
  <p>Alto volumen de copies de performance a partir de un brief. Cuatro agentes trabajan en cadena y en paralelo; vos decidís y pulís.</p>
  <div class="cm-pills"><span class="cm-pill">Meta</span><span class="cm-pill">Google RSA</span><span class="cm-pill">TikTok</span>
  <span class="cm-pill">LinkedIn</span><span class="cm-pill">Email</span><span class="cm-pill">Display</span>
  <span class="cm-pill">{'Modo demo' if modo_demo else 'Gemini · ' + modelo}</span></div>
</div>""", unsafe_allow_html=True)

slot_agentes = st.empty()
resultado: Resultado | None = st.session_state.get("resultado")
slot_agentes.markdown(html_agentes(None, set(a[0] for a in AGENTES) if resultado else set()), unsafe_allow_html=True)

col_in, col_out = st.columns([1, 1.25], gap="large")
with col_in:
    with st.container(border=True):
        st.markdown("#### 1 · Brief")
        archivo = st.file_uploader("Cargá el brief", type=["pdf", "docx", "txt", "md"], label_visibility="collapsed")
        if archivo is not None and st.session_state.get("ultimo_archivo") != archivo.name:
            st.session_state["brief_area"] = leer_brief(archivo)
            st.session_state["ultimo_archivo"] = archivo.name
        if st.button("Usar brief de ejemplo"):
            st.session_state["brief_area"] = brief_ejemplo()
        brief_texto = st.text_area("…o pegalo / editalo aquí", height=300, key="brief_area",
                                   placeholder="Marca, producto, objetivo, audiencia, insight, oferta, tono, CTAs, claims prohibidos…")
        lanzar = st.button("🚀 Generar copies", type="primary", width="stretch",
                           disabled=not (brief_texto.strip() and canales))

if lanzar:
    llm = DemoLLM() if modo_demo else GeminiLLM(clave, modelo)
    hechos: set[str] = set()
    barra = st.progress(0.0, "Arrancando el pipeline…")
    orden = [a[0] for a in AGENTES]

    def avisar(agente: str, texto: str, fraccion: float | None = None):
        _ = hechos.update(orden[:orden.index(agente)])
        slot_agentes.markdown(html_agentes(agente, hechos), unsafe_allow_html=True)
        base = {"analista": 0.05, "estratega": 0.12, "copywriter": 0.2, "qa": 0.85}[agente]
        if agente == "copywriter" and fraccion is not None:
            base = 0.2 + 0.6 * fraccion
        barra.progress(min(base, 0.99), texto)

    try:
        resultado = ejecutar_pipeline(llm, brief_texto, canales, n_angulos, variantes, paralelo, avisar)
        st.session_state["resultado"] = resultado
        barra.progress(1.0, f"Listo en {resultado.segundos:.0f} s")
        slot_agentes.markdown(html_agentes(None, set(orden)), unsafe_allow_html=True)
        _ = time.sleep(0.6)
        _ = barra.empty()
    except Exception as e:
        barra.empty()
        st.error(f"El pipeline se detuvo: {e}")
        resultado = st.session_state.get("resultado")

with col_out:
    with st.container(border=True):
        st.markdown("#### 2 · Lo que entendieron los agentes")
        if not resultado:
            st.caption("Cargá un brief y pulsá **Generar copies**. Aquí aparecen el brief estructurado y los ángulos.")
        else:
            b = resultado.brief
            st.markdown(f"**{b.get('marca', '')} · {b.get('producto', '')}**")
            st.markdown(f"<span style='color:#64748b'>Objetivo</span> {html.escape(str(b.get('objetivo', '')))}<br>"
                        f"<span style='color:#64748b'>Audiencia</span> {html.escape(str(b.get('audiencia', '')))}<br>"
                        f"<span style='color:#64748b'>Insight</span> <i>{html.escape(str(b.get('insight', '')))}</i><br>"
                        f"<span style='color:#64748b'>Tono</span> {html.escape(str(b.get('tono', '')))}", unsafe_allow_html=True)
            if b.get("claims_prohibidos"):
                st.markdown("Claims bloqueados: " + " ".join(f"`{c}`" for c in b["claims_prohibidos"]))
            if b.get("preguntas_abiertas"):
                with st.expander("Lo que el analista echa en falta en el brief"):
                    for p in b["preguntas_abiertas"]:
                        st.markdown(f"- {p}")
            st.dataframe(pd.DataFrame(resultado.angulos), width="stretch", hide_index=True,
                         column_config={"nombre": "Ángulo", "hipotesis": "Hipótesis", "hook": "Hook"})

if resultado and resultado.filas:
    df = pd.DataFrame(resultado.filas)
    aprobados = df[df["estado"] == "aprobado"]
    st.markdown("#### 3 · Copies generados")
    st.markdown(html_kpis([
        ("Aprobados", f"{len(aprobados)}", f"de {objetivo} objetivo"),
        ("Descartados", f"{resultado.descartados}", "y regenerados por QA"),
        ("QA media", f"{aprobados['qa_puntuacion'].mean():.1f}" if len(aprobados) else "—", "sobre 10, aprobados"),
        ("Llamadas", f"{resultado.llamadas_llm}", f"en {resultado.segundos:.0f} s"),
        ("Canales", f"{aprobados['canal'].nunique()}", "con copies aprobados"),
    ]), unsafe_allow_html=True)

    f1, f2, f3 = st.columns([1.2, 1.2, .8])
    canal_f = f1.multiselect("Canal", sorted(df["canal"].unique()), default=sorted(df["canal"].unique()),
                             format_func=lambda c: CANALES[c]["corto"])
    angulo_f = f2.multiselect("Ángulo", sorted(df["angulo"].unique()), default=sorted(df["angulo"].unique()))
    ver_descartados = f3.toggle("Ver descartados", value=False)
    vista = df[df["canal"].isin(canal_f) & df["angulo"].isin(angulo_f)]
    if not ver_descartados:
        vista = vista[vista["estado"] == "aprobado"]

    tab_cards, tab_tabla = st.tabs([f"🗂️ Tarjetas ({len(vista)})", "📋 Tabla"])
    with tab_cards:
        if vista.empty:
            st.caption("Nada que mostrar con estos filtros.")
        else:
            maximo = st.slider("Mostrar hasta", 12, max(12, len(vista)), min(36, len(vista)), 12,
                               help="Para no saturar la página con cientos de tarjetas.") if len(vista) > 12 else len(vista)
            filas = vista.head(maximo).to_dict("records")
            cols = st.columns(3)
            for k, fila in enumerate(filas):
                cols[k % 3].markdown(html_copy_card(fila), unsafe_allow_html=True)
    with tab_tabla:
        st.dataframe(vista.drop(columns=["hook", "id"]), width="stretch", hide_index=True)

    csv = vista.to_csv(index=False).encode("utf-8-sig")
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as xw:
        for canal, grupo in vista.groupby("canal"):
            grupo.drop(columns=["id"]).to_excel(xw, sheet_name=canal[:31].replace("/", "-"), index=False)
    d1, d2 = st.columns(2)
    d1.download_button("⬇️ Descargar CSV", csv, "copies.csv", "text/csv", width="stretch")
    d2.download_button("⬇️ Descargar Excel (una hoja por canal)", buffer.getvalue(), "copies.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", width="stretch")
