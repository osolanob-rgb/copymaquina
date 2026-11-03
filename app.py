"""
Copymáquina · Generación agéntica de copies de performance a partir de un brief
Módulo Python para IA · Máster en Inteligencia Artificial · Oscar Solano Brenes

Pipeline de agentes (cada uno es una llamada al LLM con un rol y un contrato JSON):
  1. Analista de brief   -> brief estructurado (marca, objetivo, audiencia, insight, claims, tono...)
  2. Estratega           -> N ángulos creativos (dolor, beneficio, prueba social, urgencia...)
  3. Copywriter          -> por cada ángulo x canal, un lote de copies que respeta los límites del formato
  4. QA / Compliance     -> puntúa cada copy, marca claims prohibidos; los que fallan se regeneran (hasta 2 intentos)
  5. Exportación         -> tabla filtrable + CSV / Excel

Sin clave de API la app corre en MODO DEMO: el pipeline completo funciona con un LLM
simulado para que cualquiera pueda ver el flujo (es la versión 0 del reto).
"""

from __future__ import annotations

import io
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import streamlit as st

APP_NOMBRE = "Copymáquina"
BASE_DIR = Path(__file__).parent

# ---------------------------------------------------------------------------
# Formatos y límites de caracteres por canal (fuente: especificaciones públicas de cada plataforma)
# ---------------------------------------------------------------------------
CANALES: dict[str, dict] = {
    "Meta Ads (Facebook/Instagram)": {
        "campos": {"texto_principal": 125, "titulo": 40, "descripcion": 30},
        "nota": "Texto principal ≤125 (visible sin 'ver más'), título ≤40, descripción ≤30.",
    },
    "Google Ads (RSA)": {
        "campos": {"titulo": 30, "descripcion": 90},
        "nota": "Anuncios de búsqueda responsivos: títulos ≤30, descripciones ≤90. Sin emojis.",
    },
    "TikTok Ads": {
        "campos": {"texto": 100, "cta": 20},
        "nota": "Texto del anuncio ≤100; tono nativo, hablado, sin lenguaje corporativo.",
    },
    "LinkedIn Ads": {
        "campos": {"texto_intro": 150, "titulo": 70},
        "nota": "Introducción ≤150 (visible), título ≤70. Tono profesional, orientado a resultados.",
    },
    "Email (asunto + preheader)": {
        "campos": {"asunto": 50, "preheader": 90},
        "nota": "Asunto ≤50, preheader ≤90. Nada de MAYÚSCULAS completas ni signos repetidos.",
    },
    "Display / Banner": {
        "campos": {"titular": 35, "subtitular": 60, "cta": 15},
        "nota": "Titular ≤35, subtitular ≤60, botón ≤15.",
    },
}

ANGULOS_BASE = [
    "Dolor / problema", "Beneficio directo", "Prueba social", "Urgencia / escasez",
    "Curiosidad", "Comparación / antes-después", "Objeción resuelta", "Identidad / aspiración",
]

UMBRAL_QA = 7  # puntuación mínima (0-10) para aprobar un copy


# ---------------------------------------------------------------------------
# Cliente LLM: Gemini real o simulado (modo demo)
# ---------------------------------------------------------------------------
def _extraer_json(texto: str):
    """Quita fences ```json y devuelve el objeto. Si falla, intenta recortar al primer { o [."""
    limpio = re.sub(r"```(?:json)?", "", texto).strip().rstrip("`").strip()
    try:
        return json.loads(limpio)
    except json.JSONDecodeError:
        inicio = min([i for i in (limpio.find("{"), limpio.find("[")) if i >= 0] or [0])
        fin = max(limpio.rfind("}"), limpio.rfind("]")) + 1
        return json.loads(limpio[inicio:fin])


class GeminiLLM:
    """Llamadas a Gemini con salida JSON forzada. La clave nunca se escribe en el código:
    viene de st.secrets (en el despliegue) o de la barra lateral (sesión, no se guarda)."""

    def __init__(self, api_key: str, modelo: str = "gemini-2.0-flash"):
        from google import genai  # import tardío para que el modo demo no lo necesite
        self.cliente = genai.Client(api_key=api_key)
        self.modelo = modelo

    def json(self, sistema: str, usuario: str, temperatura: float = 0.9):
        from google.genai import types
        ultimo_error = None
        for intento in range(3):
            try:
                resp = self.cliente.models.generate_content(
                    model=self.modelo,
                    contents=usuario,
                    config=types.GenerateContentConfig(
                        system_instruction=sistema,
                        temperature=temperatura,
                        response_mime_type="application/json",
                    ),
                )
                return _extraer_json(resp.text)
            except Exception as e:  # cuota, red, JSON roto -> reintento con espera
                ultimo_error = e
                time.sleep(1.5 * (intento + 1))
        raise RuntimeError(f"Gemini no respondió tras 3 intentos: {ultimo_error}")


class DemoLLM:
    """Simula los 4 agentes sin red ni coste. Produce copies plausibles a partir del brief
    para que el flujo completo se pueda probar y desplegar (versión 0)."""

    def json(self, sistema: str, usuario: str, temperatura: float = 0.9):
        if "ANALISTA" in sistema:
            texto = usuario
            marca = _primera_coincidencia(r"(?i)marca\s*[:\-]\s*(.+)", texto) or "La marca"
            producto = _primera_coincidencia(r"(?i)producto\s*[:\-]\s*(.+)", texto) or "el producto"
            return {
                "marca": marca.strip(), "producto": producto.strip(),
                "objetivo": _primera_coincidencia(r"(?i)objetivo\s*[:\-]\s*(.+)", texto) or "Conversión",
                "audiencia": _primera_coincidencia(r"(?i)audiencia\s*[:\-]\s*(.+)", texto) or "Adultos 25-45",
                "insight": _primera_coincidencia(r"(?i)insight\s*[:\-]\s*(.+)", texto) or "Quieren resolverlo sin complicarse",
                "propuesta_de_valor": _primera_coincidencia(r"(?i)propuesta de valor\s*[:\-]\s*(.+)", texto) or "Más resultado, menos esfuerzo",
                "beneficios": ["Ahorro de tiempo", "Resultado comprobable", "Fácil de empezar"],
                "tono": _primera_coincidencia(r"(?i)tono\s*[:\-]\s*(.+)", texto) or "Directo y cercano",
                "cta_preferidos": _lista(_primera_coincidencia(r"(?i)cta\s*(?:preferidos?)?\s*[:\-]\s*(.+)", texto)) or ["Comprar ahora", "Probar gratis", "Saber más"],
                "claims_prohibidos": _lista(_primera_coincidencia(r"(?i)prohibid[oa]s?\s*[:\-]\s*(.+)", texto)) or ["el mejor del mercado", "garantizado"],
                "palabras_clave": ["oferta", "resultados", "hoy"],
                "oferta": _primera_coincidencia(r"(?i)oferta\s*[:\-]\s*(.+)", texto) or "",
                "preguntas_abiertas": ["No se indica presupuesto por canal"],
            }
        if "ESTRATEGA" in sistema:
            n = int(_primera_coincidencia(r"(\d+) ángulos", usuario) or 6)
            return {"angulos": [
                {"nombre": a, "hipotesis": f"La audiencia responde al enfoque de '{a.lower()}' porque conecta con el insight del brief.",
                 "hook": f"Enfoque {a.lower()}: por qué esto importa hoy"}
                for a in ANGULOS_BASE[:n]]}
        if "COPYWRITER" in sistema:
            datos = _extraer_json(usuario.split("DATOS:")[-1])
            campos = datos["campos"]; n = datos["cantidad"]; angulo = datos["angulo"]["nombre"]
            marca = datos["brief"]["marca"]
            producto = " ".join(datos["brief"]["producto"].split()[:4]).rstrip(",;(")  # nombre corto
            ctas = datos["brief"]["cta_preferidos"]
            prohibidos = datos["brief"].get("claims_prohibidos") or []
            semillas = {
                "texto_principal": "{marca}: {producto} pensado para quien no tiene tiempo que perder. {angulo}. {cta}.",
                "titulo": "{producto} que sí cumple",
                "descripcion": "{cta} con {marca}",
                "texto": "¿Todavía sin {producto}? {angulo}. {cta} 👇",
                "cta": "{cta}",
                "texto_intro": "Equipos que usan {producto} reportan mejores resultados en semanas. {angulo}. {cta}.",
                "asunto": "{producto}: {angulo}",
                "preheader": "{marca} te lo pone fácil. {cta}.",
                "titular": "{producto}, sin vueltas",
                "subtitular": "{angulo}. {cta} con {marca}.",
            }
            copies = []
            for i in range(n):
                cta = ctas[i % len(ctas)]
                c = {"variante": i + 1}
                for campo, limite in campos.items():
                    base = semillas.get(campo, "{producto}: {cta}").format(
                        marca=marca, producto=producto, angulo=angulo, cta=cta)
                    if i % 3 == 1: base = base.replace("hoy", "esta semana").replace("sin vueltas", "al grano")
                    if i % 3 == 2: base = base.replace("¿Todavía", "¿Aún")
                    # Para enseñar el bucle de QA, la 4ª variante del primer lote cuela un claim prohibido
                    if i == 3 and n >= 4 and prohibidos and campo == list(campos)[0]:
                        base = f"{prohibidos[0].capitalize()}: " + base
                    c[campo] = base[:limite]  # el demo siempre respeta el límite
                copies.append(c)
            return {"copies": copies}
        if "QA" in sistema:
            datos = _extraer_json(usuario.split("DATOS:")[-1])
            prohibidos = [p.lower() for p in datos["brief"].get("claims_prohibidos", [])]
            resultado = []
            for c in datos["copies"]:
                texto = " ".join(str(v) for k, v in c.items() if k != "variante").lower()
                hallados = [p for p in prohibidos if p in texto]
                score = 8 if not hallados else 4
                resultado.append({"variante": c["variante"], "puntuacion": score,
                                  "claims_prohibidos_detectados": hallados,
                                  "comentario": "OK" if not hallados else f"Contiene claim prohibido: {hallados}"})
            return {"evaluaciones": resultado}
        return {}


def _primera_coincidencia(patron: str, texto: str) -> str | None:
    m = re.search(patron, texto)
    return m.group(1).strip() if m else None


def _lista(texto: str | None) -> list[str]:
    return [t.strip() for t in re.split(r"[,;]", texto) if t.strip()] if texto else []


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
- Nunca uses los claims prohibidos. No inventes cifras ni premios. Usa el tono del brief y el idioma del brief.
- Nada de relleno genérico ("soluciones innovadoras", "líder del mercado").
Responde solo con JSON: {"copies": [{"variante": int, <un campo por cada campo pedido>}]}."""

PROMPT_QA = """Eres el revisor de QA y compliance de la agencia. Evalúas cada copy contra el brief: fidelidad al brief,
claridad, fuerza del CTA, ausencia de claims prohibidos o promesas no sustentadas, y adecuación al canal.
Puntúa de 0 a 10. Por debajo de 7 el copy se descarta.
Responde solo con JSON: {"evaluaciones": [{"variante": int, "puntuacion": int, "claims_prohibidos_detectados": [str], "comentario": str}]}."""


@dataclass
class Resultado:
    brief: dict = field(default_factory=dict)
    angulos: list = field(default_factory=list)
    filas: list = field(default_factory=list)
    descartados: int = 0
    llamadas_llm: int = 0


def agente_analista(llm, brief_texto: str) -> dict:
    return llm.json(PROMPT_ANALISTA, f"BRIEF:\n{brief_texto}", temperatura=0.2)


def agente_estratega(llm, brief: dict, n: int) -> list[dict]:
    salida = llm.json(PROMPT_ESTRATEGA,
                      f"Propón {n} ángulos. Lista de referencia (puedes adaptarla): {ANGULOS_BASE}.\nBRIEF:\n{json.dumps(brief, ensure_ascii=False)}")
    return salida.get("angulos", [])[:n]


def agente_copywriter(llm, brief: dict, angulo: dict, canal: str, cantidad: int) -> list[dict]:
    spec = CANALES[canal]
    datos = {"canal": canal, "nota_canal": spec["nota"], "campos": spec["campos"],
             "cantidad": cantidad, "angulo": angulo, "brief": brief}
    salida = llm.json(PROMPT_COPYWRITER,
                      f"Escribe {cantidad} variantes para el canal {canal}.\nDATOS:{json.dumps(datos, ensure_ascii=False)}")
    return salida.get("copies", [])


def agente_qa(llm, brief: dict, canal: str, copies: list[dict]) -> dict[int, dict]:
    datos = {"canal": canal, "brief": brief, "copies": copies}
    salida = llm.json(PROMPT_QA, f"Evalúa estos copies.\nDATOS:{json.dumps(datos, ensure_ascii=False)}", temperatura=0.1)
    return {int(e["variante"]): e for e in salida.get("evaluaciones", [])}


def verificar_limites(copy: dict, campos: dict[str, int]) -> list[str]:
    """Control determinista: el LLM promete respetar límites, pero se comprueba siempre."""
    return [f"{campo} excede {limite} ({len(str(copy.get(campo, '')))})"
            for campo, limite in campos.items() if len(str(copy.get(campo, ""))) > limite]


def ejecutar_pipeline(llm, brief_texto: str, canales: list[str], n_angulos: int,
                      variantes: int, progreso=None) -> Resultado:
    r = Resultado()
    r.brief = agente_analista(llm, brief_texto); r.llamadas_llm += 1
    r.angulos = agente_estratega(llm, r.brief, n_angulos); r.llamadas_llm += 1

    total = len(canales) * len(r.angulos)
    hecho = 0
    for canal in canales:
        campos = CANALES[canal]["campos"]
        for angulo in r.angulos:
            pendientes = variantes
            intentos = 0
            while pendientes > 0 and intentos < 3:  # bucle agéntico: regenera lo que falla
                intentos += 1
                copies = agente_copywriter(llm, r.brief, angulo, canal, pendientes); r.llamadas_llm += 1
                if not copies:
                    break
                evaluaciones = agente_qa(llm, r.brief, canal, copies); r.llamadas_llm += 1
                for c in copies:
                    ev = evaluaciones.get(int(c.get("variante", 0)), {})
                    errores = verificar_limites(c, campos)
                    puntuacion = int(ev.get("puntuacion", 0))
                    aprobado = puntuacion >= UMBRAL_QA and not errores and not ev.get("claims_prohibidos_detectados")
                    if aprobado:
                        pendientes -= 1
                    else:
                        r.descartados += 1
                    fila = {"canal": canal, "angulo": angulo["nombre"], "hook": angulo.get("hook", ""),
                            **{k: v for k, v in c.items() if k != "variante"},
                            "qa_puntuacion": puntuacion, "qa_comentario": ev.get("comentario", ""),
                            "claims_prohibidos": ", ".join(ev.get("claims_prohibidos_detectados", [])),
                            "errores_formato": "; ".join(errores), "estado": "aprobado" if aprobado else "descartado",
                            "intento": intentos}
                    r.filas.append(fila)
            hecho += 1
            if progreso:
                progreso(hecho / total, f"{canal} · {angulo['nombre']}")
    return r


# ---------------------------------------------------------------------------
# Lectura del brief (PDF, DOCX, TXT/MD o texto pegado)
# ---------------------------------------------------------------------------
def leer_brief(archivo) -> str:
    nombre = archivo.name.lower()
    datos = archivo.read()
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
# Interfaz
# ---------------------------------------------------------------------------
st.set_page_config(page_title=f"{APP_NOMBRE} · copies de performance", page_icon="⚙️", layout="wide")
st.title(f"⚙️ {APP_NOMBRE}")
st.caption("Alto volumen de copies de performance a partir de un brief · pipeline de 4 agentes: analista → estratega → copywriter → QA")

with st.sidebar:
    st.header("Motor de IA")
    try:
        clave_secreta = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:  # sin secrets.toml (local o demo)
        clave_secreta = ""
    clave = clave_secreta or st.text_input("Clave de Gemini (opcional, solo esta sesión)", type="password",
                                           help="Se obtiene gratis en aistudio.google.com. No se guarda.")
    modelo = st.selectbox("Modelo", ["gemini-2.0-flash", "gemini-2.5-flash"], index=0)
    modo_demo = not clave
    if modo_demo:
        st.info("Sin clave: **modo demo** con un LLM simulado. El flujo es el real; los textos, de muestra.")
    else:
        st.success("Gemini conectado.")

    st.header("Volumen")
    canales = st.multiselect("Canales", list(CANALES), default=list(CANALES)[:2])
    n_angulos = st.slider("Ángulos creativos", 2, 8, 4)
    variantes = st.slider("Variantes por ángulo y canal", 2, 15, 5)
    estimado = len(canales) * n_angulos * variantes
    st.metric("Copies aprobados objetivo", estimado)
    st.caption(f"≈ {2 + 2 * len(canales) * n_angulos}–{2 + 6 * len(canales) * n_angulos} llamadas al modelo.")

col_in, col_out = st.columns([1, 1.4])
with col_in:
    st.subheader("1 · Brief")
    archivo = st.file_uploader("Cargá el brief (PDF, DOCX, TXT o MD)", type=["pdf", "docx", "txt", "md"])
    if archivo is not None and st.session_state.get("ultimo_archivo") != archivo.name:
        st.session_state["brief_area"] = leer_brief(archivo)   # se vuelca al editor al subirlo
        st.session_state["ultimo_archivo"] = archivo.name
    if st.button("Usar brief de ejemplo"):
        st.session_state["brief_area"] = brief_ejemplo()
    brief_texto = st.text_area("…o pegalo / editalo aquí", height=320, key="brief_area")
    lanzar = st.button("🚀 Generar copies", type="primary", disabled=not (brief_texto.strip() and canales), width="stretch")

if lanzar:
    llm = DemoLLM() if modo_demo else GeminiLLM(clave, modelo)
    barra = st.progress(0.0, "Analizando el brief…")
    try:
        resultado = ejecutar_pipeline(llm, brief_texto, canales, n_angulos, variantes,
                                      progreso=lambda f, t: barra.progress(f, f"Generando · {t}"))
        st.session_state["resultado"] = resultado
        barra.progress(1.0, "Listo")
    except Exception as e:
        barra.empty()
        st.error(f"El pipeline se detuvo: {e}")

resultado: Resultado | None = st.session_state.get("resultado")
with col_out:
    st.subheader("2 · Brief estructurado y ángulos")
    if not resultado:
        st.info("Cargá un brief y pulsá **Generar copies**. Cada agente deja su rastro aquí.")
    else:
        b = resultado.brief
        st.markdown(f"**{b.get('marca', '')} · {b.get('producto', '')}** — objetivo: {b.get('objetivo', '')}")
        st.markdown(f"Audiencia: {b.get('audiencia', '')}  \nInsight: *{b.get('insight', '')}*  \nTono: {b.get('tono', '')}")
        if b.get("claims_prohibidos"):
            st.markdown("Claims prohibidos: " + ", ".join(f"`{c}`" for c in b["claims_prohibidos"]))
        if b.get("preguntas_abiertas"):
            with st.expander("Lo que el analista echa en falta en el brief"):
                for p in b["preguntas_abiertas"]:
                    st.markdown(f"- {p}")
        st.dataframe(pd.DataFrame(resultado.angulos), width="stretch", hide_index=True)

if resultado:
    st.divider()
    st.subheader("3 · Copies generados")
    df = pd.DataFrame(resultado.filas)
    aprobados = df[df["estado"] == "aprobado"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Aprobados por QA", len(aprobados))
    m2.metric("Descartados y regenerados", resultado.descartados)
    m3.metric("QA media (aprobados)", f"{aprobados['qa_puntuacion'].mean():.1f}/10" if len(aprobados) else "—")
    m4.metric("Llamadas al modelo", resultado.llamadas_llm)

    f1, f2, f3 = st.columns(3)
    canal_f = f1.multiselect("Canal", sorted(df["canal"].unique()), default=sorted(df["canal"].unique()))
    angulo_f = f2.multiselect("Ángulo", sorted(df["angulo"].unique()), default=sorted(df["angulo"].unique()))
    ver_descartados = f3.toggle("Mostrar descartados", value=False)
    vista = df[df["canal"].isin(canal_f) & df["angulo"].isin(angulo_f)]
    if not ver_descartados:
        vista = vista[vista["estado"] == "aprobado"]
    st.dataframe(vista.drop(columns=["hook"]), width="stretch", hide_index=True)

    csv = vista.to_csv(index=False).encode("utf-8-sig")
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as xw:
        for canal, grupo in vista.groupby("canal"):
            grupo.to_excel(xw, sheet_name=canal[:31].replace("/", "-"), index=False)
    d1, d2 = st.columns(2)
    d1.download_button("⬇️ Descargar CSV", csv, "copies.csv", "text/csv", width="stretch")
    d2.download_button("⬇️ Descargar Excel (una hoja por canal)", buffer.getvalue(), "copies.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", width="stretch")
