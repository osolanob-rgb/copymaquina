# ⚙️ Copymáquina — copies de performance en volumen a partir de un brief

Oscar Solano Brenes · Reto del módulo Python para IA · Máster en Inteligencia Artificial

Cargás un brief, elegís canales y volumen, y un pipeline de cuatro agentes
(analista → estratega → copywriter → QA) devuelve el lote de copies revisado, con
límites de caracteres por formato y claims prohibidos bloqueados, descargable en Excel/CSV.

## Archivos
| Archivo | Qué es |
|---|---|
| `app.py` | La app completa |
| `ejemplos/brief_ejemplo.md` | Brief ficticio para probar |
| `requirements.txt` | Dependencias |
| `.streamlit/config.toml` | Tema y límite de subida |
| `.streamlit/secrets.toml.example` | Plantilla para la clave de Gemini (no subir el `.toml` real) |
| `ESPECIFICACION.md` | Especificación C.1 (6 secciones) |
| `AGENTS.md` | Contexto e instrucciones para agentes de programación |

## Probar en local
```bash
pip install -r requirements.txt
streamlit run app.py
```
Sin clave arranca en **modo demo** (LLM simulado, flujo completo). Para usar Gemini:
clave gratuita en https://aistudio.google.com → pegala en la barra lateral, o copiá
`.streamlit/secrets.toml.example` a `.streamlit/secrets.toml` con tu clave.

## Publicar en Streamlit Community Cloud (gratis)
1. Repo **público** en GitHub con todos los archivos (incluida la carpeta `.streamlit/`
   pero **sin** `secrets.toml`; el `.gitignore` ya lo excluye).
2. https://share.streamlit.io → iniciar sesión con GitHub → **Create app** → elegir el
   repo, rama `main`, *Main file path* `app.py`, Python 3.12.
3. **Deploy**. Tarda 1–2 minutos. La URL queda como `https://<nombre>.streamlit.app`.
4. Para activar Gemini en la app publicada: **Manage app → Settings → Secrets** y pegar
   `GEMINI_API_KEY = "tu-clave"`. Sin eso la app pública funciona en modo demo, que es
   suficiente para la versión 0 del hito.
5. Abrí la URL en una ventana privada: ese enlace va en la ficha del hito de la semana 2.

Cada `git push` a `main` redespliega automáticamente.

## Ficha del hito (5 líneas)
- **Problema**: los equipos de performance necesitan cientos de variantes de copy por campaña y las escriben a mano, sin control sistemático de límites ni de claims.
- **Usuario**: copywriters y estrategas de performance de cualquier agencia de publicidad.
- **Qué hará la app**: brief → ángulos → copies por canal y formato → QA/compliance → Excel, en minutos.
- **Pieza de IA**: API de Gemini orquestada como 4 agentes con salida JSON y bucle de regeneración.
- **Versión 0**: `https://<nombre>.streamlit.app` (modo demo sin clave; con clave, Gemini real).
