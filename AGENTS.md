# AGENTS.md

## Qué es esta app
Copymáquina: app de Streamlit que genera lotes grandes de copies de marketing de
performance a partir de un brief, mediante un pipeline de cuatro agentes LLM
(analista de brief → estratega → copywriter → QA/compliance) con un bucle de
regeneración. Proyecto del reto del módulo Python para IA (Máster en IA).

## Estructura
- `app.py` — todo: límites por canal, cliente LLM (Gemini real y `DemoLLM`), los cuatro
  agentes, el orquestador `ejecutar_pipeline`, lectura de briefs y la interfaz.
- `ejemplos/brief_ejemplo.md` — brief ficticio para demos y pruebas.
- `requirements.txt`, `.streamlit/config.toml`, `.streamlit/secrets.toml.example`.
- `ESPECIFICACION.md` — especificación de 6 secciones; los criterios de aceptación
  son la prueba de referencia.

## Cómo se ejecuta
```bash
pip install -r requirements.txt
streamlit run app.py
```
Sin `GEMINI_API_KEY` en `.streamlit/secrets.toml` la app arranca en modo demo.
Para usar Gemini en local: copiá `secrets.toml.example` a `secrets.toml` y poné la clave.

## Convenciones
- Código, comentarios e interfaz en español; nombres en español (`brief`, `angulo`,
  `copies`, `ejecutar_pipeline`).
- Cada agente es una función `agente_*(llm, ...)` con un prompt de sistema en una
  constante `PROMPT_*` que empieza por su rol en mayúsculas (ANALISTA, ESTRATEGA,
  COPYWRITER, QA). `DemoLLM` enruta por esa palabra: si renombrás un rol, actualizá ambos.
- Todos los agentes devuelven JSON; se parsea con `_extraer_json`, nunca con regex sobre
  prosa.
- Los límites de caracteres viven solo en el diccionario `CANALES`. Para añadir un canal
  se añade una entrada ahí; no hay que tocar los agentes.
- `verificar_limites` es la comprobación determinista de longitud: se ejecuta siempre,
  aunque el modelo diga que cumplió. No la quites.
- Usar `width="stretch"` en vez de `use_container_width` (Streamlit ≥ 1.50).
- Nuevas funciones del LLM: implementarlas en `GeminiLLM` **y** en `DemoLLM` para que el
  modo demo siga cubriendo el flujo completo.

## Qué no tocar
- Nunca escribas una clave de API en el código ni en archivos versionados;
  `.streamlit/secrets.toml` está en `.gitignore` y debe seguir ahí.
- No guardes briefs ni copies en disco ni en servicios externos: la app es sin estado.
- No subas briefs reales de clientes al repo: solo el ejemplo ficticio.
- `UMBRAL_QA = 7` y el máximo de 3 intentos por lote son decisiones de producto; si se
  cambian, actualizá `ESPECIFICACION.md`.

## Cómo probar
```bash
python -c "
from streamlit.testing.v1 import AppTest
at = AppTest.from_file('app.py', default_timeout=120).run()
at.button[0].click().run(); at.button[1].click().run()
assert not at.exception, at.exception
print([(m.label, m.value) for m in at.metric])"
```
Debe terminar sin excepciones con 40 aprobados (2 canales × 4 ángulos × 5 variantes).
