# Especificación: Copymáquina — copies de performance en volumen a partir de un brief

## 1. Problema y usuario
Los equipos de performance de una agencia de publicidad necesitan decenas o cientos de
variantes de copy por campaña (varios canales × varios ángulos × varias versiones) para
alimentar los tests A/B de Meta, Google, TikTok, LinkedIn, email y display. Hoy un
copywriter las escribe a mano a partir del brief, en una hoja de cálculo, cuidando a ojo
los límites de caracteres y los claims que legal no permite. Es lento, repetitivo, y las
variantes acaban pareciéndose entre sí. El usuario es el copywriter o el estratega de
performance de cualquier agencia; el cliente interno es el equipo de medios que monta
los anuncios. La IA resuelve la parte mecánica: leer el brief, proponer ángulos,
escribir las variantes dentro de cada formato y revisarlas contra el brief, para que el
humano decida y pula en vez de teclear.

## 2. Qué hace la app
Cargás el brief de la campaña (PDF, Word o texto), elegís canales, cuántos ángulos
creativos y cuántas variantes por ángulo, y pulsás un botón. Un pipeline de cuatro
agentes (analista de brief → estratega → copywriter → QA/compliance) produce el lote de
copies ya revisado, con puntuación y motivo de descarte, y lo descargás en Excel o CSV
listo para cargar en el gestor de anuncios.

## 3. Entradas y salidas
- **Entrada**: el brief (archivo .pdf/.docx/.txt/.md o texto pegado), la lista de
  canales, el número de ángulos (2–8) y de variantes por ángulo y canal (2–15).
  Opcionalmente, una clave de Gemini en la barra lateral (nunca se guarda).
- **Salida**: (a) el brief estructurado que entendió el analista, con lo que falta en
  el brief; (b) la tabla de ángulos con su hipótesis; (c) la tabla de copies con canal,
  ángulo, cada campo del formato, puntuación de QA, comentario, claims prohibidos
  detectados y estado aprobado/descartado; (d) descarga CSV y Excel (una hoja por canal).
- **Ejemplo**: brief de "Verdelab, kit de huerta en balcón, oferta primer mes a $9,
  claims prohibidos: garantizado, 100% orgánico certificado" + Meta Ads + 4 ángulos ×
  5 variantes → 20 copies aprobados de Meta (texto principal ≤125, título ≤40,
  descripción ≤30), por ejemplo título "Huerta que no se te muere" y CTA "Probá el
  primer mes a $9"; las variantes que mencionan "garantizado" aparecen como descartadas
  y regeneradas.

## 4. Qué pieza de IA usa
La API de Gemini (modelos `gemini-2.0-flash` / `gemini-2.5-flash`) con salida JSON
forzada y razonamiento interno desactivado (más rápido), orquestada como cuatro agentes
con roles y contratos distintos: los lotes del copywriter se piden en paralelo (hasta 8
llamadas simultáneas), QA revisa un lote por canal, y lo que QA rechaza o excede el límite
de caracteres (comprobado también en código, no solo por el modelo) se vuelve a generar
hasta en tres rondas. La lista de modelos se consulta a la API con la clave del usuario,
para que los cambios de nombre de Google no rompan la app. Se eligió un LLM
por API y no un modelo de Hugging Face porque la tarea es generación de texto creativo
en español con instrucciones complejas, donde un modelo pequeño local no da calidad;
Gemini tiene nivel gratuito, es el asistente del curso y su modo JSON simplifica el
pipeline. Sin clave, la app corre en modo demo con un LLM simulado que reproduce el
flujo completo (es la versión 0, sin IA real).

## 5. Restricciones
- Coste cero: Streamlit Community Cloud + nivel gratuito de Gemini; modo demo sin clave.
- Ninguna clave en el código: se lee de `st.secrets` en el despliegue o se escribe en la
  barra lateral solo para la sesión. `secrets.toml` está en `.gitignore`.
- Ningún dato personal: los briefs de ejemplo son ficticios; la app no almacena nada.
- Usable por alguien no técnico: subir archivo, tres controles, un botón, descargar.
- Límites de caracteres por formato codificados en la app y verificados de forma
  determinista; los claims prohibidos del brief se bloquean en QA.
- Ruta de despliegue: Streamlit Community Cloud desde un repo público de GitHub.

## 6. Criterios de aceptación
1. Con el brief de ejemplo, 2 canales, 4 ángulos y 5 variantes, la app termina en menos
   de 90 segundos con Gemini (menos de 10 segundos en modo demo) y muestra 40 copies
   aprobados, con el rastreador de los 4 agentes marcando cada etapa.
2. Ningún copy aprobado supera el límite de caracteres de su campo (verificable en la
   columna `errores_formato`, que debe estar vacía en los aprobados).
3. Ningún copy aprobado contiene un claim de la lista de prohibidos del brief; los que
   lo contienen aparecen como descartados con el claim identificado.
4. El Excel descargado tiene una hoja por canal y las mismas columnas que la tabla.
5. Al subir un brief en PDF, el brief estructurado muestra marca, producto y objetivo
   coherentes con el documento.
6. La URL pública abre en una ventana privada del navegador y, sin clave, genera en
   modo demo sin errores.
