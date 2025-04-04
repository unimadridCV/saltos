import streamlit as st
import cv2
import numpy as np
import pandas as pd # Para la tabla de datos
import math         # Para cálculos de ángulos
from ultralytics import YOLO
import tempfile
import time

# --- Configuración ---
RUTA_MODELO = 'yolov8n-pose.pt'
UMBRAL_CONFIANZA_PERSONA = 0.5
UMBRAL_CONFIANZA_KEYPOINT = 0.4 # Umbral ligeramente más alto puede ayudar a estabilizar ángulos
UMBRAL_VELOCIDAD_SALTO_ARRIBA = 8
UMBRAL_VELOCIDAD_SALTO_ABAJO = -1
MIN_FOTOGRAMAS_EN_AIRE = 3
ANCHO_VIDEO_MOSTRAR = 720 # Ancho en píxeles para mostrar el video en Streamlit

# Índices de Keypoints (COCO dataset)
IDX_TOBILLO_IZQ = 15
IDX_TOBILLO_DER = 16
IDX_RODILLA_IZQ = 13
IDX_RODILLA_DER = 14
IDX_CADERA_IZQ = 11
IDX_CADERA_DER = 12

# --- Variables de Estado ---
if 'saltando' not in st.session_state:
    st.session_state.saltando = False
if 'fotogramas_en_aire' not in st.session_state:
    st.session_state.fotogramas_en_aire = 0
if 'contador_saltos' not in st.session_state:
    st.session_state.contador_saltos = 0
if 'y_tobillo_anterior' not in st.session_state:
    st.session_state.y_tobillo_anterior = None
if 'ultimo_fotograma_salto' not in st.session_state:
    st.session_state.ultimo_fotograma_salto = -100
if 'datos_historicos' not in st.session_state:
     st.session_state.datos_historicos = [] # Para almacenar datos de la tabla

# --- Funciones Auxiliares ---
@st.cache_resource
def cargar_modelo(ruta_modelo):
    try:
        modelo = YOLO(ruta_modelo)
        return modelo
    except Exception as e:
        st.error(f"Error al cargar el modelo YOLO-Pose: {e}")
        return None

def reiniciar_estado():
    st.session_state.saltando = False
    st.session_state.fotogramas_en_aire = 0
    st.session_state.contador_saltos = 0
    st.session_state.y_tobillo_anterior = None
    st.session_state.ultimo_fotograma_salto = -100
    st.session_state.datos_historicos = [] # Reiniciar datos de tabla

def calcular_angulo(p1, p2, p3):
    """Calcula el ángulo (en grados) formado en p2 por p1-p2-p3."""
    if p1 is None or p2 is None or p3 is None:
        return None # No se puede calcular si falta un punto

    v1 = (p1[0] - p2[0], p1[1] - p2[1])
    v2 = (p3[0] - p2[0], p3[1] - p2[1])

    dot_product = v1[0] * v2[0] + v1[1] * v2[1]
    mag_v1 = math.sqrt(v1[0]**2 + v1[1]**2)
    mag_v2 = math.sqrt(v2[0]**2 + v2[1]**2)

    if mag_v1 * mag_v2 == 0:
        return None # Evitar división por cero si los puntos coinciden

    cos_theta = dot_product / (mag_v1 * mag_v2)
    # Asegurar que el valor esté en el rango [-1, 1] por errores de precisión
    cos_theta = max(-1.0, min(1.0, cos_theta))

    angulo_rad = math.acos(cos_theta)
    angulo_deg = math.degrees(angulo_rad)
    return angulo_deg

def obtener_punto_confiable(kpts_xy, kpts_conf, idx):
    """Devuelve las coordenadas (x, y) si el keypoint es confiable, sino None."""
    if idx < len(kpts_conf) and kpts_conf[idx] >= UMBRAL_CONFIANZA_KEYPOINT:
        return kpts_xy[idx]
    return None

def obtener_y_tobillo_mas_bajo(kpts_xy, kpts_conf):
    """
    Encuentra la coordenada Y del tobillo visible más bajo.
    Retorna None si ningún tobillo es visible con suficiente confianza.
    """
    y_tobillo_actual = None
    # Obtener las coordenadas [x, y] o None para cada tobillo
    punto_izq = obtener_punto_confiable(kpts_xy, kpts_conf, IDX_TOBILLO_IZQ)
    punto_der = obtener_punto_confiable(kpts_xy, kpts_conf, IDX_TOBILLO_DER)

    # --- CORRECCIÓN ---
    # Comprobar si el punto NO es None antes de acceder a la coordenada Y ([1])
    y_izq_val = punto_izq[1] if punto_izq is not None else -float('inf')
    y_der_val = punto_der[1] if punto_der is not None else -float('inf')
    # --- FIN CORRECCIÓN ---

    # Queremos la Y MÁS GRANDE (más abajo en la imagen)
    max_y = max(y_izq_val, y_der_val)

    # Si max_y sigue siendo -inf, significa que ningún tobillo era visible
    if max_y > -float('inf'):
         return max_y # Devolver la coordenada Y más baja (mayor valor)
    else:
         return None # Ningún tobillo visible

# --- Procesamiento Principal ---
def procesar_video(ruta_video, modelo, texto_estado, espacio_fotograma, espacio_tabla, barra_progreso):
    cap = cv2.VideoCapture(ruta_video)
    if not cap.isOpened():
        st.error("Error al abrir el archivo de video.")
        return

    ancho_fotograma = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    alto_fotograma = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_fotogramas = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    reiniciar_estado()
    contador_fotogramas = 0
    tiempo_inicio = time.time()

    while cap.isOpened():
        ret, fotograma = cap.read()
        if not ret:
            break

        contador_fotogramas += 1
        fotograma_anotado = fotograma.copy()
        tiempo_actual = contador_fotogramas / fps

        # --- Estimación de Pose ---
        resultados = modelo(fotograma, verbose=False, conf=UMBRAL_CONFIANZA_PERSONA)

        y_tobillo_actual = None
        altura_relativa = None # Altura desde el borde inferior de la imagen
        angulo_izq = None
        angulo_der = None

        if len(resultados) > 0 and resultados[0].keypoints is not None:
            res = resultados[0]
            if res.keypoints.shape[1] > 0:
                kpts_xy = res.keypoints.xy[0].cpu().numpy()
                kpts_conf = res.keypoints.conf[0].cpu().numpy()

                # Obtener puntos clave necesarios si son confiables
                tobillo_izq = obtener_punto_confiable(kpts_xy, kpts_conf, IDX_TOBILLO_IZQ)
                rodilla_izq = obtener_punto_confiable(kpts_xy, kpts_conf, IDX_RODILLA_IZQ)
                cadera_izq = obtener_punto_confiable(kpts_xy, kpts_conf, IDX_CADERA_IZQ)
                tobillo_der = obtener_punto_confiable(kpts_xy, kpts_conf, IDX_TOBILLO_DER)
                rodilla_der = obtener_punto_confiable(kpts_xy, kpts_conf, IDX_RODILLA_DER)
                cadera_der = obtener_punto_confiable(kpts_xy, kpts_conf, IDX_CADERA_DER)

                # Calcular ángulos
                angulo_izq = calcular_angulo(cadera_izq, rodilla_izq, tobillo_izq)
                angulo_der = calcular_angulo(cadera_der, rodilla_der, tobillo_der)

                # Calcular altura (basada en el tobillo más bajo)
                y_tobillo_actual = obtener_y_tobillo_mas_bajo(kpts_xy, kpts_conf)
                if y_tobillo_actual is not None:
                    # Invertir: más alto en la imagen (menor Y) -> mayor altura
                    altura_relativa = alto_fotograma - y_tobillo_actual

                # Dibujar pose (sobre fotograma_anotado)
                fotograma_anotado = res.plot(boxes=False, img=fotograma_anotado) # No dibujar cajas, solo pose

        # --- Lógica de Detección de Salto (sin cambios) ---
        velocidad_vertical = 0
        if y_tobillo_actual is not None:
            if st.session_state.y_tobillo_anterior is not None:
                velocidad_vertical = y_tobillo_actual - st.session_state.y_tobillo_anterior
                if not st.session_state.saltando and \
                   velocidad_vertical < -UMBRAL_VELOCIDAD_SALTO_ARRIBA and \
                   (contador_fotogramas - st.session_state.ultimo_fotograma_salto > fps / 2):
                    st.session_state.saltando = True
                    st.session_state.fotogramas_en_aire = 1
                elif st.session_state.saltando:
                    if velocidad_vertical < UMBRAL_VELOCIDAD_SALTO_ABAJO:
                        st.session_state.fotogramas_en_aire += 1
                    else:
                        if st.session_state.fotogramas_en_aire >= MIN_FOTOGRAMAS_EN_AIRE:
                            st.session_state.contador_saltos += 1
                            st.session_state.ultimo_fotograma_salto = contador_fotogramas
                        st.session_state.saltando = False
                        st.session_state.fotogramas_en_aire = 0
            st.session_state.y_tobillo_anterior = y_tobillo_actual
        else:
            st.session_state.y_tobillo_anterior = None
            if st.session_state.saltando:
                if st.session_state.fotogramas_en_aire >= MIN_FOTOGRAMAS_EN_AIRE:
                    st.session_state.contador_saltos += 1
                    st.session_state.ultimo_fotograma_salto = contador_fotogramas
                st.session_state.saltando = False
                st.session_state.fotogramas_en_aire = 0

        # --- Almacenar y Mostrar Datos ---
        # Añadir datos actuales a la lista histórica
        st.session_state.datos_historicos.append({
            "Tiempo (s)": round(tiempo_actual, 2),
            "Ángulo Izq (°)": round(angulo_izq, 1) if angulo_izq is not None else np.nan,
            "Ángulo Der (°)": round(angulo_der, 1) if angulo_der is not None else np.nan,
            "Altura Tobillo (px)": round(altura_relativa) if altura_relativa is not None else np.nan
        })

        # Crear DataFrame y mostrarlo (actualizar en cada frame)
        df_datos = pd.DataFrame(st.session_state.datos_historicos)
        espacio_tabla.dataframe(df_datos.tail(15)) # Mostrar las últimas 15 filas para no saturar

        # --- Mostrar Fotograma ---
        # Dibujar info adicional sobre el fotograma
        texto_estado_mostrar = f"Saltando: {'Sí' if st.session_state.saltando else 'No'}"
        texto_contador_saltos = f"Saltos: {st.session_state.contador_saltos}"
        texto_angulo_izq = f"Ang Izq: {round(angulo_izq, 1)} deg" if angulo_izq is not None else "Ang Izq: N/A"
        texto_angulo_der = f"Ang Der: {round(angulo_der, 1)} deg" if angulo_der is not None else "Ang Der: N/A"
        texto_altura = f"Altura: {round(altura_relativa)} px" if altura_relativa is not None else "Altura: N/A"

        # Posicionar textos
        y_pos = 30
        cv2.putText(fotograma_anotado, texto_estado_mostrar, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        y_pos += 30
        cv2.putText(fotograma_anotado, texto_contador_saltos, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        y_pos += 30
        cv2.putText(fotograma_anotado, texto_angulo_izq, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2) # Verde
        y_pos += 30
        cv2.putText(fotograma_anotado, texto_angulo_der, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2) # Verde
        y_pos += 30
        cv2.putText(fotograma_anotado, texto_altura, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2) # Azul


        # Redimensionar fotograma para mostrar
        nuevo_alto = int(alto_fotograma * (ANCHO_VIDEO_MOSTRAR / ancho_fotograma))
        fotograma_redimensionado = cv2.resize(fotograma_anotado, (ANCHO_VIDEO_MOSTRAR, nuevo_alto), interpolation=cv2.INTER_AREA)

        # Convertir a RGB y mostrar
        fotograma_rgb = cv2.cvtColor(fotograma_redimensionado, cv2.COLOR_BGR2RGB)
        espacio_fotograma.image(fotograma_rgb, channels="RGB")

        # Actualizar estado y progreso
        tiempo_transcurrido = time.time() - tiempo_inicio
        eta = (tiempo_transcurrido / contador_fotogramas) * (total_fotogramas - contador_fotogramas) if contador_fotogramas > 0 else 0
        texto_estado.text(f"Procesando Fotograma: {contador_fotogramas}/{total_fotogramas} | Saltos: {st.session_state.contador_saltos} | ETA: {eta:.1f}s")
        barra_progreso.progress(contador_fotogramas / total_fotogramas)

    # --- Limpieza Final ---
    cap.release()
    cv2.destroyAllWindows()
    texto_estado.text(f"¡Procesamiento Completo! Saltos Totales Detectados: {st.session_state.contador_saltos}")
    barra_progreso.progress(1.0)
    # Mostrar la tabla completa al final
    df_final = pd.DataFrame(st.session_state.datos_historicos)
    espacio_tabla.dataframe(df_final)


# --- Diseño de la Aplicación Streamlit ---
st.set_page_config(layout="wide", page_title="Análisis de Saltos con Pose")

st.title("🏃💨 Análisis Avanzado de Saltos con Estimación de Pose")
st.caption("Sube un video para detectar saltos, medir ángulos de rodilla y altura relativa.")

# Cargar Modelo
with st.spinner("Cargando modelo de estimación de pose..."):
    modelo = cargar_modelo(RUTA_MODELO)

if modelo:
    st.success("Modelo cargado correctamente.")
    archivo_subido = st.file_uploader("Elige un archivo de video...", type=["mp4", "avi", "mov", "mkv"])

    if archivo_subido is not None:
        detalles_archivo = {"NombreArchivo": archivo_subido.name, "TipoArchivo": archivo_subido.type, "TamañoArchivo": archivo_subido.size}
        st.write("Detalles del Video Subido:")
        st.json(detalles_archivo)

        with tempfile.NamedTemporaryFile(delete=False, suffix='.' + archivo_subido.name.split('.')[-1]) as tfile:
            tfile.write(archivo_subido.read())
            ruta_video = tfile.name

        st.info("Procesando video... Esto puede tardar.")

        # Crear columnas para layout
        col1, col2 = st.columns([2, 1]) # Video ocupa 2/3, tabla 1/3

        with col1:
            espacio_fotograma = st.empty() # Placeholder para el video

        with col2:
            st.subheader("Datos del Movimiento")
            espacio_tabla = st.empty() # Placeholder para la tabla

        # Placeholders debajo de las columnas
        texto_estado = st.empty()
        barra_progreso = st.progress(0)

        procesar_video(ruta_video, modelo, texto_estado, espacio_fotograma, espacio_tabla, barra_progreso)

else:
    st.error("No se pudo iniciar la aplicación debido a un error del modelo.")

st.markdown("---")
st.markdown("**Notas:**")
st.markdown(f"""
*   **Modelo:** YOLOv8n-Pose (`{RUTA_MODELO}`).
*   **Detección de Salto:** Basada en velocidad vertical del tobillo más bajo. Umbrales (`{UMBRAL_VELOCIDAD_SALTO_ARRIBA}`, `{UMBRAL_VELOCIDAD_SALTO_ABAJO}`) pueden necesitar ajuste.
*   **Ángulos:** Calculados entre Cadera-Rodilla-Tobillo. Requiere visibilidad y confianza >= `{UMBRAL_CONFIANZA_KEYPOINT}` para los 3 puntos. Se muestran en grados (°).
*   **Altura Tobillo:** Distancia en píxeles desde el tobillo más bajo detectado hasta el borde inferior del fotograma original. Un valor mayor significa que el tobillo está más arriba en la imagen (potencialmente en el aire). `NaN` si no se detecta ningún tobillo confiable.
*   **Tabla:** Muestra los últimos 15 fotogramas durante el procesamiento. Al finalizar, muestra todos los datos. `NaN` indica datos no disponibles para ese fotograma.
*   **Rendimiento:** El cálculo y la actualización constante de la tabla pueden ralentizar el procesamiento, especialmente en videos largos.
*   **Limitaciones:** Solo analiza la primera persona detectada. Oclusiones o detecciones pobres de keypoints afectarán los cálculos.
""")