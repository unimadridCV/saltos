# -*- coding: utf-8 -*- # Especificar codificación para caracteres españoles
import cv2
import mediapipe as mp
import numpy as np
import math
import streamlit as st
import pandas as pd
from datetime import datetime
import os
import time # Opcional: Puede usarse para pequeñas pausas si es necesario
import tempfile # Para manejar archivos de video temporales

# --- Configuración de la Página (DEBE SER EL PRIMER COMANDO STREAMLIT) ---
TITULO_APP = "Rastreador de Ángulos Personalizado (MediaPipe) - Gráfico y Datos en Vivo"
st.set_page_config(page_title=TITULO_APP, layout="wide")

# --- Constantes y Configuración ---
NOMBRE_PUNTO_CLAVE_A_INDICE = {
    "Nariz": 0, "Ojo Izquierdo Interno": 1, "Ojo Izquierdo": 2, "Ojo Izquierdo Externo": 3,
    "Ojo Derecho Interno": 4, "Ojo Derecho": 5, "Ojo Derecho Externo": 6,
    "Oreja Izquierda": 7, "Oreja Derecha": 8, "Boca Izquierda": 9, "Boca Derecha": 10,
    "Hombro Izquierdo": 11, "Hombro Derecho": 12, "Codo Izquierdo": 13, "Codo Derecho": 14,
    "Muñeca Izquierda": 15, "Muñeca Derecha": 16, "Meñique Izquierdo": 17, "Meñique Derecho": 18,
    "Índice Izquierdo": 19, "Índice Derecho": 20, "Pulgar Izquierdo": 21, "Pulgar Derecho": 22,
    "Cadera Izquierda": 23, "Cadera Derecha": 24, "Rodilla Izquierda": 25, "Rodilla Derecha": 26,
    "Tobillo Izquierdo": 27, "Tobillo Derecho": 28, "Talón Izquierdo": 29, "Talón Derecho": 30,
    "Índice Pie Izquierdo": 31, "Índice Pie Derecho": 32
}
NOMBRES_PUNTOS_CLAVE = list(NOMBRE_PUNTO_CLAVE_A_INDICE.keys())
UMBRAL_VISIBILIDAD = 0.6
DIRECTORIO_DATOS = "datos_angulos"
PREFIJO_REGISTRO = "registro_angulo_"
ALPHA = 0.4 # Factor EMA para suavizado (opcional)
PLOT_MAX_PUNTOS = 100 # Puntos máx en gráfico en vivo
TABLA_MAX_FILAS = 15 # Filas máx en tabla en vivo
COLUMNAS_CSV = ['Marca_Tiempo', 'Angulo (Grados)', 'Punto_A', 'Punto_B_Vertice', 'Punto_C', 'Alerta_Riesgo']


# --- Inicialización de MediaPipe y Funciones Auxiliares ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
@st.cache_resource # Cachear el modelo para no recargarlo
def load_mediapipe_model():
    print("Cargando modelo MediaPipe Pose...")
    return mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=0)
pose = load_mediapipe_model()

def calculate_angle(a, b, c):
    if a is None or b is None or c is None: return None
    a = np.array(a); b = np.array(b); c = np.array(c)
    ba = a - b; bc = c - b
    dot_product = np.dot(ba, bc)
    magnitude_ba = np.linalg.norm(ba); magnitude_bc = np.linalg.norm(bc)
    if magnitude_ba == 0 or magnitude_bc == 0: return None
    cosine_angle = np.clip(dot_product / (magnitude_ba * magnitude_bc), -1.0, 1.0)
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

def get_landmark_coords_by_index(landmarks, index, img_shape):
    if index is None or landmarks is None or not hasattr(landmarks, 'landmark') or len(landmarks.landmark) <= index: return None, 0.0
    lm = landmarks.landmark[index]
    if not (hasattr(lm, 'x') and hasattr(lm, 'y')): return None, 0.0
    coords = [lm.x * img_shape[1], lm.y * img_shape[0]]
    visibility = lm.visibility if hasattr(lm, 'visibility') else 0.0
    return coords, visibility

# *** NUEVA FUNCIÓN AUXILIAR para intercambiar Izquierdo/Derecho ***
def swap_left_right(landmark_name):
    """Intercambia 'Izquierdo' y 'Derecho' en nombres de puntos clave."""
    if "Izquierdo" in landmark_name:
        return landmark_name.replace("Izquierdo", "Derecho")
    elif "Derecho" in landmark_name:
        return landmark_name.replace("Derecho", "Izquierdo")
    # Considerar otros casos si es necesario (ej. Boca Izquierda -> Boca Derecha)
    elif "Izquierda" in landmark_name:
         return landmark_name.replace("Izquierda", "Derecha")
    elif "Derecha" in landmark_name:
         return landmark_name.replace("Derecha", "Izquierda")
    else:
        return landmark_name # No cambiar si no es Izquierdo/Derecho

# --- Aplicación Streamlit ---
st.title(TITULO_APP)
st.info("""
**Instrucciones:**
1.  Seleccione Fuente, Puntos Clave y Umbrales de Riesgo.
2.  Haga clic en "Iniciar Rastreo" o "Procesar Video".
3.  Observe el video, el gráfico y la tabla de datos actualizándose en vivo.
4.  Haga clic en "Detener" al finalizar. Los datos completos se guardarán.
""")

# --- Inicialización del Estado de la Sesión ---
# ... (sin cambios en la inicialización del estado) ...
if 'is_running' not in st.session_state: st.session_state.is_running = False
if 'log_data_df' not in st.session_state: st.session_state.log_data_df = pd.DataFrame(columns=COLUMNAS_CSV)
if 'latest_csv_path' not in st.session_state: st.session_state.latest_csv_path = None
if 'cap' not in st.session_state: st.session_state.cap = None
if 'smoothed_points' not in st.session_state: st.session_state.smoothed_points = {}
if 'video_source' not in st.session_state: st.session_state.video_source = 'Webcam'
if 'uploaded_file_info' not in st.session_state: st.session_state.uploaded_file_info = None
if 'temp_video_path' not in st.session_state: st.session_state.temp_video_path = None
if 'selected_point_a' not in st.session_state: st.session_state.selected_point_a = "Hombro Izquierdo"
if 'selected_point_b' not in st.session_state: st.session_state.selected_point_b = "Codo Izquierdo"
if 'selected_point_c' not in st.session_state: st.session_state.selected_point_c = "Muñeca Izquierda"
if 'live_chart_df' not in st.session_state: st.session_state.live_chart_df = pd.DataFrame(columns=['Marca_Tiempo', 'Angulo Normal', 'Angulo Riesgo']).set_index('Marca_Tiempo')
if 'last_input_mode' not in st.session_state: st.session_state.last_input_mode = None

# --- Controles de la Barra Lateral ---
# ... (sin cambios en la barra lateral) ...
with st.sidebar:
    st.header("1. Fuente de Entrada")
    current_video_source = st.radio("Seleccione Fuente:", ('Webcam', 'Archivo de Video'), key='source_choice', horizontal=True, disabled=st.session_state.is_running, index=0 if st.session_state.video_source == 'Webcam' else 1)
    uploaded_file = None
    if current_video_source == 'Archivo de Video':
        uploaded_file = st.file_uploader("Subir archivo de video:", type=["mp4", "avi", "mov", "mkv"], disabled=st.session_state.is_running)
        if uploaded_file:
            st.session_state.uploaded_file_info = {"name": uploaded_file.name, "type": uploaded_file.type} # Guardar info básica
        elif st.session_state.temp_video_path and os.path.exists(st.session_state.temp_video_path): # Si se reusa un archivo temporal
             fname = os.path.basename(st.session_state.temp_video_path)
             st.info(f"Usando archivo: {fname}")

    st.header("2. Puntos Clave")
    default_idx_a = NOMBRES_PUNTOS_CLAVE.index(st.session_state.selected_point_a) if st.session_state.selected_point_a in NOMBRES_PUNTOS_CLAVE else 0
    default_idx_b = NOMBRES_PUNTOS_CLAVE.index(st.session_state.selected_point_b) if st.session_state.selected_point_b in NOMBRES_PUNTOS_CLAVE else 0
    default_idx_c = NOMBRES_PUNTOS_CLAVE.index(st.session_state.selected_point_c) if st.session_state.selected_point_c in NOMBRES_PUNTOS_CLAVE else 0

    kp_a_name = st.selectbox("Punto A (Proximal):", NOMBRES_PUNTOS_CLAVE, index=default_idx_a, key="kp_a_select", disabled=st.session_state.is_running)
    kp_b_name = st.selectbox("Punto B (Vértice Ángulo):", NOMBRES_PUNTOS_CLAVE, index=default_idx_b, key="kp_b_select", disabled=st.session_state.is_running)
    kp_c_name = st.selectbox("Punto C (Distal):", NOMBRES_PUNTOS_CLAVE, index=default_idx_c, key="kp_c_select", disabled=st.session_state.is_running)

    st.session_state.selected_point_a = kp_a_name
    st.session_state.selected_point_b = kp_b_name
    st.session_state.selected_point_c = kp_c_name

    st.header("3. Umbrales de Riesgo (Grados)")
    angle_min_thresh = st.slider("Ángulo Mínimo Normal:", 0, 180, 40, disabled=st.session_state.is_running)
    angle_max_thresh = st.slider("Ángulo Máximo Normal:", 0, 180, 170, disabled=st.session_state.is_running)

    st.header("4. Control")
    selected_points_set = {kp_a_name, kp_b_name, kp_c_name}
    are_points_unique = len(selected_points_set) == 3
    if not are_points_unique and not st.session_state.is_running:
        st.error("Por favor, seleccione tres puntos clave *diferentes*.")

    col_btn1, col_btn2 = st.columns(2)
    start_button_text = "Iniciar Rastreo" if current_video_source == 'Webcam' else "Procesar Video"
    start_button_disabled = st.session_state.is_running or \
                            not are_points_unique or \
                            (current_video_source == 'Archivo de Video' and uploaded_file is None and st.session_state.temp_video_path is None)

    with col_btn1:
        start_button = st.button(start_button_text, key="start", disabled=start_button_disabled, use_container_width=True)
    with col_btn2:
        stop_button = st.button("Detener", key="stop", disabled=not st.session_state.is_running, use_container_width=True)


# --- Reinicio de Datos si Cambia el Modo de Entrada ---
# ... (sin cambios en la lógica de reinicio) ...
if current_video_source != st.session_state.last_input_mode and st.session_state.last_input_mode is not None:
    st.session_state.log_data_df = pd.DataFrame(columns=COLUMNAS_CSV)
    st.session_state.live_chart_df = pd.DataFrame(columns=['Marca_Tiempo', 'Angulo Normal', 'Angulo Riesgo']).set_index('Marca_Tiempo')
    st.session_state.latest_csv_path = None
    st.warning("Fuente de entrada cambiada. Se reiniciaron los datos.")
st.session_state.video_source = current_video_source
st.session_state.last_input_mode = current_video_source


# --- Lógica de los Botones ---
# ... (sin cambios en las funciones cleanup, reset_smoothing, initialize_dfs) ...
def cleanup_temp_file():
    if st.session_state.temp_video_path and os.path.exists(st.session_state.temp_video_path):
        try: os.remove(st.session_state.temp_video_path)
        except Exception as e: print(f"Error limpiando archivo temporal: {e}")
    st.session_state.temp_video_path = None

def reset_smoothing_dict():
    st.session_state.smoothed_points = {}

def initialize_live_dfs():
    st.session_state.log_data_df = pd.DataFrame(columns=COLUMNAS_CSV)
    st.session_state.live_chart_df = pd.DataFrame(columns=['Angulo Normal', 'Angulo Riesgo'], index=pd.to_datetime([]))
    st.session_state.live_chart_df.index.name = 'Marca_Tiempo'

# ... (sin cambios en la lógica de start_button y stop_button) ...
if start_button:
    st.session_state.is_running = True
    st.session_state.latest_csv_path = None
    reset_smoothing_dict()
    initialize_live_dfs()

    input_source = 0
    if st.session_state.video_source == 'Archivo de Video':
        current_file = uploaded_file
        if current_file is not None:
             try:
                 file_info_to_keep = {"name": current_file.name}
                 cleanup_temp_file()
                 with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(current_file.name)[1]) as tfile:
                     tfile.write(current_file.read())
                     st.session_state.temp_video_path = tfile.name
                 input_source = st.session_state.temp_video_path
                 st.session_state.uploaded_file_info = file_info_to_keep
             except Exception as e:
                 st.error(f"Error al manejar archivo subido: {e}")
                 st.session_state.is_running = False
        elif st.session_state.temp_video_path and os.path.exists(st.session_state.temp_video_path):
            input_source = st.session_state.temp_video_path
            st.info(f"Reutilizando archivo: {os.path.basename(input_source)}")
        else:
            st.error("No se proporcionó ningún archivo de video válido.")
            st.session_state.is_running = False

    if st.session_state.is_running:
        try:
            st.session_state.cap = cv2.VideoCapture(input_source)
            if not st.session_state.cap.isOpened():
                st.error(f"Error: No se pudo abrir la fuente de video: {input_source}")
                st.session_state.is_running = False
                cleanup_temp_file()
        except Exception as e:
             st.error(f"Excepción al abrir VideoCapture: {e}")
             st.session_state.is_running = False
             cleanup_temp_file()
    st.rerun()

if stop_button:
    st.session_state.is_running = False
    if st.session_state.cap is not None:
        st.session_state.cap.release()
        st.session_state.cap = None
    reset_smoothing_dict()

    if not st.session_state.log_data_df.empty:
        df_to_save = st.session_state.log_data_df.copy()
        try:
            df_to_save['Marca_Tiempo'] = pd.to_datetime(df_to_save['Marca_Tiempo']).dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            df_to_save['Angulo (Grados)'] = pd.to_numeric(df_to_save['Angulo (Grados)'], errors='coerce').round(2)
        except Exception as format_e:
            st.warning(f"Error formateando datos para CSV: {format_e}. Guardando como está.")

        if not os.path.exists(DIRECTORIO_DATOS): os.makedirs(DIRECTORIO_DATOS)
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        sel_a = st.session_state.selected_point_a.replace(" ", "")[:5]
        sel_b = st.session_state.selected_point_b.replace(" ", "")[:5]
        sel_c = st.session_state.selected_point_c.replace(" ", "")[:5]
        filename = f"{PREFIJO_REGISTRO}{sel_a}_{sel_b}_{sel_c}_{timestamp_str}.csv"
        filepath = os.path.join(DIRECTORIO_DATOS, filename)
        try:
            df_to_save.to_csv(filepath, index=False, encoding='utf-8-sig')
            st.session_state.latest_csv_path = filepath
            st.success(f"Datos guardados en: {filepath}")
        except Exception as e:
            st.error(f"Error al guardar datos CSV: {e}")
            st.session_state.latest_csv_path = None
    elif not st.session_state.is_running:
        st.warning("No se registraron datos en esta sesión.")

    initialize_live_dfs()
    st.rerun()


# --- Área Principal: Video/Feed, Gráfico en Vivo, Tabla en Vivo ---
col_feed, col_live_analysis = st.columns([3, 2])

with col_feed:
    st.subheader("Video / Feed")
    frame_placeholder = st.empty()
    progress_placeholder_container = st.container()

with col_live_analysis:
    st.subheader("Análisis en Vivo")
    angle_placeholder = st.empty()
    chart_placeholder = st.empty()
    live_data_placeholder = st.empty()


# --- Procesamiento Principal (si está corriendo) ---
total_frames = 0
kp_idx_a, kp_idx_b, kp_idx_c = None, None, None

if st.session_state.is_running and st.session_state.cap is not None:
    # Obtener total_frames si es video
    if st.session_state.video_source == 'Archivo de Video':
        total_frames = int(st.session_state.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        with progress_placeholder_container:
            progress_bar = st.progress(0)

    # *** Determinar los NOMBRES EFECTIVOS y luego los ÍNDICES a usar ***
    # Usar los nombres seleccionados originalmente
    orig_kp_name_a = st.session_state.selected_point_a
    orig_kp_name_b = st.session_state.selected_point_b
    orig_kp_name_c = st.session_state.selected_point_c

    # Determinar los nombres a USAR para buscar coordenadas (intercambiados si es webcam)
    effective_kp_name_a = orig_kp_name_a
    effective_kp_name_b = orig_kp_name_b
    effective_kp_name_c = orig_kp_name_c
    if st.session_state.video_source == 'Webcam':
        effective_kp_name_a = swap_left_right(orig_kp_name_a)
        effective_kp_name_b = swap_left_right(orig_kp_name_b)
        effective_kp_name_c = swap_left_right(orig_kp_name_c)
        # print(f"Webcam: Usando {effective_kp_name_a}, {effective_kp_name_b}, {effective_kp_name_c}") # Debug

    # Obtener los ÍNDICES a partir de los nombres efectivos
    kp_idx_a = NOMBRE_PUNTO_CLAVE_A_INDICE.get(effective_kp_name_a)
    kp_idx_b = NOMBRE_PUNTO_CLAVE_A_INDICE.get(effective_kp_name_b)
    kp_idx_c = NOMBRE_PUNTO_CLAVE_A_INDICE.get(effective_kp_name_c)
    # *** Fin de la determinación de índices ***

    if kp_idx_a is None or kp_idx_b is None or kp_idx_c is None:
        st.error(f"Error interno: No se encontraron índices para los puntos efectivos: {effective_kp_name_a}, {effective_kp_name_b}, {effective_kp_name_c}")
        st.session_state.is_running = False # Detener si hay error
    else:
        frame_count = 0
        while st.session_state.is_running: # <<<<< BUCLE PRINCIPAL >>>>>
            ret, frame = st.session_state.cap.read()
            if not ret:
                # ... (Fin de video/error logic) ...
                st.session_state.is_running = False
                if st.session_state.video_source == 'Archivo de Video':
                    with progress_placeholder_container: st.success("Procesamiento completado!")
                else: st.warning("Webcam detenida o perdida.")
                break

            frame_count += 1
            # --- Procesamiento del Frame ---
            # Aplicar flip ANTES de procesar con MediaPipe si es webcam
            if st.session_state.video_source == 'Webcam': frame = cv2.flip(frame, 1)

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            results = pose.process(image_rgb)
            # Usar el frame original (ya flipeado si es webcam) para dibujar
            image_bgr = frame.copy() # Crear copia para no modificar el original de cap.read
            image_bgr.flags.writeable = True

            calculated_angle = None
            injury_alert = False
            coords_a, coords_b, coords_c = None, None, None
            all_visible_this_frame = False
            display_color = (0, 0, 255) # Rojo por defecto

            if results.pose_landmarks:
                landmarks = results.pose_landmarks
                h, w, _ = image_bgr.shape
                # Usar los índices kp_idx_a, kp_idx_b, kp_idx_c (ya ajustados para webcam)
                pt_a_coords, pt_a_vis = get_landmark_coords_by_index(landmarks, kp_idx_a, (h,w))
                pt_b_coords, pt_b_vis = get_landmark_coords_by_index(landmarks, kp_idx_b, (h,w))
                pt_c_coords, pt_c_vis = get_landmark_coords_by_index(landmarks, kp_idx_c, (h,w))

                all_visible_this_frame = (pt_a_coords is not None and pt_b_coords is not None and pt_c_coords is not None and
                                          pt_a_vis > UMBRAL_VISIBILIDAD and pt_b_vis > UMBRAL_VISIBILIDAD and pt_c_vis > UMBRAL_VISIBILIDAD)

                if all_visible_this_frame:
                    current_a, current_b, current_c = np.array(pt_a_coords), np.array(pt_b_coords), np.array(pt_c_coords)
                    # Aplicar Suavizado EMA
                    # Usar índices directamente para el diccionario de suavizado
                    st.session_state.smoothed_points[kp_idx_a] = ALPHA * current_a + (1 - ALPHA) * st.session_state.smoothed_points.get(kp_idx_a, current_a)
                    st.session_state.smoothed_points[kp_idx_b] = ALPHA * current_b + (1 - ALPHA) * st.session_state.smoothed_points.get(kp_idx_b, current_b)
                    st.session_state.smoothed_points[kp_idx_c] = ALPHA * current_c + (1 - ALPHA) * st.session_state.smoothed_points.get(kp_idx_c, current_c)
                    coords_a, coords_b, coords_c = st.session_state.smoothed_points[kp_idx_a], st.session_state.smoothed_points[kp_idx_b], st.session_state.smoothed_points[kp_idx_c]

                    calculated_angle = calculate_angle(coords_a, coords_b, coords_c)

                    if calculated_angle is not None:
                        if not (angle_min_thresh <= calculated_angle <= angle_max_thresh):
                            injury_alert = True; display_color = (0, 0, 255)
                        else:
                            injury_alert = False; display_color = (0, 255, 0)

                        # --- Actualizar Datos para Log y Gráfico ---
                        current_time = datetime.now()
                        # 1. Log Completo (Usar NOMBRES ORIGINALES seleccionados)
                        new_log_row = pd.DataFrame([{
                            'Marca_Tiempo': current_time,
                            'Angulo (Grados)': calculated_angle,
                            'Punto_A': orig_kp_name_a, # Log nombre original
                            'Punto_B_Vertice': orig_kp_name_b, # Log nombre original
                            'Punto_C': orig_kp_name_c, # Log nombre original
                            'Alerta_Riesgo': injury_alert
                        }])
                        # ... (concatenación a log_data_df sin cambios) ...
                        for col in new_log_row.columns:
                             if col in st.session_state.log_data_df.columns:
                                 if st.session_state.log_data_df[col].dtype != new_log_row[col].dtype:
                                     try: st.session_state.log_data_df[col] = st.session_state.log_data_df[col].astype(new_log_row[col].dtype)
                                     except Exception: pass
                        st.session_state.log_data_df = pd.concat([st.session_state.log_data_df, new_log_row], ignore_index=True)


                        # 2. Gráfico en Vivo
                        if st.session_state.live_chart_df is not None:
                            # ... (preparación de new_chart_row y concatenación sin cambios) ...
                             normal_angle = calculated_angle if not injury_alert else np.nan
                             riesgo_angle = calculated_angle if injury_alert else np.nan
                             new_chart_row = pd.DataFrame({'Angulo Normal': [normal_angle],'Angulo Riesgo': [riesgo_angle]}, index=[pd.to_datetime(current_time)])
                             new_chart_row.index.name = 'Marca_Tiempo'
                             if isinstance(st.session_state.live_chart_df.index, pd.DatetimeIndex):
                                  st.session_state.live_chart_df = pd.concat([st.session_state.live_chart_df, new_chart_row])
                             else: st.session_state.live_chart_df = new_chart_row
                             if len(st.session_state.live_chart_df) > PLOT_MAX_PUNTOS: st.session_state.live_chart_df = st.session_state.live_chart_df.iloc[-PLOT_MAX_PUNTOS:]

                    else: display_color = (255, 165, 0) # Naranja
                else: # No todos visibles
                    display_color = (0, 0, 255) # Rojo
                    coords_a = st.session_state.smoothed_points.get(kp_idx_a)
                    coords_b = st.session_state.smoothed_points.get(kp_idx_b)
                    coords_c = st.session_state.smoothed_points.get(kp_idx_c)
            else: # No landmarks
                 display_color = (0, 0, 255) # Rojo
                 coords_a = st.session_state.smoothed_points.get(kp_idx_a)
                 coords_b = st.session_state.smoothed_points.get(kp_idx_b)
                 coords_c = st.session_state.smoothed_points.get(kp_idx_c)

            # --- Dibujo en el Frame ---
            # El dibujo usa las coords (que vienen de los índices ya ajustados) sobre image_bgr (que ya está flipeado si es webcam)
            if coords_a is not None and coords_b is not None and coords_c is not None:
                pt_a_draw, pt_b_draw, pt_c_draw = tuple(coords_a.astype(int)), tuple(coords_b.astype(int)), tuple(coords_c.astype(int))
                circle_fill = -1; line_thickness = 2
                cv2.circle(image_bgr, pt_a_draw, 5, display_color, circle_fill)
                cv2.circle(image_bgr, pt_b_draw, 7, display_color, circle_fill)
                cv2.circle(image_bgr, pt_c_draw, 5, display_color, circle_fill)
                cv2.line(image_bgr, pt_a_draw, pt_b_draw, display_color, line_thickness)
                cv2.line(image_bgr, pt_b_draw, pt_c_draw, display_color, line_thickness)
                if calculated_angle is not None:
                    cv2.putText(image_bgr, f"{calculated_angle:.1f} deg", tuple(pt_b_draw + np.array([10,-10])), cv2.FONT_HERSHEY_SIMPLEX, 0.7, display_color, 2, cv2.LINE_AA)
                if injury_alert:
                    cv2.putText(image_bgr, "RIESGO", tuple(pt_b_draw + np.array([10, 10])), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)


            # --- Actualizar Placeholders de UI ---

            # 1. Frame de video (CORREGIDO - ya estaba bien, solo confirmar)
            frame_placeholder.image(image_bgr, channels="BGR", use_container_width=True)

            # 2. Barra de progreso
            if st.session_state.video_source == 'Archivo de Video' and total_frames > 0:
                 progress = min(float(frame_count) / total_frames, 1.0)
                 with progress_placeholder_container: progress_bar.progress(progress)

            # 3. Métrica, Gráfico, Tabla (en la otra columna)
            with col_live_analysis:
                # Métrica
                # ... (actualización de angle_placeholder sin cambios) ...
                angle_label_text = f"Ángulo ({st.session_state.selected_point_b})" # Mostrar nombre original
                if calculated_angle is not None:
                    delta_text = "¡Riesgo!" if injury_alert else "Normal"
                    delta_color = "inverse" if injury_alert else "normal"
                    angle_placeholder.metric(angle_label_text, f"{calculated_angle:.1f}°", delta=delta_text, delta_color=delta_color)
                else:
                    angle_placeholder.metric(angle_label_text, "N/A", delta="No Detectado", delta_color="off")

                # Gráfico
                # ... (actualización de chart_placeholder sin cambios) ...
                if st.session_state.live_chart_df is not None and not st.session_state.live_chart_df.empty:
                     cols_to_plot = [col for col in ['Angulo Normal', 'Angulo Riesgo'] if col in st.session_state.live_chart_df.columns]
                     if cols_to_plot and isinstance(st.session_state.live_chart_df.index, pd.DatetimeIndex):
                          chart_placeholder.line_chart(st.session_state.live_chart_df[cols_to_plot], color=["#007bff", "#dc3545"], use_container_width=True)

                # Tabla
                # ... (actualización de live_data_placeholder sin cambios) ...
                if not st.session_state.log_data_df.empty:
                     display_df = st.session_state.log_data_df.copy()
                     display_df['Marca_Tiempo'] = pd.to_datetime(display_df['Marca_Tiempo']).dt.strftime('%H:%M:%S.%f')[:-3]
                     display_df['Angulo (Grados)'] = pd.to_numeric(display_df['Angulo (Grados)'], errors='coerce').round(1)
                     display_df.rename(columns={'Alerta_Riesgo': 'Riesgo', 'Angulo (Grados)': 'Ángulo'}, inplace=True)
                     cols_to_show = ['Marca_Tiempo', 'Ángulo', 'Riesgo', 'Punto_A', 'Punto_B_Vertice', 'Punto_C']
                     display_df_final = display_df[[col for col in cols_to_show if col in display_df.columns]]
                     live_data_placeholder.dataframe(display_df_final.tail(TABLA_MAX_FILAS), use_container_width=True, height=300)

            # time.sleep(0.01) # Pausa opcional


    # --- Limpieza después del bucle ---
    # ... (sin cambios) ...
    if st.session_state.cap is not None:
        st.session_state.cap.release()
        st.session_state.cap = None
    reset_smoothing_dict()
    if not stop_button:
         time.sleep(0.5)
         st.rerun()


# --- Mensajes y Estado Cuando NO está Corriendo ---
# ... (sin cambios) ...
elif not st.session_state.is_running:
    with col_feed:
        if st.session_state.video_source == 'Webcam':
            msg = "Webcam detenida. Configure y presione 'Iniciar Rastreo'."
        else:
            file_info = st.session_state.uploaded_file_info
            temp_path_exists = st.session_state.temp_video_path and os.path.exists(st.session_state.temp_video_path)
            if file_info:
                 msg = f"Listo para procesar '{file_info['name']}'. Configure y presione 'Procesar Video'."
                 if not temp_path_exists: msg += " (Necesita volver a subir el archivo)."
            elif temp_path_exists:
                 msg = f"Listo para re-procesar '{os.path.basename(st.session_state.temp_video_path)}'. Configure y presione 'Procesar Video'."
            else:
                 msg = "Suba un archivo de video, configure y presione 'Procesar Video'."
        frame_placeholder.info(msg)
        with progress_placeholder_container: progress_placeholder_container.empty()

    with col_live_analysis:
        angle_placeholder.metric(f"Ángulo ({st.session_state.selected_point_b})", "--", delta="Detenido", delta_color="off")
        chart_placeholder.info("El gráfico en vivo aparecerá aquí al iniciar.")
        live_data_placeholder.info("El registro de datos en vivo aparecerá aquí.")


# --- Resumen Final de la Sesión (después de detener o terminar) ---
# ... (sin cambios) ...
st.divider()
st.subheader("Resumen de la Última Sesión Procesada")

latest_path = st.session_state.get('latest_csv_path')
final_df_to_display = None
final_df_exists = False

if latest_path and os.path.exists(latest_path):
    try:
        final_df_to_display = pd.read_csv(latest_path)
        st.dataframe(final_df_to_display, use_container_width=True)
        with open(latest_path, "rb") as fp:
            st.download_button(label="Descargar Último CSV", data=fp, file_name=os.path.basename(latest_path), mime="text/csv", key="download_final_csv")
        final_df_exists = True
    except Exception as e:
        st.error(f"Error al leer el archivo CSV guardado ({latest_path}): {e}")
        final_df_to_display = None

if not final_df_exists and not st.session_state.log_data_df.empty and not st.session_state.is_running:
     st.info("Mostrando datos de la última ejecución (desde memoria).")
     final_df_to_display = st.session_state.log_data_df.copy()
     st.dataframe(final_df_to_display, use_container_width=True)
     final_df_exists = True

elif not final_df_exists and not st.session_state.is_running:
    st.info("No hay datos registrados de la última sesión para mostrar.")

if final_df_exists and final_df_to_display is not None and not final_df_to_display.empty:
    try:
        st.subheader("Gráfico de Ángulo (Resumen Completo Sesión)")
        df_chart_final = final_df_to_display.copy()
        df_chart_final['Marca_Tiempo'] = pd.to_datetime(df_chart_final['Marca_Tiempo'], errors='coerce')
        df_chart_final['Angulo (Grados)'] = pd.to_numeric(df_chart_final['Angulo (Grados)'], errors='coerce')
        if 'Alerta_Riesgo' not in df_chart_final.columns: df_chart_final['Alerta_Riesgo'] = False
        else: df_chart_final['Alerta_Riesgo'] = df_chart_final['Alerta_Riesgo'].astype(bool)
        df_chart_final.dropna(subset=['Marca_Tiempo', 'Angulo (Grados)'], inplace=True)

        if not df_chart_final.empty:
             df_chart_final['Angulo Normal'] = df_chart_final.apply(lambda row: row['Angulo (Grados)'] if not row['Alerta_Riesgo'] else np.nan, axis=1)
             df_chart_final['Angulo Riesgo'] = df_chart_final.apply(lambda row: row['Angulo (Grados)'] if row['Alerta_Riesgo'] else np.nan, axis=1)
             df_chart_final.set_index('Marca_Tiempo', inplace=True)
             st.line_chart(df_chart_final[['Angulo Normal', 'Angulo Riesgo']], color=["#007bff", "#dc3545"], use_container_width=True)
        else: st.warning("No se pudieron graficar los datos finales (vacío después de limpieza).")
    except Exception as e:
        st.error(f"Error al generar el gráfico de resumen final: {e}")
        st.exception(e)