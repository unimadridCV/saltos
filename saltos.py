# -*- coding: utf-8 -*- # Especificar codificación
import cv2
import mediapipe as mp
import numpy as np
import math
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import os
import time
import tempfile

# --- Configuración de la Página ---
TITULO_APP = "Detector de Saltos: Altura Estimada y Relativa (MediaPipe)"
st.set_page_config(page_title=TITULO_APP, layout="wide")


# --- Constantes y Configuración ---
NOMBRE_PUNTO_CLAVE_A_INDICE = {
    "Tobillo Izquierdo": 31, "Tobillo Derecho": 32
}
UMBRAL_VISIBILIDAD = 0.6
DIRECTORIO_DATOS = "datos_saltos_completo"
PREFIJO_REGISTRO = "registro_salto_completo_"
OUTPUT_VIDEO_DIR = "videos_procesados"
DEFAULT_WEBCAM_FPS = 20.0
VIDEO_CODEC = 'mp4v'; VIDEO_EXTENSION = '.mp4'
ALPHA = 0.5 # EMA suavizado Y tobillo
FLOOR_ALPHA = 0.05 # EMA nivel suelo
PLOT_MAX_PUNTOS = 150
TABLA_MAX_FILAS = 20
COLUMNAS_CSV = ['Marca_Tiempo', 'Estado_Salto', 'Altura_Rel_Px', 'Altura_Max_Estimada_m']

# Constantes de Salto
LEFT_ANKLE_IDX = NOMBRE_PUNTO_CLAVE_A_INDICE["Tobillo Izquierdo"]
RIGHT_ANKLE_IDX = NOMBRE_PUNTO_CLAVE_A_INDICE["Tobillo Derecho"]
VELOCITY_THRESHOLD_UP = -4.5
VELOCITY_THRESHOLD_DOWN = 1.5
MIN_FRAMES_IN_AIR = 8
GRAVITY = 9.81 # m/s^2

# Constantes para Texto en Video (Fuentes más grandes)
FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE_LARGE = 1.8 # Tamaño MUY grande para estado
FONT_SCALE_MEDIUM = 1.2 # Tamaño grande para métricas
FONT_THICKNESS = 3 # Grosor aumentado

os.makedirs(DIRECTORIO_DATOS, exist_ok=True)
os.makedirs(OUTPUT_VIDEO_DIR, exist_ok=True)

# --- Inicialización de MediaPipe y Funciones Auxiliares ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
@st.cache_resource
def load_mediapipe_model():
    print("Cargando modelo MediaPipe Pose...")
    return mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=0)
pose = load_mediapipe_model()

def get_landmark_coords_by_index(landmarks, index, img_shape):
    if index is None or landmarks is None or not hasattr(landmarks, 'landmark') or len(landmarks.landmark) <= index: return None, 0.0
    lm = landmarks.landmark[index]; coords = [lm.x * img_shape[1], lm.y * img_shape[0]]; visibility = lm.visibility if hasattr(lm, 'visibility') else 0.0
    if not (hasattr(lm, 'x') and hasattr(lm, 'y')): return None, 0.0
    return coords, visibility

# --- Aplicación Streamlit ---
st.title(TITULO_APP)
st.info(f"""
**Instrucciones:**
1.  Seleccione Fuente. Si es archivo, súbalo.
2.  Haga clic en "Iniciar Detección" o "Procesar Video".
3.  Observe video, gráfico altura relativa (px) y tabla. Salto confirmado tras {MIN_FRAMES_IN_AIR} frames en aire.
4.  Métrica muestra pico estimado (cm) del último salto confirmado.
5.  Haga clic en "Detener". Datos CSV y video procesado se guardarán.
""")
st.warning("**Nota:** Altura Estimada usa `h = g * t² / 8`. Altura Relativa (px) es instantánea.")

# --- Inicialización del Estado de la Sesión ---
# ... (Igual) ...
if 'is_running' not in st.session_state: st.session_state.is_running = False
if 'log_data_df' not in st.session_state: st.session_state.log_data_df = pd.DataFrame(columns=COLUMNAS_CSV)
if 'latest_csv_path' not in st.session_state: st.session_state.latest_csv_path = None
if 'cap' not in st.session_state: st.session_state.cap = None
if 'video_source' not in st.session_state: st.session_state.video_source = 'Webcam'
if 'uploaded_file_info' not in st.session_state: st.session_state.uploaded_file_info = None
if 'temp_video_path' not in st.session_state: st.session_state.temp_video_path = None
if 'live_plot_df' not in st.session_state: st.session_state.live_plot_df = pd.DataFrame(columns=['Altura Relativa (px)'], index=pd.to_datetime([]))
if 'jump_status' not in st.session_state: st.session_state.jump_status = "En Tierra"
if 'last_smoothed_ankle_y' not in st.session_state: st.session_state.last_smoothed_ankle_y = None
if 'floor_y_level' not in st.session_state: st.session_state.floor_y_level = None
if 'jump_start_time' not in st.session_state: st.session_state.jump_start_time = None
if 'last_physics_peak_height_m' not in st.session_state: st.session_state.last_physics_peak_height_m = None
if 'frames_in_air_count' not in st.session_state: st.session_state.frames_in_air_count = 0
if 'video_writer' not in st.session_state: st.session_state.video_writer = None
if 'output_video_path' not in st.session_state: st.session_state.output_video_path = None
if 'last_input_mode' not in st.session_state: st.session_state.last_input_mode = None

# --- Controles de la Barra Lateral ---
with st.sidebar:
    st.header("1. Fuente de Entrada")
    current_video_source = st.radio("Seleccione Fuente:", ('Webcam', 'Archivo'), key='source_choice', horizontal=True, disabled=st.session_state.is_running, index=0 if st.session_state.video_source == 'Webcam' else 1)
    uploaded_file = None
    if current_video_source == 'Archivo': uploaded_file = st.file_uploader("Subir:", type=["mp4", "avi", "mov", "mkv"], disabled=st.session_state.is_running)
    if uploaded_file: st.session_state.uploaded_file_info = {"name": uploaded_file.name, "type": uploaded_file.type}
    elif st.session_state.temp_video_path and os.path.exists(st.session_state.temp_video_path): fname = os.path.basename(st.session_state.temp_video_path); st.info(f"Usando: {fname}")
    st.header("2. Control"); col_btn1, col_btn2 = st.columns(2); start_button_text = "Iniciar" if current_video_source == 'Webcam' else "Procesar"; start_button_disabled = st.session_state.is_running or (current_video_source == 'Archivo' and uploaded_file is None and st.session_state.temp_video_path is None)
    with col_btn1: start_button = st.button(start_button_text, key="start", disabled=start_button_disabled, use_container_width=True)
    with col_btn2: stop_button = st.button("Detener", key="stop", disabled=not st.session_state.is_running, use_container_width=True)

# --- Reinicio de Datos si Cambia el Modo de Entrada ---
if current_video_source != st.session_state.last_input_mode and st.session_state.last_input_mode is not None:
    st.session_state.log_data_df = pd.DataFrame(columns=COLUMNAS_CSV); st.session_state.live_plot_df = pd.DataFrame(columns=['Altura Relativa (px)'], index=pd.to_datetime([])); st.session_state.live_plot_df.index.name = 'Marca_Tiempo'
    st.session_state.latest_csv_path = None; st.session_state.jump_status = "En Tierra"; st.session_state.last_smoothed_ankle_y = None
    st.session_state.floor_y_level = None; st.session_state.jump_start_time = None; st.session_state.last_physics_peak_height_m = None
    st.session_state.frames_in_air_count = 0
    if st.session_state.video_writer is not None: st.session_state.video_writer.release()
    st.session_state.video_writer = None; st.session_state.output_video_path = None
    st.warning("Fuente cambiada. Datos reiniciados.");
st.session_state.video_source = current_video_source; st.session_state.last_input_mode = current_video_source

# --- Lógica de los Botones ---
def cleanup_temp_file():
     if st.session_state.temp_video_path and os.path.exists(st.session_state.temp_video_path):
        try: os.remove(st.session_state.temp_video_path)
        except Exception as e: print(f"Error limpiando temp: {e}")
     st.session_state.temp_video_path = None
def reset_jump_state():
    st.session_state.last_smoothed_ankle_y = None; st.session_state.floor_y_level = None; st.session_state.jump_status = "En Tierra"; st.session_state.jump_start_time = None; st.session_state.last_physics_peak_height_m = None; st.session_state.frames_in_air_count = 0
def initialize_session_data():
    st.session_state.log_data_df = pd.DataFrame(columns=COLUMNAS_CSV); st.session_state.live_plot_df = pd.DataFrame(columns=['Altura Relativa (px)'], index=pd.to_datetime([])); st.session_state.live_plot_df.index.name = 'Marca_Tiempo'
    if st.session_state.video_writer is not None: st.session_state.video_writer.release()
    st.session_state.video_writer = None; st.session_state.output_video_path = None; st.session_state.latest_csv_path = None
    reset_jump_state()
def release_video_writer():
    if st.session_state.get('video_writer') is not None: print(f"Liberando VideoWriter..."); st.session_state.video_writer.release(); st.session_state.video_writer = None

if start_button:
    st.session_state.is_running = True; initialize_session_data()
    input_source = 0; source_fps = DEFAULT_WEBCAM_FPS; frame_width, frame_height = None, None
    if st.session_state.video_source == 'Archivo':
        current_file = uploaded_file
        if current_file is not None:
             try: file_info_to_keep = {"name": current_file.name}; cleanup_temp_file();
             except Exception as e: st.error(f"Error preparando archivo: {e}"); st.session_state.is_running = False
             if st.session_state.is_running:
                  try:
                      with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(current_file.name)[1]) as tfile: tfile.write(current_file.read()); st.session_state.temp_video_path = tfile.name
                      input_source = st.session_state.temp_video_path; st.session_state.uploaded_file_info = file_info_to_keep
                  except Exception as e: st.error(f"Error guardando temporal: {e}"); st.session_state.is_running = False
        elif st.session_state.temp_video_path and os.path.exists(st.session_state.temp_video_path): input_source = st.session_state.temp_video_path; st.info(f"Reutilizando: {os.path.basename(input_source)}")
        else: st.error("Archivo no válido."); st.session_state.is_running = False
    if st.session_state.is_running:
        try:
            cap = cv2.VideoCapture(input_source)
            if not cap.isOpened(): st.error(f"Error abriendo fuente: {input_source}"); st.session_state.is_running = False
            else:
                st.session_state.cap = cap; ret, first_frame = cap.read()
                if ret:
                    frame_height, frame_width, _ = first_frame.shape
                    if st.session_state.video_source == 'Archivo': fps_from_file = cap.get(cv2.CAP_PROP_FPS); source_fps = fps_from_file if fps_from_file and fps_from_file > 0 else DEFAULT_WEBCAM_FPS;
                    if source_fps == DEFAULT_WEBCAM_FPS and st.session_state.video_source == 'Archivo': st.warning(f"FPS no obtenido, usando {source_fps}.")
                    else: source_fps = DEFAULT_WEBCAM_FPS
                    print(f"Video iniciado: {frame_width}x{frame_height} @ {source_fps:.2f} FPS"); timestamp_str_file = datetime.now().strftime("%Y%m%d_%H%M%S"); output_filename = f"salto_{timestamp_str_file}{VIDEO_EXTENSION}"; output_path = os.path.join(OUTPUT_VIDEO_DIR, output_filename); fourcc = cv2.VideoWriter_fourcc(*VIDEO_CODEC)
                    try: writer = cv2.VideoWriter(output_path, fourcc, source_fps, (frame_width, frame_height))
                    except Exception as e: st.error(f"Error creando VideoWriter: {e}"); st.session_state.is_running = False
                    if 'writer' in locals() and writer.isOpened(): st.session_state.video_writer = writer; st.session_state.output_video_path = output_path; st.sidebar.success(f"Grabando: {output_filename}")
                    elif st.session_state.is_running: st.error(f"Error abriendo VideoWriter: {output_path}"); st.session_state.is_running = False
                else: st.error("Error leyendo primer frame."); st.session_state.is_running = False; cap.release(); st.session_state.cap = None
        except Exception as e: st.error(f"Excepción VideoCapture: {e}"); st.session_state.is_running = False
    if not st.session_state.is_running:
        if st.session_state.cap is not None: st.session_state.cap.release(); st.session_state.cap = None
        release_video_writer(); cleanup_temp_file()
    st.rerun()

if stop_button:
    st.session_state.is_running = False
    if st.session_state.cap is not None: st.session_state.cap.release(); st.session_state.cap = None
    release_video_writer()
    if not st.session_state.log_data_df.empty:
        df_to_save = st.session_state.log_data_df.copy()
        try: df_to_save['Marca_Tiempo'] = pd.to_datetime(df_to_save['Marca_Tiempo']).dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3];
        except Exception as format_e: st.warning(f"Error formateando tiempo CSV: {format_e}.")
        if 'Altura_Rel_Px' in df_to_save.columns: df_to_save['Altura_Rel_Px'] = pd.to_numeric(df_to_save['Altura_Rel_Px'], errors='coerce').round(1)
        if 'Altura_Max_Estimada_m' in df_to_save.columns: df_to_save['Altura_Max_Estimada_m'] = pd.to_numeric(df_to_save['Altura_Max_Estimada_m'], errors='coerce').round(3)
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S"); filename = f"{PREFIJO_REGISTRO}{timestamp_str}.csv"; filepath = os.path.join(DIRECTORIO_DATOS, filename)
        try: df_to_save.to_csv(filepath, index=False, encoding='utf-8-sig'); st.session_state.latest_csv_path = filepath; st.success(f"Datos guardados: {filepath}")
        except Exception as e: st.error(f"Error guardando CSV: {e}"); st.session_state.latest_csv_path = None
    elif not st.session_state.is_running: st.warning("No se registraron datos.")
    # No resetear DFs aquí
    st.rerun()

# --- Área Principal ---
col_feed, col_live_analysis = st.columns([2, 3])
with col_feed: st.subheader("Video / Feed"); frame_placeholder = st.empty(); progress_placeholder_container = st.container()
with col_live_analysis:
    st.subheader("Análisis de Salto en Vivo"); last_jump_height_placeholder = st.empty(); jump_chart_placeholder = st.empty()
    st.markdown("---"); st.subheader("Últimos Datos Registrados"); live_data_placeholder = st.empty()

# --- Procesamiento Principal ---
total_frames = 0; left_ankle_idx_eff = LEFT_ANKLE_IDX; right_ankle_idx_eff = RIGHT_ANKLE_IDX

# ***** CORRECCIÓN INDENTACIÓN *****
if st.session_state.is_running and st.session_state.cap is not None:
    # --- Lógica Barra Progreso (Movida aquí adentro) ---
    progress_bar = None # Inicializar fuera del if interno
    if st.session_state.video_source == 'Archivo':
        total_frames = int(st.session_state.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames > 0:
            with progress_placeholder_container: # 'with' indentado bajo 'if'
                progress_bar = st.progress(0) # Asignar el objeto progress bar
        else:
            with progress_placeholder_container: # 'with' indentado bajo 'else'
                st.empty()
    else: # Es Webcam, no hay barra de progreso
        with progress_placeholder_container: # 'with' indentado bajo 'else'
            st.empty() # Asegurarse de limpiar el área
    # --- Fin Lógica Barra ---

    frame_count = 0
    calculated_physics_height_this_frame = np.nan

    while st.session_state.is_running: # <<<<< BUCLE PRINCIPAL >>>>>
        ret, frame = st.session_state.cap.read()
        if not ret:
            st.session_state.is_running = False;
            if st.session_state.video_source == 'Archivo':
                 with progress_placeholder_container: st.success("Procesamiento completado!")
            else: st.warning("Webcam detenida/perdida.")
            break

        frame_count += 1; current_time = datetime.now(); calculated_physics_height_this_frame = np.nan

        if st.session_state.video_source == 'Webcam': frame = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB); image_rgb.flags.writeable = False
        results = pose.process(image_rgb); image_bgr = frame.copy(); image_bgr.flags.writeable = True
        previous_official_status = st.session_state.jump_status; current_official_jump_status = previous_official_status
        potential_next_status = "En Tierra"; avg_ankle_y = None; relative_height_pixels = np.nan; current_smoothed_y = None

        if results.pose_landmarks:
            landmarks = results.pose_landmarks; h, w, _ = image_bgr.shape
            left_coords, la_vis = get_landmark_coords_by_index(landmarks, left_ankle_idx_eff, (h, w))
            right_coords, ra_vis = get_landmark_coords_by_index(landmarks, right_ankle_idx_eff, (h, w))
            visible_ankles_y = []
            if left_coords and la_vis > UMBRAL_VISIBILIDAD: visible_ankles_y.append(left_coords[1])
            if right_coords and ra_vis > UMBRAL_VISIBILIDAD: visible_ankles_y.append(right_coords[1])

            if len(visible_ankles_y) > 0:
                avg_ankle_y = np.mean(visible_ankles_y)
                if st.session_state.last_smoothed_ankle_y is not None: current_smoothed_y = ALPHA * avg_ankle_y + (1 - ALPHA) * st.session_state.last_smoothed_ankle_y
                else: current_smoothed_y = avg_ankle_y
                vertical_velocity = 0.0
                if st.session_state.last_smoothed_ankle_y is not None: vertical_velocity = st.session_state.last_smoothed_ankle_y - current_smoothed_y
                if vertical_velocity < VELOCITY_THRESHOLD_UP: potential_next_status = "En Aire"
                elif vertical_velocity > VELOCITY_THRESHOLD_DOWN: potential_next_status = "En Tierra"
                else: potential_next_status = previous_official_status

                if potential_next_status == "En Aire":
                    st.session_state.frames_in_air_count += 1
                    if st.session_state.frames_in_air_count >= MIN_FRAMES_IN_AIR:
                        if previous_official_status == "En Tierra": current_official_jump_status = "En Aire"; st.session_state.jump_start_time = current_time; st.session_state.last_physics_peak_height_m = None
                    # else: current_official_jump_status permanece como estaba
                else: # Potencial Tierra
                    if previous_official_status == "En Aire":
                        current_official_jump_status = "En Tierra"
                        if st.session_state.jump_start_time is not None:
                             time_in_air = (current_time - st.session_state.jump_start_time).total_seconds()
                             if time_in_air > 0.05: calculated_physics_height_this_frame = (GRAVITY * (time_in_air ** 2)) / 8.0; st.session_state.last_physics_peak_height_m = calculated_physics_height_this_frame
                             else: calculated_physics_height_this_frame = np.nan; st.session_state.last_physics_peak_height_m = None
                             st.session_state.jump_start_time = None
                        else: calculated_physics_height_this_frame = np.nan; st.session_state.last_physics_peak_height_m = None
                    st.session_state.frames_in_air_count = 0

                if current_official_jump_status == "En Tierra": # Actualizar suelo basado en estado OFICIAL
                    if st.session_state.floor_y_level is None: st.session_state.floor_y_level = current_smoothed_y
                    else: st.session_state.floor_y_level = FLOOR_ALPHA * current_smoothed_y + (1 - FLOOR_ALPHA) * st.session_state.floor_y_level

                # Calcular Altura Relativa (px) SIEMPRE que sea posible
                if st.session_state.floor_y_level is not None and current_smoothed_y is not None:
                    raw_height_px = st.session_state.floor_y_level - current_smoothed_y
                    relative_height_pixels = max(0, raw_height_px) # Siempre >= 0
                else: relative_height_pixels = np.nan # NaN si no hay suelo o pos actual

                st.session_state.last_smoothed_ankle_y = current_smoothed_y
            else: # No tobillos
                st.session_state.last_smoothed_ankle_y = None; st.session_state.frames_in_air_count = 0
                if previous_official_status == "En Aire": # Forzar aterrizaje
                     current_official_jump_status = "En Tierra"
                     if st.session_state.jump_start_time is not None:
                          time_in_air = (current_time - st.session_state.jump_start_time).total_seconds();
                          if time_in_air > 0.05: calculated_physics_height_this_frame = (GRAVITY * (time_in_air ** 2)) / 8.0; st.session_state.last_physics_peak_height_m = calculated_physics_height_this_frame
                          else: calculated_physics_height_this_frame = np.nan; st.session_state.last_physics_peak_height_m = None
                          st.session_state.jump_start_time = None
                     else: calculated_physics_height_this_frame = np.nan; st.session_state.last_physics_peak_height_m = None
                else: current_official_jump_status = "En Tierra"
                relative_height_pixels = 0.0 if st.session_state.floor_y_level is not None else np.nan

        else: # No landmarks
             st.session_state.last_smoothed_ankle_y = None; st.session_state.frames_in_air_count = 0
             if previous_official_status == "En Aire": # Forzar aterrizaje
                  current_official_jump_status = "En Tierra"
                  if st.session_state.jump_start_time is not None:
                       time_in_air = (current_time - st.session_state.jump_start_time).total_seconds();
                       if time_in_air > 0.05: calculated_physics_height_this_frame = (GRAVITY * (time_in_air ** 2)) / 8.0; st.session_state.last_physics_peak_height_m = calculated_physics_height_this_frame
                       else: calculated_physics_height_this_frame = np.nan; st.session_state.last_physics_peak_height_m = None
                       st.session_state.jump_start_time = None
                  else: calculated_physics_height_this_frame = np.nan; st.session_state.last_physics_peak_height_m = None
             else: current_official_jump_status = "En Tierra"
             relative_height_pixels = 0.0 if st.session_state.floor_y_level is not None else np.nan

        st.session_state.jump_status = current_official_jump_status

        # --- Dibujo en Frame (Fuentes Grandes, Colores Claros) ---
        jump_text = st.session_state.jump_status
        jump_color = (255, 0, 0) if jump_text == "En Aire" else (0, 0, 255) # Azul BGR Aire, Rojo BGR Tierra
        try: h_for_text = h # Usar h si está definida
        except NameError: h_for_text = 480 # Fallback
        text_y_base = int(h_for_text * 0.1)
        cv2.putText(image_bgr, f"{jump_text}", (15, text_y_base), FONT_FACE, FONT_SCALE_LARGE, jump_color, FONT_THICKNESS)
        y_offset = int(h_for_text * 0.08)
        if relative_height_pixels is not None and not np.isnan(relative_height_pixels):
             cv2.putText(image_bgr, f"{relative_height_pixels:.0f} px", (15, text_y_base + y_offset), FONT_FACE, FONT_SCALE_MEDIUM, (200, 255, 200), FONT_THICKNESS)
        if st.session_state.last_physics_peak_height_m is not None and not np.isnan(st.session_state.last_physics_peak_height_m):
             pico_cm_display = round(st.session_state.last_physics_peak_height_m * 100)
             cv2.putText(image_bgr, f"{pico_cm_display} cm", (15, text_y_base + 2 * y_offset), FONT_FACE, FONT_SCALE_MEDIUM, (255, 200, 200), FONT_THICKNESS)
        if results.pose_landmarks:
            landmarks = results.pose_landmarks;
            l_coords, la_vis = get_landmark_coords_by_index(landmarks, LEFT_ANKLE_IDX, (h,w)); r_coords, ra_vis = get_landmark_coords_by_index(landmarks, RIGHT_ANKLE_IDX, (h,w))
            if l_coords and la_vis > UMBRAL_VISIBILIDAD: cv2.circle(image_bgr, tuple(map(int,l_coords)), 5, (255, 255, 0), -1)
            if r_coords and ra_vis > UMBRAL_VISIBILIDAD: cv2.circle(image_bgr, tuple(map(int,r_coords)), 5, (255, 255, 0), -1)

        # --- Guardar Frame ---
        if st.session_state.video_writer is not None:
             try: st.session_state.video_writer.write(image_bgr)
             except Exception as write_e: print(f"Error escribiendo frame: {write_e}")

        # --- Actualizar Datos Logs y Gráficos ---
        log_rel_px = relative_height_pixels; log_pico_m = calculated_physics_height_this_frame
        new_log_row = pd.DataFrame([{'Marca_Tiempo': current_time, 'Estado_Salto': st.session_state.jump_status, 'Altura_Rel_Px': log_rel_px, 'Altura_Max_Estimada_m': log_pico_m}])
        st.session_state.log_data_df = pd.concat([st.session_state.log_data_df, new_log_row], ignore_index=True)

        if st.session_state.live_plot_df is not None:
            height_px_to_plot = log_rel_px
            new_jump_row = pd.DataFrame({'Altura Relativa (px)': [height_px_to_plot]}, index=[pd.to_datetime(current_time)]); new_jump_row.index.name = 'Marca_Tiempo'
            if isinstance(st.session_state.live_plot_df.index, pd.DatetimeIndex): st.session_state.live_plot_df = pd.concat([st.session_state.live_plot_df, new_jump_row])
            else: st.session_state.live_plot_df = new_jump_row
            if len(st.session_state.live_plot_df) > PLOT_MAX_PUNTOS: st.session_state.live_plot_df = st.session_state.live_plot_df.iloc[-PLOT_MAX_PUNTOS:]

        # --- Actualizar UI ---
        frame_placeholder.image(image_bgr, channels="BGR", use_container_width=True)
        if progress_bar is not None and total_frames > 0: # Actualizar barra si existe
             progress = min(float(frame_count) / total_frames, 1.0);
             progress_bar.progress(progress) # No necesita 'with' para actualizar

        with col_live_analysis:
            if st.session_state.last_physics_peak_height_m is not None: pico_cm_metric = round(st.session_state.last_physics_peak_height_m); last_jump_height_placeholder.metric("Último Pico Estimado", f"{pico_cm_metric} cm")
            else: last_jump_height_placeholder.metric("Último Pico Estimado", "N/A")

            if st.session_state.live_plot_df is not None and not st.session_state.live_plot_df.empty:
                live_plot_cols = [c for c in ['Altura Relativa (px)'] if c in st.session_state.live_plot_df.columns]
                if live_plot_cols and isinstance(st.session_state.live_plot_df.index, pd.DatetimeIndex): jump_chart_placeholder.line_chart(st.session_state.live_plot_df[live_plot_cols], color=["#007bff"], height=200)

            if not st.session_state.log_data_df.empty:
                 display_df = st.session_state.log_data_df.copy()
                 display_df['Marca_Tiempo'] = pd.to_datetime(display_df['Marca_Tiempo']).dt.strftime('%H:%M:%S.%f')[:-3]
                 display_df['AltRel (px)'] = pd.to_numeric(display_df['Altura_Rel_Px'], errors='coerce').round(0).astype('Float64').astype(str).replace('<NA>', '-').replace('nan', '-')
                 display_df['Pico (cm)'] = (pd.to_numeric(display_df['Altura_Max_Estimada_m'], errors='coerce')).round(0).astype('Float64').astype(str).replace('<NA>', '-').replace('nan', '-')
                 display_df.rename(columns={'Estado_Salto': 'Estado'}, inplace=True); cols_show = ['Marca_Tiempo', 'Estado', 'AltRel (px)', 'Pico (cm)']
                 live_data_placeholder.dataframe(display_df[cols_show].tail(TABLA_MAX_FILAS), use_container_width=True, height=300)

        # time.sleep(0.01)

    # --- Limpieza final ---
    release_video_writer()
    if st.session_state.cap is not None: st.session_state.cap.release(); st.session_state.cap = None
    reset_jump_state()
    if not stop_button: time.sleep(0.5); st.rerun()

# --- Mensajes Cuando NO está Corriendo ---
# --- Mensajes Cuando NO está Corriendo ---
elif not st.session_state.is_running:
    with col_feed:
        if st.session_state.video_source == 'Webcam':
            msg = "Webcam detenida. Presione 'Iniciar Detección'."
        else: # Source is 'Archivo'
            file_info = st.session_state.uploaded_file_info
            temp_path_exists = st.session_state.temp_video_path and os.path.exists(st.session_state.temp_video_path)

            # --- Logic for 'Archivo' source message (CORRECTLY INDENTED) ---
            if file_info and temp_path_exists:
                # File uploaded in this session and temp file exists
                msg = f"Listo para procesar '{file_info['name']}'. Presione 'Procesar Video'."
            elif file_info and not temp_path_exists:
                # Info exists from previous upload, but temp file gone
                 msg = f"Archivo '{file_info['name']}' seleccionado previamente. Necesita volver a subirlo."
            elif not file_info and temp_path_exists:
                 # Temp file exists from maybe a previous run, but no upload info this session
                 msg = f"Listo para re-procesar archivo temporal. Presione 'Procesar Video'."
            else: # No file info, no temp path
                msg = "Suba un archivo de video y presione 'Procesar Video'."
            # --- End logic for 'Archivo' ---

        # This is outside the inner if/else, applies to both sources
        frame_placeholder.info(msg)
        # Ensure progress bar area is cleared when not running
        with progress_placeholder_container:
             progress_placeholder_container.empty()

    # This is also outside the inner if/else
    with col_live_analysis:
        last_jump_height_placeholder.metric("Último Pico Estimado", "N/A")
        jump_chart_placeholder.info("Gráfico altura relativa (px) aparecerá aquí.")
        live_data_placeholder.info("Registro datos en vivo aparecerá aquí.")

# --- Resumen Final ---
st.divider(); st.subheader("Resumen de la Última Sesión Procesada")
latest_csv_path = st.session_state.get('latest_csv_path'); latest_video_path = st.session_state.get('output_video_path')
final_df_to_display = None; final_df_exists = False

# Mostrar Tabla Final y Descarga CSV
if latest_csv_path and os.path.exists(latest_csv_path):
    try:
        final_df_to_display = pd.read_csv(latest_csv_path)
        if 'Altura_Max_Estimada_m' in final_df_to_display.columns: final_df_to_display['Pico (cm)'] = (pd.to_numeric(final_df_to_display['Altura_Max_Estimada_m'], errors='coerce')).round(0).astype('Float64').astype(str).replace('<NA>', '-').replace('nan', '-')
        if 'Altura_Rel_Px' in final_df_to_display.columns: final_df_to_display['AltRel (px)'] = pd.to_numeric(final_df_to_display['Altura_Rel_Px'], errors='coerce').round(0).astype('Float64').astype(str).replace('<NA>', '-').replace('nan', '-')
        display_cols_final = ['Marca_Tiempo', 'Estado_Salto', 'AltRel (px)', 'Pico (cm)']; rename_map = {'Estado_Salto':'Estado'}
        final_df_to_display_renamed = final_df_to_display[[col for col in display_cols_final if col in final_df_to_display.columns]].rename(columns=rename_map)
        st.dataframe(final_df_to_display_renamed, use_container_width=True)
        with open(latest_csv_path, "rb") as fp: st.download_button(label="Descargar Datos CSV", data=fp, file_name=os.path.basename(latest_csv_path), mime="text/csv", key="download_final_csv")
        final_df_exists = True
    except Exception as e: st.error(f"Error al leer/mostrar CSV ({latest_csv_path}): {e}"); final_df_to_display = None
elif not st.session_state.log_data_df.empty and not st.session_state.is_running:
     st.info("Mostrando datos desde memoria."); final_df_to_display = st.session_state.log_data_df.copy()
     if 'Altura_Max_Estimada_m' in final_df_to_display.columns: final_df_to_display['Pico (cm)'] = (pd.to_numeric(final_df_to_display['Altura_Max_Estimada_m'], errors='coerce')).round(0).astype('Float64').astype(str).replace('<NA>', '-').replace('nan', '-')
     if 'Altura_Rel_Px' in final_df_to_display.columns: final_df_to_display['AltRel (px)'] = pd.to_numeric(final_df_to_display['Altura_Rel_Px'], errors='coerce').round(0).astype('Float64').astype(str).replace('<NA>', '-').replace('nan', '-')
     display_cols_final = ['Marca_Tiempo', 'Estado_Salto', 'AltRel (px)', 'Pico (cm)']; rename_map = {'Estado_Salto':'Estado'}
     final_df_to_display_renamed = final_df_to_display[[col for col in display_cols_final if col in final_df_to_display.columns]].rename(columns=rename_map)
     st.dataframe(final_df_to_display_renamed, use_container_width=True); final_df_exists = True
elif not st.session_state.is_running: st.info("No hay datos registrados.")

# Descarga Video Procesado
if latest_video_path and os.path.exists(latest_video_path):
    st.success(f"Video procesado guardado: {latest_video_path}")
    try:
        with open(latest_video_path, "rb") as file: st.download_button(label="Descargar Video Procesado", data=file, file_name=os.path.basename(latest_video_path), mime="video/mp4")
    except Exception as e: st.error(f"No se pudo abrir video para descarga: {e}")
elif not st.session_state.is_running: st.info("No se encontró video procesado.")

# Gráfico Final Altura Relativa (px)
if final_df_exists and final_df_to_display is not None and not final_df_to_display.empty:
    st.subheader("Gráfico Altura Relativa (px) (Resumen Completo)")
    try:
        df_chart_final_rel = final_df_to_display.copy()
        if 'Altura_Rel_Px' in df_chart_final_rel.columns and 'Marca_Tiempo' in df_chart_final_rel.columns:
            df_chart_final_rel['Marca_Tiempo'] = pd.to_datetime(df_chart_final_rel['Marca_Tiempo'], errors='coerce')
            df_chart_final_rel['Altura_Rel_Px'] = pd.to_numeric(df_chart_final_rel['Altura_Rel_Px'], errors='coerce') # Numérico para plotear
            df_chart_final_rel.dropna(subset=['Marca_Tiempo'], inplace=True)
            if not df_chart_final_rel.empty:
                 df_chart_final_rel.set_index('Marca_Tiempo', inplace=True)
                 st.line_chart(df_chart_final_rel[['Altura_Rel_Px']], use_container_width=True, color=["#007bff"])
                 st.caption("Altura relativa en píxeles desde el nivel del suelo detectado.")
            else: st.warning("No se pudieron graficar datos finales AltRel(px).")
        else: st.warning("Columnas requeridas ('Marca_Tiempo', 'Altura_Rel_Px') no encontradas.")
    except Exception as e: st.error(f"Error gráfico final Altura Relativa: {e}"); st.exception(e)