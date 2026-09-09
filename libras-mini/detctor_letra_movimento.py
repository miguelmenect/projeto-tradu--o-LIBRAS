import argparse
import time
from collections import deque
from pathlib import Path
 
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions
 
CAMINHO_DETECTOR_PADRAO = "hand_landmarker.task"
 
# mesmos indices de referencia usados no script de treino
# POORSICOES E ESCALAS
PULSO = 0
BASE_DEDO_MEDIO = 9
PONTA_INDICADOR = 8
