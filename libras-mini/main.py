import cv2
import mediapipe as mp
import numpy as np

# inicializa o mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# atrbui valor de captura da camera
cap = cv2.VideoCapture(0)

def punho_fechado(landmarks): #função recebendo de parametro lista com os pontos da mão com referencia
    """
    Verifica se os dedos estão fechados
    """

    #4 - polegar
    #8 - idicador
    #12 - dedo do meio
    #16- dedo anelr
    #20-mindinho

    dedos = [4, 8, 12, 16, 20]  # pontas dos dedos
    base = 0  # punho

    fechado = True
    for dedo in dedos:
        #se o dedo encontrado no array for menor que punho então "mão fechada" é false
        if landmarks[dedo].y < landmarks[base].y:
            fechado = False #mão fechada é false

    return fechado

while True:
    success, frame = cap.read() #tira as fotos pela camera e atribia para rame
    if not success:
        break 

    frame = cv2.flip(frame, 1) #inverte valor de imagem de frame(horizotalmente)

    #convete a imagem para bgr, frmato qual mediapipe capaz de ler
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)#envia imagem para media pipe procssar

    #SE HOUVER ALGUMA MÃO NA IMAGEM ELE PROCEDE
    if result.multi_hand_landmarks:
        #array de mãos detectadas e percorre cada mão
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                #desenha os pontos da mão na imagem
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )


            if punho_fechado(hand_landmarks.landmark):
                #gesto detectado agora esceve o texto na tela
                cv2.putText(
                    frame,
                    "GESTO DETECTADO",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

    cv2.imshow("Projeto LIBRAS", frame)#frame é imagem do gesto detectado e texto

    if cv2.waitKey(1) & 0xFF == 27: #uebra ao pressinar a tecla "esc"
        break

cap.release()
cv2.destroyAllWindows()
