import cv2
import mediapipe as mp
import numpy as np

# inicializa o mediapipe
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=VisionRunningMode.IMAGE
)

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

    dedos = [(8, 6), (12, 10), (16, 14), (20, 18)]  # pontas dos dedos
    base = landmarks[0]  # punho

    fechado = True
    for ponta, pip in dedos:
            dist_ponta = distancia(landmarks[ponta], base)
            dist_pip = distancia(landmarks[pip], base)
    
            if dist_ponta > dist_pip: #se o dedo encontrado no array for menor que punho então "mão fechada" é false
                fechado = False #mão fechada é false

    return fechado

def distancia(p1, p2):
    return ((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2) ** 0.5

with HandLandmarker.create_from_options(options) as detector:
    while True:
        
        success, frame = cap.read() #tira as fotos pela camera e atribia para rame
       
        if not success:
            break 

        frame = cv2.flip(frame, 0) #inverte valor de imagem de frame(horizotalmente)
        
        print('frame2', frame)
        #convete a imagem para bgr, frmato qual mediapipe capaz de ler
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #SE HOUVER ALGUMA MÃO NA IMAGEM ELE PROCEDE
      
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame_rgb
        )
        result = detector.detect(mp_image)
        if result.hand_landmarks:
                #array de mãos detectadas e percorre cada mão
                for hand_landmarks in result.hand_landmarks:
                    if punho_fechado(hand_landmarks):
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

        cv2.imshow("Projeto LIBRAS", frame)

        if cv2.waitKey(1) & 0xFF == 27: #sai do loop quando clica no esc
            break

cap.release()
cv2.destroyAllWindows()
