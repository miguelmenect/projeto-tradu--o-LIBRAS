import json
import cv2
import h5py
import mediapipe as mp
import numpy as np
from tensorflow import keras
from txt_area import AreaDeTexto

# Constantes — precisam ser asmeesmas usadas no script de treino,
# senão o vetor de features fica diferente do que o modelo aprendeu

CAMINHO_MODELO = "modelo_libras_mlp.h5"
CAMINHO_DETECTOR = "hand_landmarker.task"
CONFIANCA_MINIMA = 0.70  # abaixo disso, ignora a previsao

LETRAS_PERMITIDAS = {"A", "B"}  # letras de teste a serem exibidas
ALTURA_BARRA_TEXTO = 60 

PULSO = 0
BASE_DEDO_MEDIO = 9
PONTAS_DEDOS = [4, 8, 12, 16, 20]
BASE_INDICADOR = 5


def normalizar_landmarks(landmarks) -> np.ndarray:
    """mesma funçao usada no treino — gera o vetor de 72 features."""
    pontos = np.asarray([[p.x, p.y, p.z] for p in landmarks], dtype=np.float32)

    #se o padrão de pontos de landmarks/esqueleto da mão não for o esperado, cai e erro
    if pontos.shape != (21, 3):
        raise ValueError(f"Esperados 21 pontos da mão; recebidos {pontos.shape}.")

    pontos = pontos - pontos[PULSO]
    escala = max(float(np.linalg.norm(pontos[BASE_DEDO_MEDIO])), 1e-6)
    pontos = pontos / escala

    fechamento_dedos = [float(np.linalg.norm(pontos[p])) for p in PONTAS_DEDOS]
    polegar_indicador = float(
        np.linalg.norm(pontos[PONTAS_DEDOS[0]] - pontos[BASE_INDICADOR])
    )
    espacamentos = [
        float(np.linalg.norm(pontos[PONTAS_DEDOS[i]] - pontos[PONTAS_DEDOS[i + 1]]))
        for i in range(1, len(PONTAS_DEDOS) - 1)
    ]

    return np.concatenate(
        (
            pontos.ravel(),
            np.asarray(fechamento_dedos, dtype=np.float32),
            np.asarray([polegar_indicador], dtype=np.float32),
            np.asarray(espacamentos, dtype=np.float32),
        )
    ).astype(np.float32, copy=False)

#função recebe caminho do do modelo_libras_mlp.h5
def carregar_modelo(caminho: str):
    #carrega modelo keras e as classes salvas nos atributos do h5

    #aqui o modelo recebe o caminho do arquivo .h5 e carrega o modelo e as classes
    modelo = keras.models.load_model(caminho)

    #efetua leitura do arquivo de treino .h5
    with h5py.File(caminho, "r") as arquivo_h5:
        #classe recebe valor da lista de letras do arquivo de treino 
        classes = json.loads(arquivo_h5.attrs["classes"])
    return modelo, classes

#recebe modelo já carregado, a lista de letras e os ontos da mão(landmarks)
def prever_letra(modelo, classes, landmarks):
    #Roda modelo em uma mão detectada e retorna (letra, confianca/precisão)

    #transforma o landmarks em um array de 72 numeros que é o o que o medolo entende
    features = normalizar_landmarks(landmarks)

    #formata o array em um formato espera para receber esses dados
    features = np.expand_dims(features, axis=0)

    #recebe esse array para o modelo e retorna a letra prevista e a confiança dessa previsão
    probabilidades = modelo.predict(features, verbose=0)[0]
    indice = int(np.argmax(probabilidades))
    return classes[indice], float(probabilidades[indice])


def desenhar_barra_de_texto(frame, texto: str):
    altura, largura = frame.shape[:2]
 
    sobreposicao = frame.copy()
    cv2.rectangle(
        sobreposicao,
        (0, altura - ALTURA_BARRA_TEXTO),
        (largura, altura),
        (0, 0, 0),
        -1,
    )
    cv2.addWeighted(sobreposicao, 0.6, frame, 0.4, 0, dst=frame)
 
    texto_exibido = texto if texto else "_"
    cv2.putText(
        frame,
        texto_exibido,
        (15, altura - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
    )


def main() -> None:
    modelo, classes = carregar_modelo(CAMINHO_MODELO)
    print(f"Modelo carregado. Filtrando apenas: {sorted(LETRAS_PERMITIDAS)}")

    area_de_texto = AreaDeTexto(frames_para_confirmar=15)

    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=CAMINHO_DETECTOR),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=1,
    )

    cap = cv2.VideoCapture(0)

    with HandLandmarker.create_from_options(options) as detector:
        while True:
            success, frame = cap.read()
            if not success:
                break

            # flip(frame, 1) espelha horizontalmente (efeito "espelho" da webcam)
            frame = cv2.flip(frame, 1)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            result = detector.detect(mp_image)

            letra_valida_no_frame = None

            if result.hand_landmarks:
                for hand_landmarks in result.hand_landmarks:
                    try:
                        letra, confianca = prever_letra(modelo, classes, hand_landmarks)
                    except ValueError:
                        continue

                    # só reage se a letra prevista estiver na lista permitida
                    # e a acuracy for alta o suficiente
                    if letra in LETRAS_PERMITIDAS and confianca >= CONFIANCA_MINIMA:
                        texto = f"{letra} ({confianca * 100:.0f}%)"
                        cv2.putText(
                            frame,
                            texto,
                            (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.2,
                            (0, 255, 0),
                            2,
                        )
                        letra_valida_no_frame = letra
                    break

            area_de_texto.atualizar(letra_valida_no_frame)
 
            desenhar_barra_de_texto(frame, area_de_texto.obter_texto())

            cv2.imshow("Projeto LIBRAS", frame)

            if cv2.waitKey(1) & 0xFF == 27:  #sai do loop quando clica no esc
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()