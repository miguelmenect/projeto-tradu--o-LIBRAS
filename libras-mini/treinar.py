
import argparse
import json
import os
import time
import urllib.request
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")

import mediapipe as mp
import numpy as np
from matplotlib import pyplot as plt
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from tensorflow import keras
from tensorflow.keras import layers


CAMINHO_DETECTOR_PADRAO = "hand_landmarker.task"
URL_DETECTOR = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
CAMINHO_MODELO_PADRAO = "modelo_libras_mlp.h5"
CAMINHO_RELATORIO_PADRAO = "relatorio_treinamento.png"
EXTENSOES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VERSAO_ARTEFATO = 2

PULSO = 0
BASE_DEDO_MEDIO = 9
PONTAS_DEDOS = [4, 8, 12, 16, 20]
BASE_INDICADOR = 5
NUMERO_FEATURES = 72


def garantir_detector(caminho: str | Path) -> None:
    caminho = Path(caminho)
    if caminho.is_file():
        return
    urllib.request.urlretrieve(URL_DETECTOR, caminho)


def normalizar_landmarks(landmarks) -> np.ndarray:
    pontos = np.asarray([[p.x, p.y, p.z] for p in landmarks], dtype=np.float32)
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


def listar_classes(pasta_dataset: str | Path) -> list[tuple[str, list[Path]]]:
    raiz = Path(pasta_dataset)

    classes = []
    nomes_encontrados = set()
    for pasta in sorted(
        (item for item in raiz.iterdir() if item.is_dir()),
        key=lambda item: item.name.lower(),
    ):
        nome = pasta.name.strip().upper()
        if nome in nomes_encontrados:
            raise ValueError(f"Há mais de uma pasta representando a classe '{nome}'.")
        imagens = sorted(
            (
                item
                for item in pasta.iterdir()
                if item.is_file() and item.suffix.lower() in EXTENSOES
            ),
            key=lambda item: item.name.lower(),
        )
        if imagens:
            classes.append((nome, imagens))
            nomes_encontrados.add(nome)

    if len(classes) < 2:
        raise ValueError(
            "São necessárias pelo menos duas pastas com fotos, por exemplo "
            "'dataset/A/' e 'dataset/B/'."
        )
    return classes


def extrair_dataset(
    detector, classes: list[tuple[str, list[Path]]], minimo_por_classe: int
) -> dict[str, np.ndarray]:
    dados = {}
    for classe, imagens in classes:
        vetores = []
        sem_mao = 0
        print(f"\nClasse '{classe}': processando {len(imagens)} foto(s)...")
        for indice, caminho in enumerate(imagens, start=1):
            try:
                imagem = mp.Image.create_from_file(str(caminho))
                resultado = detector.detect(imagem)
            except Exception as erro:
                print(f"  Aviso: não foi possível ler '{caminho.name}': {erro}")
                continue

            if not resultado.hand_landmarks:
                sem_mao += 1
            else:
                vetores.append(normalizar_landmarks(resultado.hand_landmarks[0]))

            if indice % 50 == 0 or indice == len(imagens):
                print(f"  {indice}/{len(imagens)}", end="\r", flush=True)

        print(
            f"  {len(vetores)} amostra(s) válida(s); "
            f"{sem_mao} foto(s) sem mão detectada."
        )
        if len(vetores) < minimo_por_classe:
            raise ValueError(
                f"A classe '{classe}' tem apenas {len(vetores)} amostras válidas. "
                f"Capture pelo menos {minimo_por_classe} fotos válidas."
            )
        dados[classe] = np.asarray(vetores, dtype=np.float32)
    return dados


def criar_modelo(X_normalizacao: np.ndarray, numero_classes: int) -> keras.Model:
    normalizador = layers.Normalization(axis=-1)
    normalizador.adapt(X_normalizacao)

    modelo = keras.Sequential(
        [
            keras.Input(shape=(NUMERO_FEATURES,)),
            normalizador,
            layers.Dense(128, activation="relu"),
            layers.Dense(64, activation="relu"),
            layers.Dense(numero_classes, activation="softmax"),
        ]
    )
    modelo.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return modelo

def separar_validacao(dados: dict[str, np.ndarray]):
    X_treino, y_treino, X_validacao, y_validacao = [], [], [], []
    for classe, vetores in dados.items():
        corte = min(max(int(len(vetores) * 0.8), 1), len(vetores) - 1)
        X_treino.extend(vetores[:corte])
        y_treino.extend([classe] * corte)
        X_validacao.extend(vetores[corte:])
        y_validacao.extend([classe] * (len(vetores) - corte))
    return (
        np.asarray(X_treino, dtype=np.float32),
        np.asarray(y_treino),
        np.asarray(X_validacao, dtype=np.float32),
        np.asarray(y_validacao),
    )

def salvar_relatorio_visual(
    historico: keras.callbacks.History,
    y_val: np.ndarray,
    previsoes: np.ndarray,
    classes: list[str],
    acuracia: float,
    caminho: str | Path,
) -> None:
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)

    epocas = np.arange(1, len(historico.history["loss"]) + 1)
    figura = plt.figure(figsize=(17, 14))
    grade = figura.add_gridspec(2, 2, height_ratios=(1, 1.35))
    eixo_acuracia = figura.add_subplot(grade[0, 0])
    eixo_perda = figura.add_subplot(grade[0, 1])
    eixo_matriz = figura.add_subplot(grade[1, :])

    eixo_acuracia.plot(epocas, historico.history["accuracy"], label="Treino")
    eixo_acuracia.plot(
        epocas, historico.history["val_accuracy"], label="Validação"
    )
    eixo_acuracia.set_title("Acurácia por época")
    eixo_acuracia.set_xlabel("Época")
    eixo_acuracia.set_ylabel("Acurácia")
    eixo_acuracia.set_ylim(0, 1.02)
    eixo_acuracia.grid(alpha=0.3)
    eixo_acuracia.legend()
    eixo_perda.plot(epocas, historico.history["loss"], label="Treino")
    eixo_perda.plot(epocas, historico.history["val_loss"], label="Validação")
    eixo_perda.set_title("Perda por época")
    eixo_perda.set_xlabel("Época")
    eixo_perda.set_ylabel("Perda")
    eixo_perda.grid(alpha=0.3)
    eixo_perda.legend()

    matriz = confusion_matrix(y_val, previsoes, labels=classes, normalize="true")
    exibicao = ConfusionMatrixDisplay(confusion_matrix=matriz, display_labels=classes)
    exibicao.plot(
        ax=eixo_matriz,
        cmap="Blues",
        include_values=False,
        colorbar=True,
    )
    eixo_matriz.set_title("Matriz de confusão normalizada por letra")
    eixo_matriz.tick_params(axis="x", labelrotation=45)
    figura.suptitle(
        f"Validação acurácia: {acuracia * 100:.2f}%",
        fontsize=16,
    )
    figura.tight_layout(rect=(0, 0, 1, 0.96))
    figura.savefig(caminho, dpi=160, bbox_inches="tight")
    plt.close(figura)


def verificar_qualidade(
    acuracia: float,
    metricas: dict,
    classes: list[str],
    acuracia_minima: float,
    recall_minimo: float,
) -> None:
    """Impede que uma avaliação fraca substitua o último modelo salvo."""
    classes_com_recall_baixo = [
        classe
        for classe in classes
        if float(metricas[classe]["recall"]) < recall_minimo
    ]
    if acuracia >= acuracia_minima and not classes_com_recall_baixo:
        return

    motivos = []
    if acuracia < acuracia_minima:
        motivos.append(
            f"acurácia {acuracia * 100:.2f}% abaixo de "
            f"{acuracia_minima * 100:.2f}%"
        )
    if classes_com_recall_baixo:
        motivos.append(
            "recall abaixo do mínimo em: " + ", ".join(classes_com_recall_baixo)
        )
    raise RuntimeError(
        "O modelo avaliado não atingiu os critérios de qualidade ("
        + "; ".join(motivos)
        + "). O modelo existente não foi substituído. "
        "Confira o relatório em PNG e capture mais exemplos das letras indicadas."
    )


def treinar(args: argparse.Namespace) -> None:
    if args.sem_validacao:
        raise ValueError(
            "--sem-validacao foi desativado para evitar salvar um modelo sem "
            "avaliação. Execute o treino sem essa opção."
        )

    keras.utils.set_random_seed(args.semente)
    classes = listar_classes(args.dataset)
    garantir_detector(args.detector)

    opcoes = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=args.detector),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=args.confianca_deteccao,
    )
    with vision.HandLandmarker.create_from_options(opcoes) as detector:
        dados = extrair_dataset(detector, classes, args.minimo_por_classe)

    classes_ordenadas = sorted(dados.keys())
    mapa_classe_indice = {classe: indice for indice, classe in enumerate(classes_ordenadas)}
    numero_classes = len(classes_ordenadas)

    X_treino, y_treino, X_val, y_val = separar_validacao(dados)
    y_treino_idx = np.asarray([mapa_classe_indice[c] for c in y_treino])
    y_val_idx = np.asarray([mapa_classe_indice[c] for c in y_val])
    print(
        f"\nValidando com {len(X_treino)} amostras de treino e "
        f"{len(X_val)} amostras separadas..."
    )
    modelo = criar_modelo(X_treino, numero_classes)
    parada_antecipada = keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)
    historico = modelo.fit(
        X_treino,
        y_treino_idx,
        validation_data=(X_val, y_val_idx),
        epochs=300,
        batch_size=32,
        callbacks=[parada_antecipada],
        shuffle=True,
        verbose=2,
    )
    previsoes_idx = np.argmax(modelo.predict(X_val, verbose=0), axis=1)
    previsoes = np.asarray([classes_ordenadas[i] for i in previsoes_idx])
    acuracia = accuracy_score(y_val, previsoes)
    metricas = classification_report(
        y_val,
        previsoes,
        labels=classes_ordenadas,
        zero_division=0,
        output_dict=True,
    )
    print(f"Acurácia: {acuracia * 100:.2f}%")
    print(
        classification_report(
            y_val,
            previsoes,
            labels=classes_ordenadas,
            zero_division=0,
            digits=3,
        )
    )
    salvar_relatorio_visual(
        historico,
        y_val,
        previsoes,
        classes_ordenadas,
        acuracia,
        args.relatorio,
    )
    print(f"Relatório visual salvo em '{args.relatorio}'.")
    verificar_qualidade(
        acuracia,
        metricas,
        classes_ordenadas,
        args.acuracia_minima,
        args.recall_minimo,
    )

    melhor_epoca = int(np.argmin(historico.history["val_loss"]) + 1)
    caminho = Path(args.modelo)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    temporario = caminho.with_name(caminho.stem + ".tmp" + caminho.suffix)
    modelo.save(temporario)
    with h5py.File(temporario, "a") as arquivo_h5:
        arquivo_h5.attrs["versao"] = VERSAO_ARTEFATO
        arquivo_h5.attrs["numero_features"] = NUMERO_FEATURES
        arquivo_h5.attrs["classes"] = json.dumps(classes)
        arquivo_h5.attrs["melhor_epoca"] = melhor_epoca
    os.replace(temporario, caminho)

def inteiro_nao_negativo(valor: str) -> int:
    numero = int(valor)
    if numero < 0:
        raise argparse.ArgumentTypeError("o valor não pode ser negativo")
    return numero


def probabilidade(valor: str) -> float:
    numero = float(valor)
    if not 0 <= numero <= 1:
        raise argparse.ArgumentTypeError("o valor deve ficar entre 0 e 1")
    return numero


def criar_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extrai os pontos das fotos e treina o modelo de Libras."
    )
    parser.add_argument("--dataset", default="dataset")
    parser.add_argument("--modelo", default=CAMINHO_MODELO_PADRAO)
    parser.add_argument("--detector", default=CAMINHO_DETECTOR_PADRAO)
    parser.add_argument(
        "--relatorio",
        default=CAMINHO_RELATORIO_PADRAO,
        help="Imagem PNG com curvas e matriz de confusão (padrão: %(default)s).",
    )
    parser.add_argument("--minimo-por-classe", type=inteiro_positivo, default=20)
    parser.add_argument("--confianca-deteccao", type=probabilidade, default=0.5)
    parser.add_argument(
        "--acuracia-minima",
        type=probabilidade,
        default=0.90,
        help="Mínimo na validação para substituir o modelo salvo (padrão: 0.90).",
    )
    parser.add_argument(
        "--recall-minimo",
        type=probabilidade,
        default=0.70,
        help="Recall mínimo de cada letra para substituir o modelo (padrão: 0.70).",
    )
    parser.add_argument(
        "--semente",
        type=inteiro_nao_negativo,
        default=42,
        help="Semente para reproduzir o treinamento (padrão: 42).",
    )
    parser.add_argument(
        "--sem-validacao",
        action="store_true",
        help="Obsoleto: o treino seguro sempre exige validação.",
    )
    return parser


def main() -> None:
    treinar(criar_parser().parse_args())


if __name__ == "__main__":
    main()
