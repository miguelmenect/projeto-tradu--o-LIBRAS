"""
Olá professores ou quem quer que esteja lendo eu tive que publicar todo o dataset
no google drive por conta do tamanho da imagem eu vou colocar o link abaixo para que
você possa acessar vou deixar disponivel até a correção do tcc desde já agradeço a 
compreensão e me desculpe por ter que colocar as imagens em outro lugar.

link: https://drive.google.com/drive/folders/10j2qGn-FrpO0KwCTCxIu4izh7af92qRP?usp=sharing
"""
import argparse
import time
from pathlib import Path
import cv2

def capturar(args: argparse.Namespace) -> None:
    classe = args.classe
    if classe is None:
        rec_classe = input("Qual letra ou sinal deseja capturar? ")
        classe = rec_classe.strip().upper()

    pasta_saida = Path(args.dataset) / classe
    pasta_saida.mkdir(parents=True, exist_ok=True)
    maior_indice = 0
    for arquivo in pasta_saida.glob("*.jpg"):
        nome = arquivo.stem
        if nome.isdigit():
            numero = int(nome)
            if numero > maior_indice:
                maior_indice = numero
    indice = maior_indice + 1
    camera = cv2.VideoCapture(args.camera)
    inicio_contagem = None
    proxima_foto = None
    salvas = 0
    try:
        while salvas < args.quantidade:
            _, frame = camera.read()
            altura, largura = frame.shape[:2]
            lado = int(min(largura, altura) * 0.62)
            x1 = (largura - lado) // 2
            y1 = (altura - lado) // 2
            x2 = x1 + lado
            y2 = y1 + lado
            agora = time.monotonic()
            restante = None
            if inicio_contagem is not None:
                restante = max(0.0, inicio_contagem + args.espera - agora)

            if (restante is not None and restante <= 0 and proxima_foto is not None and agora >= proxima_foto):
                caminho = pasta_saida / f"{indice:05d}.jpg"
                regiao_mao = frame[y1:y2, x1:x2].copy()
                if not cv2.imwrite(str(caminho), regiao_mao):
                    raise OSError(f"Não foi possível salvar '{caminho}'.")
                salvas += 1
                indice += 1
                proxima_foto = agora + args.intervalo

            exibicao = cv2.flip(frame, 1)
            cv2.rectangle(exibicao, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(
                exibicao,
                f"Letra: {classe}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
            )
            if inicio_contagem is None:
                status = "Aperte ESPACO para iniciar"
            elif restante is not None and restante > 0:
                status = f"Teeempo: {restante:.1f}s"
            else:
                status = f"Fotos: {salvas}/{args.quantidade}"
            cv2.putText(
                exibicao,
                status,
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 220, 255),
                2,
            )
            cv2.putText(
                exibicao,
                "Q: sair",
                (20, exibicao.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )
            cv2.imshow("Captura da letra de Libras", exibicao)
            tecla = cv2.waitKey(1) & 0xFF
            if tecla == ord("q") or tecla == 27:
                break
            if tecla == ord(" ") and inicio_contagem is None:
                inicio_contagem = time.monotonic()
                proxima_foto = inicio_contagem + args.espera
               
    finally:
        camera.release()
        cv2.destroyAllWindows()

parser = argparse.ArgumentParser(description="Captura fotos da webcam em uma pasta por letra/sinal.")
parser.add_argument("classe", nargs="?")
parser.add_argument("--dataset", default="dataset")
parser.add_argument("--quantidade", type=int, default=400)
parser.add_argument("--intervalo", type=float, default=0.10)
parser.add_argument("--espera", type=float,default=2.0)
parser.add_argument("--camera", type=int, default=0)
parser.add_argument("--largura", type=int, default=1280)
parser.add_argument("--altura", type=int, default=720)
args = parser.parse_args()
capturar(args)
