"""
Uso:
    python -m src.main caminho/para/imagem.jpg [--saida pasta] [--morfologia abertura|fechamento] [--mediana RAIO]

"""
import argparse

import sys

from . import classify, preprocessing, rank, segmentation, visualize


def main(argv=None):
    parser = argparse.ArgumentParser(description="Identifica o naipe de uma carta de baralho.")
    parser.add_argument("imagem")
    parser.add_argument("--saida", help="pasta para salvar as saídas intermediárias")
    parser.add_argument("--morfologia", choices=["abertura", "fechamento"])
    parser.add_argument("--mediana", type=int, default=0, help="raio do filtro de mediana antes do Otsu (0 = desligado)")
    args = parser.parse_args(argv)

    rgb = preprocessing.load_rgb(args.imagem)
    try:
        extracted = classify.extract(rgb, morphology=args.morfologia, median=args.mediana)
    except segmentation.JokerFound:
        print(f"Imagem: {args.imagem}")
        print("Carta identificada: coringa (sem naipe)")
        return 0
    except ValueError as error:
        print(f"Imagem: {args.imagem}")
        print(f"Falha na segmentação: {error}")
        return 1
    rules = (classify.load_rules() if extracted["foto"] else None) or classify.SYNTHETIC_RULES
    suit = classify.decide(extracted["cor"], extracted["features"], rules)

    print(f"Imagem: {args.imagem}")
    print(f"Tipo: {'foto (segmentação do canto)' if extracted['foto'] else 'sintética (imagem inteira)'}")
    print(f"Naipe identificado: {suit}")
    templates = rank.load_templates()
    if extracted["canto"] and templates:
        print(f"Valor identificado: {rank.recognize(extracted['canto'], templates)}")
    print(f"Cor detectada: {extracted['cor']}")
    print(f"Threshold (Otsu): {extracted['threshold_otsu']}")
    print(f"Características: { {k: round(float(v), 4) for k, v in extracted['features'].items()} }")

    if args.saida and extracted["foto"]:
        corner = visualize.save_intermediate(rgb, args.saida, args.morfologia)
        print(f"Caixa do valor (y0, y1, x0, x1): {tuple(int(v) for v in corner['valor'])}")
        print(f"Caixa do naipe (y0, y1, x0, x1): {tuple(int(v) for v in corner['naipe'])}")
        print(f"Saídas intermediárias salvas em: {args.saida}")


if __name__ == "__main__":
    sys.exit(main())
