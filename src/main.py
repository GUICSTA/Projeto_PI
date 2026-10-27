"""
Uso:
    python -m src.main caminho/para/imagem.jpg [--saida pasta] [--morfologia abertura|fechamento]

"""
import argparse

from . import classify, preprocessing, visualize


def main(argv=None):
    parser = argparse.ArgumentParser(description="Identifica o naipe de uma carta de baralho.")
    parser.add_argument("imagem")
    parser.add_argument("--saida", help="pasta para salvar as saídas intermediárias")
    parser.add_argument("--morfologia", choices=["abertura", "fechamento"])
    args = parser.parse_args(argv)

    rgb = preprocessing.load_rgb(args.imagem)
    extracted = classify.extract(rgb, morphology=args.morfologia)
    rules = (classify.load_rules() if extracted["foto"] else None) or classify.SYNTHETIC_RULES
    suit = classify.decide(extracted["cor"], extracted["features"], rules)

    print(f"Imagem: {args.imagem}")
    print(f"Tipo: {'foto (segmentação do canto)' if extracted['foto'] else 'sintética (imagem inteira)'}")
    print(f"Naipe identificado: {suit}")
    print(f"Cor detectada: {extracted['cor']}")
    print(f"Threshold (Otsu): {extracted['threshold_otsu']}")
    print(f"Características: { {k: round(float(v), 4) for k, v in extracted['features'].items()} }")

    if args.saida and extracted["foto"]:
        corner = visualize.save_intermediate(rgb, args.saida, args.morfologia)
        print(f"Caixa do valor (y0, y1, x0, x1): {tuple(int(v) for v in corner['valor'])}")
        print(f"Caixa do naipe (y0, y1, x0, x1): {tuple(int(v) for v in corner['naipe'])}")
        print(f"Saídas intermediárias salvas em: {args.saida}")


if __name__ == "__main__":
    main()
