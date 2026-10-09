"""Score one comment (optionally with the message it replies to).

    python predict.py --comment "Oh great, another 3 hour queue. Love that." --context "server is down again"
    python predict.py            # interactive mode
"""
import argparse

from src.inference import SarcasmPredictor


def show(res: dict) -> None:
    label = "SARCASTIC / PASSIVE-AGGRESSIVE" if res["flag"] else "not flagged"
    print(f"  -> {label}  (score {res['prob']:.2f})  attention: "
          + ", ".join(f"{w} ({a})" for w, a in res["top_words"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/lstm_attn_ctx.pt")
    ap.add_argument("--comment")
    ap.add_argument("--context", default="")
    args = ap.parse_args()
    pred = SarcasmPredictor(args.ckpt)

    if args.comment:
        show(pred.predict(args.comment, args.context))
        return
    print("Interactive mode. Empty comment to quit.")
    while True:
        ctx = input("\nPrevious message (optional): ").strip()
        com = input("Comment: ").strip()
        if not com:
            break
        show(pred.predict(com, ctx))


if __name__ == "__main__":
    main()
