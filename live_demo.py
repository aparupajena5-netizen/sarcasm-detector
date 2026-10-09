"""Simulated live-stream chat moderation.

Messages arrive one by one. For each message the *previous chat message* is used as context,
the model scores it in real time, and flagged messages are printed with the words the
attention mechanism focused on. Also tracks messages/second throughput.

    python live_demo.py                          # replay sample_chat.txt
    python live_demo.py --file my_chat.txt --delay 0.5
    python live_demo.py --interactive            # you type the live chat
File format: one message per line, optionally "username: message".
"""
import argparse
import time
from collections import deque

from src.inference import SarcasmPredictor


def parse_line(line: str):
    line = line.strip()
    if ": " in line and len(line.split(": ", 1)[0]) <= 24:
        user, msg = line.split(": ", 1)
        return user, msg
    return "anon", line


def stream_lines(args):
    if args.interactive:
        print("Type chat messages (Ctrl+C to stop).")
        while True:
            yield input("chat> ")
    else:
        with open(args.file, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield line
                    time.sleep(args.delay)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/lstm_attn_ctx.pt")
    ap.add_argument("--file", default="sample_chat.txt")
    ap.add_argument("--delay", type=float, default=0.3, help="seconds between messages")
    ap.add_argument("--context-window", type=int, default=1, help="how many previous messages form the context")
    ap.add_argument("--interactive", action="store_true")
    args = ap.parse_args()

    pred = SarcasmPredictor(args.ckpt)
    history = deque(maxlen=args.context_window)
    n = flagged = 0
    infer_time = 0.0
    try:
        for raw in stream_lines(args):
            user, msg = parse_line(raw)
            context = " ".join(history)
            t0 = time.perf_counter()
            res = pred.predict(msg, context)
            infer_time += time.perf_counter() - t0
            n += 1
            if res["flag"]:
                flagged += 1
                words = ", ".join(w for w, _ in res["top_words"])
                print(f"[FLAG {res['prob']:.2f}] {user}: {msg}    <- focus: {words}")
            else:
                print(f"[ ok  {res['prob']:.2f}] {user}: {msg}")
            history.append(msg)
    except (KeyboardInterrupt, EOFError):
        pass
    if n:
        print(f"\n{n} messages, {flagged} flagged ({flagged / n:.0%}), "
              f"avg latency {1000 * infer_time / n:.1f} ms/message")


if __name__ == "__main__":
    main()
