"""Entry point: wire up two paddle controllers and run the game loop.

Examples:
    # sanity check the physics with no LLM involved
    python -m pingpong_llm.main --left heuristic --right heuristic

    # a local Ollama model vs the scripted opponent
    python -m pingpong_llm.main --left ollama --model llama3.2 --right heuristic --render pygame

    # two different local models facing off
    python -m pingpong_llm.main --left ollama --model llama3.2 \\
        --right ollama --right-model qwen2.5:7b --render pygame
"""

from __future__ import annotations

import argparse
import time

from .controllers import Controller, HeuristicController, OllamaController
from .game import PongGame
from .render_ascii import clear_screen, render as render_ascii


def build_controller(kind: str, model: str, host: str, decision_interval: float) -> Controller:
    if kind == "heuristic":
        return HeuristicController()
    if kind == "ollama":
        return OllamaController(model=model, host=host, decision_interval=decision_interval)
    raise ValueError(f"unknown controller kind: {kind}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--left", choices=["heuristic", "ollama"], default="ollama")
    parser.add_argument("--right", choices=["heuristic", "ollama"], default="heuristic")
    parser.add_argument("--model", default="llama3.2", help="Ollama model for the left paddle")
    parser.add_argument("--right-model", default=None, help="Ollama model for the right paddle (defaults to --model)")
    parser.add_argument("--host", default="http://localhost:11434", help="Ollama server URL")
    parser.add_argument("--decision-interval", type=float, default=0.5, help="Seconds between LLM decisions")
    parser.add_argument("--render", choices=["ascii", "pygame", "none"], default="ascii")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--max-score", type=int, default=5)
    parser.add_argument("--max-seconds", type=float, default=None, help="Safety cutoff for the whole match")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    left = build_controller(args.left, args.model, args.host, args.decision_interval)
    right = build_controller(args.right, args.right_model or args.model, args.host, args.decision_interval)

    game = PongGame()

    renderer = None
    if args.render == "pygame":
        from .render_pygame import PygameRenderer

        renderer = PygameRenderer()

    dt = 1.0 / args.fps
    start = time.monotonic()
    try:
        while True:
            left_action = left.get_action(game.get_state("left"))
            right_action = right.get_action(game.get_state("right"))
            scored = game.step(dt, left_action, right_action)

            if args.render == "ascii":
                clear_screen()
                print(render_ascii(game))
            elif args.render == "pygame":
                if not renderer.render(game):
                    break

            if scored:
                print(f"{scored} scores! {game.left_score}-{game.right_score}")

            if game.left_score >= args.max_score or game.right_score >= args.max_score:
                print(f"Match over: {game.left_score}-{game.right_score}")
                break
            if args.max_seconds is not None and time.monotonic() - start > args.max_seconds:
                print("Time limit reached.")
                break

            time.sleep(dt)
    finally:
        left.close()
        right.close()
        if renderer is not None:
            renderer.close()


if __name__ == "__main__":
    main()
