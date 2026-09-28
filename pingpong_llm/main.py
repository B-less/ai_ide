"""Entry point: wire up two paddle controllers and run the game loop.

Examples:
    # sanity check the physics with no LLM involved
    python -m pingpong_llm.main --left heuristic --right heuristic

    # a local Ollama model vs the scripted opponent
    python -m pingpong_llm.main --left ollama --model llama3.2 --right heuristic --render pygame

    # give a slow local model a bigger paddle and a slower ball
    python -m pingpong_llm.main --left ollama --model llama3.2 --right heuristic --preset slow-model

    # reasoning models (e.g. nemotron-3-nano) answer far faster with thinking off
    python -m pingpong_llm.main --left ollama --model nemotron-3-nano:4b --no-think --preset slow-model

    # two different local models facing off
    python -m pingpong_llm.main --left ollama --model llama3.2 \\
        --right ollama --right-model qwen2.5:7b --render pygame
"""

from __future__ import annotations

import argparse
import sys
import time

from .controllers import (
    Controller,
    HeuristicController,
    OllamaController,
    check_ollama_model,
    warm_up_ollama_model,
)
from .game import BALL_BASE_SPEED, BALL_SPEEDUP, PADDLE_HEIGHT, PADDLE_SPEED, PongGame
from .render_ascii import clear_screen, enable_ansi, render as render_ascii

# Bundles of paddle/ball physics tuned for how much reaction time a model needs.
# "slow-model" gives a bigger paddle and a much slower, non-accelerating ball,
# so infrequent/late decisions from a slow local model still connect.
PRESETS = {
    "default": {
        "paddle_speed": PADDLE_SPEED,
        "paddle_height": PADDLE_HEIGHT,
        "ball_speed": BALL_BASE_SPEED,
        "ball_speedup": BALL_SPEEDUP,
    },
    "slow-model": {
        "paddle_speed": PADDLE_SPEED + 40,
        "paddle_height": PADDLE_HEIGHT * 2.25,
        "ball_speed": BALL_BASE_SPEED * 0.25,
        "ball_speedup": 1.0,
    },
}


def build_controller(kind: str, model: str, args: argparse.Namespace) -> Controller:
    if kind == "heuristic":
        return HeuristicController()
    if kind == "ollama":
        return OllamaController(
            model=model,
            host=args.host,
            decision_interval=args.decision_interval,
            timeout=args.timeout,
            think=args.think,
        )
    raise ValueError(f"unknown controller kind: {kind}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--left", choices=["heuristic", "ollama"], default="ollama")
    parser.add_argument("--right", choices=["heuristic", "ollama"], default="heuristic")
    parser.add_argument("--model", default="llama3.2", help="Ollama model for the left paddle")
    parser.add_argument("--right-model", default=None, help="Ollama model for the right paddle (defaults to --model)")
    parser.add_argument("--host", default="http://localhost:11434", help="Ollama server URL")
    parser.add_argument("--decision-interval", type=float, default=0.5, help="Seconds between LLM decisions")
    parser.add_argument(
        "--timeout", type=float, default=30.0, help="Seconds to wait for one LLM reply before giving up (default: 30)"
    )
    parser.add_argument(
        "--think",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Turn a reasoning model's thinking on/off (--no-think is much faster). Omit to use the model's default",
    )
    parser.add_argument(
        "--preset",
        choices=list(PRESETS),
        default="default",
        help="Paddle/ball physics tuned for how fast the model can react (default: default)",
    )
    parser.add_argument("--paddle-speed", type=float, default=None, help="Override the preset's paddle speed (units/sec)")
    parser.add_argument("--paddle-height", type=float, default=None, help="Override the preset's paddle height")
    parser.add_argument("--ball-speed", type=float, default=None, help="Override the preset's base ball speed (units/sec)")
    parser.add_argument(
        "--ball-speedup", type=float, default=None, help="Override the preset's per-hit ball speed multiplier"
    )
    parser.add_argument("--render", choices=["ascii", "pygame", "none"], default="ascii")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--max-score", type=int, default=5)
    parser.add_argument("--max-seconds", type=float, default=None, help="Safety cutoff for the whole match")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    left_model = args.model
    right_model = args.right_model or args.model

    for kind, model in ((args.left, left_model), (args.right, right_model)):
        if kind == "ollama":
            problem = check_ollama_model(args.host, model)
            if problem:
                sys.exit(problem)
            print(f"Loading {model} into memory (first time can take a minute)...", flush=True)
            started = time.monotonic()
            problem = warm_up_ollama_model(args.host, model)
            if problem:
                print(f"  warm-up failed, continuing anyway: {problem}")
            else:
                print(f"  ready in {time.monotonic() - started:.1f}s")

    left = build_controller(args.left, left_model, args)
    right = build_controller(args.right, right_model, args)

    preset = PRESETS[args.preset]
    game = PongGame(
        paddle_speed=args.paddle_speed if args.paddle_speed is not None else preset["paddle_speed"],
        paddle_height=args.paddle_height if args.paddle_height is not None else preset["paddle_height"],
        ball_base_speed=args.ball_speed if args.ball_speed is not None else preset["ball_speed"],
        ball_speedup=args.ball_speedup if args.ball_speedup is not None else preset["ball_speedup"],
    )

    renderer = None
    if args.render == "ascii":
        enable_ansi()
    elif args.render == "pygame":
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
                for label, controller in (("L", left), ("R", right)):
                    status = controller.status()
                    if status:
                        print(f" {label}: {status}")
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
