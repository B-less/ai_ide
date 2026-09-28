# pingpong_llm

A tiny Pong-style physics sim where a local [Ollama](https://ollama.com) model
controls a paddle by reading the game state as text and replying with a move.

This is **not** OpenAI's hide-and-seek experiment (that used reinforcement
learning trained over millions of self-play episodes on a physics engine).
Here a pretrained LLM is just prompted once every `--decision-interval`
seconds to pick `UP`, `DOWN`, or `STAY` — no training/learning loop, and the
model won't get better over time. It's a quick way to see how well an LLM can
reason about a simple real-time control task.

## How it works

- `game.py` — dependency-free ball/paddle physics (`PongGame`).
- `controllers.py` — `HeuristicController` (scripted, tracks the ball) and
  `OllamaController` (queries `POST /api/generate` on a local Ollama server
  in a background thread, so slow inference never blocks the physics loop —
  the paddle just keeps its last decided action until a new one arrives).
- `render_ascii.py` — zero-dependency terminal renderer.
- `render_pygame.py` — optional windowed renderer (needs `pygame`).
- `main.py` — CLI that wires two controllers together and runs the loop.

## Setup

```bash
pip install -r requirements.txt      # requests only
pip install pygame                   # optional, for --render pygame

ollama serve                         # if not already running
ollama pull llama3.2                 # or any model you have
```

## Run it

```bash
# sanity-check the physics with no LLM involved
python -m pingpong_llm.main --left heuristic --right heuristic

# a local model vs. the scripted opponent, ascii rendering
python -m pingpong_llm.main --left ollama --model llama3.2 --right heuristic

# same, with a real window
python -m pingpong_llm.main --left ollama --model llama3.2 --right heuristic --render pygame

# two different local models facing off
python -m pingpong_llm.main \
  --left ollama --model llama3.2 \
  --right ollama --right-model qwen2.5:7b \
  --render pygame
```

Useful flags: `--decision-interval` (seconds between LLM calls, default 0.5 —
lower is more responsive but hammers Ollama harder), `--host` (Ollama server
URL if not on localhost:11434), `--max-score`, `--max-seconds`.

## Tuning ideas

- Faster/slower ball: edit `BALL_BASE_SPEED` / `BALL_SPEEDUP` in `game.py`.
- Give the model less time pressure: raise `PADDLE_SPEED` down, or lower
  `BALL_BASE_SPEED`, so a 0.5s decision cadence is enough to react.
- Try a chat-style prompt with few-shot examples instead of the raw
  `/api/generate` call in `controllers.py::build_prompt` if a model's
  one-word replies are unreliable.
