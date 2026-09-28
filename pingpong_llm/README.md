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

Run these from the repo root (`ai_ide/`). Use a virtual environment: recent
Ubuntu/Debian refuse a system-wide `pip install` (the
`externally-managed-environment` error), and a venv needs no `sudo`.

Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1           # if blocked: Set-ExecutionPolicy -Scope Process Bypass
pip install -r pingpong_llm\requirements.txt
```

Linux / macOS:

```bash
python3 -m venv .venv                # on Ubuntu, may first need: sudo apt install python3-venv
source .venv/bin/activate
pip install -r pingpong_llm/requirements.txt
```

Optional: `pip install pygame` for `--render pygame`. Re-activate the venv in
every new terminal.

Then make sure Ollama has a model: `ollama list` to see what you have,
`ollama pull llama3.2` to get one. Cloud models (names ending in `-cloud`,
e.g. `nemotron-3-nano:30b-cloud`) work too after `ollama signin`; they go
through the same local Ollama API, so nothing else changes.

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

The script checks up front that Ollama is reachable and the model is
installed, and exits with a clear message if not. In ascii mode, each LLM
paddle gets a status line under the board showing its current move, how long
its last reply took, and any error — the first thing to look at if a paddle
seems frozen.

Useful flags: `--decision-interval` (seconds between LLM calls, default 0.5 —
lower is more responsive but hammers Ollama harder), `--timeout` (seconds to
wait for one reply, default 30), `--host` (Ollama server URL if not on
localhost:11434), `--max-score`, `--max-seconds`.

## Tuning for slower models

A slow local model might only produce a decision every few seconds (query
time stacks on top of `--decision-interval`), so the default physics — tuned
for a decision roughly every 0.5s — can be unfair: the ball crosses the whole
court before a bigger/laggier model gets a second move in.

```bash
python -m pingpong_llm.main --left ollama --model <slow-model> --right heuristic --preset slow-model
```

`--preset slow-model` gives you a paddle more than double the default height,
a ball at a quarter of the default speed (roughly a 12s full-court crossing
instead of ~3s), and no per-hit speedup (rallies don't ramp up), so
infrequent, late corrections from CPU-bound local models still connect. Tune
further with `--paddle-speed`, `--paddle-height`, `--ball-speed`, and
`--ball-speedup`, which override individual values from whichever `--preset`
you picked — e.g. `--ball-speed 30` if even `slow-model` feels too fast.

### Reasoning models: turn thinking off

Models like `nemotron-3-nano`, `qwen3`, and `deepseek-r1` think before they
answer, which can turn a sub-second reply into many seconds. For a one-word
move that thinking rarely helps, so try `--no-think` first:

```bash
python -m pingpong_llm.main --left ollama --model nemotron-3-nano:4b --no-think --preset slow-model
```

Leave the flag off to use the model's own default; `--think` forces it on
(fun for comparing whether reasoning plays better despite reacting later).

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `Can't reach Ollama at ...` | Start the Ollama app or run `ollama serve`; check `--host`. |
| `Model '...' isn't available` | `ollama pull <model>`, or pick one from the listed installed models. |
| Status line shows `ERROR ... timed out` | The model is slower than `--timeout`; raise it, add `--no-think`, or use a smaller model. |
| Paddle moves but always too late | `--preset slow-model`, then lower `--ball-speed` further. |
| Garbage like `←[H←[J` instead of a redrawn board | Use Windows Terminal, or `--render pygame`. |
| `externally-managed-environment` from pip | Use the venv setup above. |

## Tests

```bash
python -m unittest discover -s tests
```

Covers move parsing (including inline `<think>` blocks), paddle/ball physics,
scoring, and the presets. No Ollama needed.

## Other tuning ideas

- Try a chat-style prompt with few-shot examples instead of the raw
  `/api/generate` call in `controllers.py::build_prompt` if a model's
  one-word replies are unreliable.
