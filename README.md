# ai_ide

Experiments with putting AI models into small simulations.

## Projects

- [`pingpong_llm/`](pingpong_llm/README.md): a Pong-style physics sim where a
  local or cloud [Ollama](https://ollama.com) model controls a paddle by
  reading the game state as text and replying `UP`, `DOWN`, or `STAY`. Play
  it against a scripted opponent or another model, in the terminal or a
  pygame window.

Quick start (after setting up a virtual environment as described in the
project README):

```bash
python -m pingpong_llm.main --left heuristic --right heuristic  # no LLM needed
python -m pingpong_llm.main --no-think                          # nemotron-3-nano:30b-cloud vs. a bot
```

Run the tests with `python -m unittest discover -s tests`.
