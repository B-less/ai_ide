"""Paddle controllers: a scripted heuristic and an Ollama-backed LLM agent."""

from __future__ import annotations

import re
import threading
import time
from abc import ABC, abstractmethod

import requests

_ACTION_RE = re.compile(r"\b(UP|DOWN|STAY)\b", re.IGNORECASE)
_THINK_RE = re.compile(r"<think>.*?(</think>|$)", re.IGNORECASE | re.DOTALL)


def parse_action(text: str) -> str:
    # Some reasoning models inline their chain of thought, which mentions every move.
    answer = _THINK_RE.sub("", text or "")
    match = _ACTION_RE.search(answer)
    return match.group(1).upper() if match else "STAY"


class Controller(ABC):
    @abstractmethod
    def get_action(self, state: dict) -> str:
        """Return one of UP/DOWN/STAY given the current game state. Must not block."""

    def status(self) -> str | None:
        return None

    def close(self) -> None:
        pass


class HeuristicController(Controller):
    """Simple scripted opponent: chases the ball's y position."""

    def get_action(self, state: dict) -> str:
        deadzone = state["own_height"] * 0.15
        diff = state["ball_y"] - state["own_y"]
        if diff > deadzone:
            return "DOWN"
        if diff < -deadzone:
            return "UP"
        return "STAY"


def build_prompt(state: dict) -> str:
    return (
        f"You are controlling the {state['side'].upper()} paddle in a Pong-like game.\n"
        f"Court size: {state['court_width']:.0f}x{state['court_height']:.0f} (x grows right, y grows down).\n"
        f"Your paddle: x={state['own_x']:.0f}, y={state['own_y']:.0f}, height={state['own_height']:.0f}.\n"
        f"Opponent paddle: x={state['opp_x']:.0f}, y={state['opp_y']:.0f}, height={state['opp_height']:.0f}.\n"
        f"Ball: x={state['ball_x']:.0f}, y={state['ball_y']:.0f}, "
        f"vx={state['ball_vx']:.0f}, vy={state['ball_vy']:.0f}.\n"
        f"Score: you {state['own_score']} - opponent {state['opp_score']}.\n"
        "Move your paddle to intercept the ball and score on the opponent's side.\n"
        "Respond with EXACTLY one word: UP, DOWN, or STAY. No explanation."
    )


def check_ollama_model(host: str, model: str) -> str | None:
    """Return a human-readable problem if the Ollama server or model isn't usable, else None."""
    host = host.rstrip("/")
    try:
        resp = requests.get(f"{host}/api/tags", timeout=5)
        resp.raise_for_status()
    except requests.RequestException as exc:
        return f"Can't reach Ollama at {host} ({exc}). Is the Ollama app / `ollama serve` running?"
    names = {m.get("name") for m in resp.json().get("models", [])}
    if model in names or f"{model}:latest" in names:
        return None
    available = ", ".join(sorted(n for n in names if n)) or "none"
    return f"Model '{model}' isn't available in Ollama. Run `ollama pull {model}`. Installed: {available}"


class OllamaController(Controller):
    """Queries an Ollama model in the background and caches its latest decision.

    Physics runs in real time and LLM inference is comparatively slow, so this
    controller never blocks the game loop: `get_action` just returns whatever
    action a background thread last decided, and hands it the freshest state
    to think about next.
    """

    def __init__(
        self,
        model: str,
        host: str = "http://localhost:11434",
        decision_interval: float = 0.5,
        timeout: float = 30.0,
        think: bool | None = None,
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.decision_interval = decision_interval
        self.timeout = timeout
        self.think = think

        self._lock = threading.Lock()
        self._latest_state: dict | None = None
        self._action = "STAY"
        self._decisions = 0
        self._last_latency: float | None = None
        self._last_error: str | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def get_action(self, state: dict) -> str:
        with self._lock:
            self._latest_state = state
            return self._action

    def status(self) -> str:
        with self._lock:
            if self._last_error:
                return f"{self.model}: ERROR {self._last_error} (holding {self._action})"
            if self._last_latency is None:
                return f"{self.model}: waiting for first reply..."
            return (
                f"{self.model}: {self._action:<4} | last reply {self._last_latency:.1f}s"
                f" | {self._decisions} decisions"
            )

    def _loop(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                state = self._latest_state
            if state is not None:
                self._query(state)
            self._stop.wait(self.decision_interval)

    def _query(self, state: dict) -> None:
        payload = {"model": self.model, "prompt": build_prompt(state), "stream": False}
        if self.think is not None:
            payload["think"] = self.think
        started = time.monotonic()
        try:
            resp = requests.post(f"{self.host}/api/generate", json=payload, timeout=self.timeout)
            resp.raise_for_status()
            action = parse_action(resp.json().get("response", ""))
        except requests.RequestException as exc:
            error = (str(exc) or type(exc).__name__).splitlines()[0][:120]
            with self._lock:
                is_new = error != self._last_error
                self._last_error = error
            if is_new:
                print(f"[ollama:{self.model}] request failed, keeping last action: {error}")
            return
        with self._lock:
            self._action = action
            self._decisions += 1
            self._last_latency = time.monotonic() - started
            self._last_error = None

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1)
