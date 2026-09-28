"""Paddle controllers: a scripted heuristic and an Ollama-backed LLM agent."""

from __future__ import annotations

import re
import threading
import time
from abc import ABC, abstractmethod

import requests

VALID_ACTIONS = ("UP", "DOWN", "STAY")
_ACTION_RE = re.compile(r"\b(UP|DOWN|STAY)\b", re.IGNORECASE)


def parse_action(text: str) -> str:
    match = _ACTION_RE.search(text or "")
    return match.group(1).upper() if match else "STAY"


class Controller(ABC):
    @abstractmethod
    def get_action(self, state: dict) -> str:
        """Return one of UP/DOWN/STAY given the current game state. Must not block."""

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


class OllamaController(Controller):
    """Queries a local Ollama model in the background and caches its latest decision.

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
        timeout: float = 10.0,
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.decision_interval = decision_interval
        self.timeout = timeout

        self._lock = threading.Lock()
        self._latest_state: dict | None = None
        self._action = "STAY"
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def get_action(self, state: dict) -> str:
        with self._lock:
            self._latest_state = state
            return self._action

    def _loop(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                state = self._latest_state
            if state is not None:
                action = self._query(state)
                if action is not None:
                    with self._lock:
                        self._action = action
            self._stop.wait(self.decision_interval)

    def _query(self, state: dict) -> str | None:
        prompt = build_prompt(state)
        try:
            resp = requests.post(
                f"{self.host}/api/generate",
                json={"model": self.model, "prompt": prompt, "stream": False},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return parse_action(resp.json().get("response", ""))
        except requests.RequestException as exc:
            print(f"[ollama:{self.model}] request failed, keeping last action: {exc}")
            return None

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1)
