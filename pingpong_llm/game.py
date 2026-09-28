"""Minimal Pong-style physics for the ping-pong LLM demo."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

WIDTH = 800.0
HEIGHT = 400.0

PADDLE_HEIGHT = 80.0
PADDLE_MARGIN = 20.0
PADDLE_SPEED = 300.0  # units/sec

BALL_RADIUS = 8.0
BALL_BASE_SPEED = 260.0
BALL_SPEEDUP = 1.05
BALL_MAX_SPEED = 600.0


@dataclass
class Paddle:
    x: float
    y: float = HEIGHT / 2
    height: float = PADDLE_HEIGHT

    def move(self, action: str, dt: float) -> None:
        if action == "UP":
            self.y -= PADDLE_SPEED * dt
        elif action == "DOWN":
            self.y += PADDLE_SPEED * dt
        half = self.height / 2
        self.y = max(half, min(HEIGHT - half, self.y))


@dataclass
class Ball:
    x: float = WIDTH / 2
    y: float = HEIGHT / 2
    vx: float = 0.0
    vy: float = 0.0


@dataclass
class PongGame:
    left: Paddle = field(default_factory=lambda: Paddle(x=PADDLE_MARGIN))
    right: Paddle = field(default_factory=lambda: Paddle(x=WIDTH - PADDLE_MARGIN))
    ball: Ball = field(default_factory=Ball)
    left_score: int = 0
    right_score: int = 0

    def __post_init__(self) -> None:
        self.reset_ball()

    def reset_ball(self, serve_towards: str | None = None) -> None:
        self.ball.x = WIDTH / 2
        self.ball.y = HEIGHT / 2
        direction = 1 if serve_towards == "right" else -1 if serve_towards == "left" else random.choice([-1, 1])
        angle = random.uniform(-0.35, 0.35)
        self.ball.vx = direction * BALL_BASE_SPEED
        self.ball.vy = BALL_BASE_SPEED * angle

    def step(self, dt: float, left_action: str, right_action: str) -> str | None:
        """Advance the simulation by dt seconds. Returns 'left'/'right' if that side just scored."""
        self.left.move(left_action, dt)
        self.right.move(right_action, dt)

        ball = self.ball
        ball.x += ball.vx * dt
        ball.y += ball.vy * dt

        if ball.y - BALL_RADIUS <= 0 and ball.vy < 0:
            ball.y = BALL_RADIUS
            ball.vy = -ball.vy
        elif ball.y + BALL_RADIUS >= HEIGHT and ball.vy > 0:
            ball.y = HEIGHT - BALL_RADIUS
            ball.vy = -ball.vy

        self._maybe_bounce(self.left, going_left=True)
        self._maybe_bounce(self.right, going_left=False)

        if ball.x < 0:
            self.right_score += 1
            self.reset_ball(serve_towards="right")
            return "right"
        if ball.x > WIDTH:
            self.left_score += 1
            self.reset_ball(serve_towards="left")
            return "left"
        return None

    def _maybe_bounce(self, paddle: Paddle, going_left: bool) -> None:
        ball = self.ball
        moving_towards = ball.vx < 0 if going_left else ball.vx > 0
        if not moving_towards:
            return
        within_x = (
            abs(ball.x - paddle.x) <= BALL_RADIUS + 6
        )
        half = paddle.height / 2
        within_y = paddle.y - half <= ball.y <= paddle.y + half
        if within_x and within_y:
            ball.vx = -ball.vx * BALL_SPEEDUP
            ball.vx = max(-BALL_MAX_SPEED, min(BALL_MAX_SPEED, ball.vx))
            offset = (ball.y - paddle.y) / half  # -1..1, where on the paddle it hit
            ball.vy += offset * 150
            ball.vy = max(-BALL_MAX_SPEED, min(BALL_MAX_SPEED, ball.vy))

    def get_state(self, side: str) -> dict:
        own, opp = (self.left, self.right) if side == "left" else (self.right, self.left)
        own_score, opp_score = (
            (self.left_score, self.right_score) if side == "left" else (self.right_score, self.left_score)
        )
        return {
            "side": side,
            "court_width": WIDTH,
            "court_height": HEIGHT,
            "own_x": own.x,
            "own_y": own.y,
            "own_height": own.height,
            "opp_x": opp.x,
            "opp_y": opp.y,
            "opp_height": opp.height,
            "ball_x": self.ball.x,
            "ball_y": self.ball.y,
            "ball_vx": self.ball.vx,
            "ball_vy": self.ball.vy,
            "own_score": own_score,
            "opp_score": opp_score,
        }
