"""Cheap terminal renderer so the demo runs with zero extra dependencies."""

from __future__ import annotations

from .game import HEIGHT, PongGame, WIDTH

COLS = 60
ROWS = 20


def render(game: PongGame) -> str:
    grid = [[" "] * COLS for _ in range(ROWS)]

    def put(x: float, y: float, ch: str) -> None:
        col = min(COLS - 1, max(0, int(x / WIDTH * COLS)))
        row = min(ROWS - 1, max(0, int(y / HEIGHT * ROWS)))
        grid[row][col] = ch

    for paddle, ch in ((game.left, "|"), (game.right, "|")):
        half = paddle.height / 2
        top, bottom = paddle.y - half, paddle.y + half
        col = min(COLS - 1, max(0, int(paddle.x / WIDTH * COLS)))
        row_top = min(ROWS - 1, max(0, int(top / HEIGHT * ROWS)))
        row_bottom = min(ROWS - 1, max(0, int(bottom / HEIGHT * ROWS)))
        for row in range(row_top, row_bottom + 1):
            grid[row][col] = ch

    put(game.ball.x, game.ball.y, "o")

    border = "+" + "-" * COLS + "+"
    lines = [f" Score: {game.left_score} - {game.right_score} ".center(COLS + 2, "="), border]
    lines += ["|" + "".join(row) + "|" for row in grid]
    lines.append(border)
    return "\n".join(lines)


def clear_screen() -> None:
    print("\033[H\033[J", end="")
