"""Optional pygame renderer. Only imported when --render pygame is requested."""

from __future__ import annotations

from .game import BALL_RADIUS, HEIGHT, PongGame, WIDTH

try:
    import pygame
except ImportError as exc:  # pragma: no cover - exercised only without pygame installed
    raise ImportError(
        "pygame is required for --render pygame. Install it with `pip install pygame`."
    ) from exc


class PygameRenderer:
    def __init__(self) -> None:
        pygame.init()
        self.screen = pygame.display.set_mode((int(WIDTH), int(HEIGHT)))
        pygame.display.set_caption("ping-pong-llm")
        self.font = pygame.font.SysFont(None, 36)

    def render(self, game: PongGame) -> bool:
        """Draw one frame. Returns False if the user closed the window."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

        self.screen.fill((10, 10, 10))
        for paddle in (game.left, game.right):
            half = paddle.height / 2
            rect = pygame.Rect(int(paddle.x) - 5, int(paddle.y - half), 10, int(paddle.height))
            pygame.draw.rect(self.screen, (240, 240, 240), rect)

        pygame.draw.circle(self.screen, (240, 240, 240), (int(game.ball.x), int(game.ball.y)), int(BALL_RADIUS))

        score_surf = self.font.render(f"{game.left_score}   {game.right_score}", True, (240, 240, 240))
        self.screen.blit(score_surf, (WIDTH / 2 - score_surf.get_width() / 2, 10))

        pygame.display.flip()
        return True

    def close(self) -> None:
        pygame.quit()
