import unittest

from pingpong_llm.controllers import HeuristicController, build_prompt, is_cloud_model, parse_action
from pingpong_llm.game import HEIGHT, WIDTH, PongGame
from pingpong_llm.main import PRESETS


class ParseActionTest(unittest.TestCase):
    def test_plain_answers(self):
        self.assertEqual(parse_action("UP"), "UP")
        self.assertEqual(parse_action("  down.\n"), "DOWN")
        self.assertEqual(parse_action("Stay"), "STAY")

    def test_ignores_inline_reasoning(self):
        self.assertEqual(parse_action("<think>go UP or DOWN?</think>\nSTAY"), "STAY")

    def test_unterminated_reasoning_falls_back_to_stay(self):
        self.assertEqual(parse_action("<think>probably UP"), "STAY")

    def test_does_not_match_inside_words(self):
        self.assertEqual(parse_action("upward"), "STAY")

    def test_garbage_falls_back_to_stay(self):
        self.assertEqual(parse_action(""), "STAY")
        self.assertEqual(parse_action(None), "STAY")


class CloudModelTest(unittest.TestCase):
    def test_detects_cloud_tags(self):
        self.assertTrue(is_cloud_model("nemotron-3-nano:30b-cloud"))
        self.assertTrue(is_cloud_model("glm-4.6:cloud"))
        self.assertFalse(is_cloud_model("nemotron-3-nano:4b"))
        self.assertFalse(is_cloud_model("llama3.2"))


class PromptTest(unittest.TestCase):
    def test_only_the_last_line_changes_between_moves(self):
        # Keeps Ollama's prefix cache useful: slow CPUs re-read only the changed tail.
        game = PongGame()
        before = build_prompt(game.get_state("left")).splitlines()
        for _ in range(10):
            game.step(1 / 30, "DOWN", "STAY")
        after = build_prompt(game.get_state("left")).splitlines()
        self.assertEqual(before[:-1], after[:-1])
        self.assertNotEqual(before[-1], after[-1])

    def test_live_state_line_is_short(self):
        game = PongGame()
        self.assertLess(len(build_prompt(game.get_state("right")).splitlines()[-1].split()), 12)


class PhysicsTest(unittest.TestCase):
    def test_paddle_stays_on_court(self):
        game = PongGame()
        for _ in range(200):
            game.step(1 / 30, "UP", "DOWN")
        self.assertEqual(game.left.y, game.left.height / 2)
        self.assertEqual(game.right.y, HEIGHT - game.right.height / 2)

    def test_ball_bounces_off_paddle(self):
        game = PongGame()
        game.ball.x, game.ball.y = game.left.x + 10, game.left.y
        game.ball.vx, game.ball.vy = -200, 0
        game.step(1 / 60, "STAY", "STAY")
        self.assertGreater(game.ball.vx, 0)

    def test_missed_ball_scores_for_opponent(self):
        game = PongGame()
        game.ball.x, game.ball.y = 1, 5
        game.ball.vx, game.ball.vy = -200, 0
        self.assertEqual(game.step(1 / 30, "STAY", "STAY"), "right")
        self.assertEqual((game.left_score, game.right_score), (0, 1))
        self.assertEqual((game.ball.x, game.ball.y), (WIDTH / 2, HEIGHT / 2))

    def test_presets_apply_to_paddles_and_ball(self):
        p = PRESETS["slow-model"]
        game = PongGame(
            paddle_speed=p["paddle_speed"],
            paddle_height=p["paddle_height"],
            ball_base_speed=p["ball_speed"],
            ball_speedup=p["ball_speedup"],
        )
        self.assertEqual(game.left.height, p["paddle_height"])
        self.assertEqual(game.right.speed, p["paddle_speed"])
        self.assertEqual(abs(game.ball.vx), p["ball_speed"])

    def test_heuristic_bots_sustain_rallies(self):
        game = PongGame()
        bot = HeuristicController()
        for _ in range(30 * 20):
            game.step(1 / 30, bot.get_action(game.get_state("left")), bot.get_action(game.get_state("right")))
        self.assertLess(game.left_score + game.right_score, 10)


if __name__ == "__main__":
    unittest.main()
