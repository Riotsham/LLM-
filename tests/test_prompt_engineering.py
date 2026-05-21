import unittest

from llm.prompt_engineering import decide_action, generate_llm_prompt


class DecisionEngineTests(unittest.TestCase):
    def test_crisis_phrase_overrides_other_signals(self):
        result = decide_action("joy", "Sometimes I want to die", 0.95)
        self.assertEqual(result["mode"], "crisis")
        self.assertIn("safety-focused", result["prompt"])

    def test_support_mode_for_negative_emotion(self):
        result = decide_action("fear", "I feel very overwhelmed today", 0.8)
        self.assertEqual(result["mode"], "support")
        self.assertIn("empathetic", result["prompt"])

    def test_normal_mode_for_positive_emotion(self):
        result = decide_action("joy", "Today was great", 0.9)
        self.assertEqual(result["mode"], "normal")
        self.assertIn("friendly", result["prompt"])

    def test_uncertain_mode_for_low_confidence(self):
        result = decide_action("fear", "Not sure how I feel", 0.4)
        self.assertEqual(result["mode"], "uncertain")
        self.assertIn("uncertain", result["prompt"])

    def test_uncertain_mode_for_missing_label(self):
        result = decide_action("", "I do not know", 0.95)
        self.assertEqual(result["mode"], "uncertain")

    def test_generate_prompt_includes_user_text(self):
        prompt = generate_llm_prompt("support", "I am anxious")
        self.assertIn("I am anxious", prompt)


if __name__ == "__main__":
    unittest.main()
