import random
import time


def get_next_animation_state(current_state, assistant_state):
    """
    current_state: last expression
    assistant_state: "neutral", "happy", "sad", "talking"
    Use random blinking every few seconds
    """
    # blinking logic
    if random.random() < 0.03:
        return "blink1"
    if assistant_state == "talking":
        return random.choice(["talk1", "talk2", "talk3"])
    return assistant_state
