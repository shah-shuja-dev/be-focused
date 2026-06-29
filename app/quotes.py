"""quotes.py -- Morgan Freeman motivational quote bank."""

import random

QUOTES = {
    "motivational": [
        "Get back to work. The only thing standing between you and your goal is the story you keep telling yourself.",
        "Whatever it is you are waiting for, it is not coming. Get back to that screen and make it happen.",
        "Was that break worth it? Every second counts. Eyes on the screen.",
        "You want to be great? Then act like it. Get back to work.",
        "Challenges make life interesting. Overcoming them makes life meaningful. Now go overcome something.",
        "The path to success is to take massive, determined action. Starting right now.",
        "Do not sit still. You are wasting time, and time is the most valuable thing you have.",
        "I have learned that it does not matter what happens to you, what matters is what you do about it. Do something.",
        "Get up. Get moving. Whatever you were just doing, it was not more important than your goals.",
    ],
    "stern": [
        "Hey. I see you. Get back to work.",
        "That was thirty seconds of your life you are not getting back. Stop it.",
        "Put it down. Now.",
        "I am watching. And I am not impressed.",
        "Was that more important than your goals? I doubt it.",
    ],
    "funny": [
        "In a world where people actually finish their work, you are not in it. Yet.",
        "Even Red in Shawshank had more focus than you right now.",
        "This is your conscience speaking. Or Morgan Freeman. Honestly at this point same thing.",
        "The penguins in March of the Penguins never stopped walking. Just saying.",
        "Did you just do what I think you did? I am Morgan Freeman and I am judging you.",
    ],
}


def get_quote(style: str = "motivational") -> str:
    return random.choice(QUOTES.get(style, QUOTES["motivational"]))