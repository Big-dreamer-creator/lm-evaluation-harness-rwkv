# Adapted from ModelScope/EvalScope, Apache-2.0; see ../NOTICE.md.
from __future__ import annotations


FALLBACK_REFERENCES = {
    34377376: [
        '<!DOCTYPE html>\n<html lang="en">\n<head>\n    <meta charset="UTF-8">\n    <meta name="viewport" content="width=device-width, initial-scale=1.0">\n    <title>Random Joke Generator</title>\n    <style>\n        body { font-family: Arial, sans-serif; text-align: center; background-color: #f0f0f0; }\n        button { font-size: 20px; padding: 10px 20px; cursor: pointer; }\n    </style>\n    <script>\n        function showRandomJoke() {\n            const jokes = [\n                "Why don\'t scientists trust atoms? Because they make up everything!",\n                "Why did the chicken go to the seance? To get to the other side.",\n                "Why don\'t some couples go to the gym? Because some relationships don\'t work out.",\n                "Did you hear about the mathematician who\'s afraid of negative numbers? He\'ll stop at nothing to avoid them!"\n            ];\n            document.getElementById("jokeDisplay").innerHTML = jokes[Math.floor(Math.random() * jokes.length)];\n        }\n    </script>\n</head>\n<body>\n    <h1>Random Joke Generator</h1>\n    <button onclick="showRandomJoke()">Show me a joke!</button>\n    <p id="jokeDisplay"></p>\n</body>\n</html>',
        "Add `#jokeDisplay { color: red; }` to the CSS in the style block to display jokes in red.",
    ]
}


def first_turn_prompt(question: str, answer: str, reference: str) -> tuple[str, str]:
    reference_section = ""
    instruction = "Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response."
    if reference:
        instruction = "Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. Your evaluation should consider correctness and helpfulness. You will be given a reference answer and the assistant's answer. Begin by comparing the assistant's answer with the reference answer and identify any mistakes in the JSON explanation field."
        reference_section = f"\n\n[The Start of Reference Answer]\n{reference}\n[The End of Reference Answer]"
    prompt = f"[Instruction]\n{instruction}\n\n[Question]\n{question}{reference_section}\n\n[The Start of Assistant's Answer]\n{answer}\n[The End of Assistant's Answer]"
    return (prompt, "You are a helpful assistant.")


def second_turn_prompt(
    turns: list[str], answers: list[str], references: list[str], use_reference: bool
) -> tuple[str, str]:
    reference_section = ""
    system_prompt = "Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response. You should focus on the assistant's answer to the second user question. Provide a short explanation in the JSON explanation field. Be as objective as possible."
    if use_reference:
        if len(references) != 2:
            raise ValueError(
                "MT-Bench reference-guided judging requires two reference answers."
            )
        system_prompt = "Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question. Your evaluation should consider correctness and helpfulness. You will be given a reference answer and the assistant's answer. You should focus on the assistant's answer to the second question. Begin by comparing the assistant's answer with the reference answer and identify any mistakes in the JSON explanation field."
        reference_section = f"<|The Start of Reference Answer|>\n\n### User:\n{turns[0]}\n\n### Reference answer:\n{references[0]}\n\n### User:\n{turns[1]}\n\n### Reference answer:\n{references[1]}\n\n<|The End of Reference Answer|>\n\n\n"
    prompt = f"{reference_section}<|The Start of Assistant's Conversation with User|>\n\n### User:\n{turns[0]}\n\n### Assistant:\n{answers[0]}\n\n### User:\n{turns[1]}\n\n### Assistant:\n{answers[1]}\n\n<|The End of Assistant's Conversation with User|>"
    return (prompt, system_prompt)
