class Prompt:
    def __init__(self, thinking_style, task):
        self.thinking_style = thinking_style
        self.task = task

    def zero_shot_prompt(self):
        return f"{self.thinking_style} thought process required. The task at hand: {self.task}."

    def one_shot_prompt(self, example):
        return f"Using {self.thinking_style} thinking. Here's a task: {self.task}. See this example: {example}."

    def few_shot_prompt(self, examples, n=3):
        n = min(n, len(examples))  # Ensure n doesn't exceed available examples
        examples_text = ' '.join(examples[:n])
        return f"Approach this task with {self.thinking_style} thinking: {self.task}. Examples: {examples_text}"