class Prompt:
    def __init__(self, thinking_style, task, examples):
        self.thinking_style = thinking_style
        self.task = task
        self.examples = examples  # This should be a list of dictionaries or tuples

    def format_example(self, example):
        # Assuming each example is a dictionary with 'question' and 'answer' keys
        return f"Q: {example['question']} A: {example['answer']}"

    def __str__(self):
        # Join all examples in a formatted string
        formatted_examples = '; '.join([self.format_example(example) for example in self.examples])
        return f"{self.thinking_style}: {self.task}. Examples: {formatted_examples}"

    def get_prompt(self):
        # This method explicitly concatenates and returns the prompt string
        return str(self)

    def mutate_prompt(self, mutation_type, *args):
        if mutation_type == 'add_example':
            # Expecting args to contain dictionaries with 'question' and 'answer'
            self.examples.extend(args)
        elif mutation_type == 'change_thinking_style':
            # Expecting args[0] to be the new thinking style
            self.thinking_style = args[0]
        elif mutation_type == 'change_task':
            # Expecting args[0] to be the new task
            self.task = args[0]
        else:
            raise ValueError("Unknown mutation type")

# Example usage:
thinking_style = "Analytical"
task = "Solve the following math problems"
examples = [
    {'question': 'What is the square root of 16?', 'answer': '4'},
    {'question': 'If you have 5 apples and you eat 2, how many do you have left?', 'answer': '3'}
]

prompt = Prompt(thinking_style, task, examples)

# Mutate the prompt
prompt.mutate_prompt('add_example', {'question': 'What is 10 + 4?', 'answer': '14'})
prompt.mutate_prompt('change_thinking_style', 'Creative')
prompt.mutate_prompt('change_task', 'Answer the following questions creatively')

print(prompt.get_prompt())