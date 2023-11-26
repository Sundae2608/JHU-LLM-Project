from model import Model
from prompt import Prompt

import random

def pick_random_variation(text):
    """
    Splits the input text into individual variations and picks one at random.
    Assumes that each variation is on a new line and starts with a numeric followed by a dot.

    Args:
    text (str): The input text containing variations.

    Returns:
    str: A randomly selected variation.
    """
    # Split the text into lines
    lines = text.split('\n')
    
    # Extract variations by removing the initial numeric and dot
    variations = [line.split('. ', 1)[1] if '. ' in line else line for line in lines if line.strip()]

    # Pick a random variation
    return random.choice(variations) if variations else ""

class PromptMutator:
    def __init__(self, model: Model):
        self.model = model
        
    def _mutate_prompt1(self, prompt: Prompt, mutator):
        mutate_prompt = f"{mutator}: \"{prompt.task}\".\nRewritten instructions:\n  1."
        generated_text = self.model.generate(mutate_prompt)
        variation = pick_random_variation("1. " + generated_text)
        return Prompt(variation.strip(), prompt.thinking_style, prompt.system_instruction)

    def random_mutate(self, prompt: Prompt, mutator):
        mutation_methods = [self._mutate_prompt1]
        selected_method = random.choice(mutation_methods)
        return selected_method(prompt, mutator)
    
    def crossover(self, prompt1: Prompt, prompt2: Prompt):
        '''
        In a crossover scheme
        '''