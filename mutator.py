from model import Model
from prompt import Prompt

import random

class PromptMutator:
    def __init__(self, model: Model):
        self.model = model
        
    def _mutate_prompt1(self, prompt: Prompt, mutator, thinking_style):
        mutate_prompt = f"{mutator}: {prompt.task}.\Rewritten instructions:\n  1."
        generated_text = self.model.generate(mutate_prompt, stop=["2.", "2)"])
        return Prompt(generated_text.strip())
        
    def _mutate_prompt2(self, prompt: Prompt, mutator, thinking_style):
        mutate_prompt = f"{thinking_style} {mutator}: {prompt.task}.\Rewritten instructions:\n  1. "
        generated_text = self.model.generate(mutate_prompt, stop=["2.", "2)"])
        return Prompt(generated_text.strip())
    
    def random_mutate(self, prompt: Prompt, mutator, thinking_style):
        mutation_methods = [self._mutate_prompt1, self._mutate_prompt2]
        selected_method = random.choice(mutation_methods)
        return selected_method(prompt, mutator, thinking_style)