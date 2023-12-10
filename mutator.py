from enum import Enum
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

class MutationType(Enum):
    PROMPT_MUTATE = 1
    CROSSOVER = 2

class PromptMutator:
    def __init__(self, model: Model, max_num_examples, num_examples_mutate_prob, curr_gene_id):
        self.model = model
        self.max_num_examples = max_num_examples
        self.num_examples_mutate_prob = num_examples_mutate_prob
        self.curr_gene_id = curr_gene_id
    
    def next_gene_id(self):
        return_gene = self.curr_gene_id
        self.curr_gene_id += 1
        return return_gene
        
    def _mutate_prompt1(self, prompt: Prompt, mutator):
        mutate_prompt = f"{mutator}: \"{prompt.task}\".\nRewritten instructions:\n  1."
        generated_text = self.model.generate(mutate_prompt)
        variation = pick_random_variation("1. " + generated_text)
        if random.uniform(0, 1) < self.num_examples_mutate_prob:
            mutated_num_examples = random.randint(0, self.max_num_examples)
        else:
            mutated_num_examples = prompt.num_examples
        mutation_trace = (
            MutationType.PROMPT_MUTATE, prompt, mutated_num_examples
        )
        return Prompt(variation.strip(), prompt.system_instruction, prompt.thinking_style, mutated_num_examples, self.next_gene_id(), mutation_trace)

    def random_mutate(self, prompt: Prompt, mutator):
        mutation_methods = [self._mutate_prompt1]
        selected_method = random.choice(mutation_methods)
        return selected_method(prompt, mutator)
    
    def crossover(self, prompt1: Prompt, prompt2: Prompt):
        '''
        In a crossover scheme, basically we pick a random part from either gene
        '''
        gene1 = prompt1.gene()
        gene2 = prompt2.gene()
        
        offspring1 = list(gene1)
        offspring2 = list(gene2)
        
        def same_as_parent(indices):
            return len(set(indices)) <= 1 
        
        while True:
            new_gene_indices1 = []
            new_gene_indices2 = []
            for i in range(len(parent1)):
                if random.random() < 0.5:  # 50% probability for each element
                    new_gene_indices1.append(0)
                    new_gene_indices2.append(1)
                else:
                    new_gene_indices1.append(1)
                    new_gene_indices2.append(0)
            if not same_as_parent(new_gene_indices1):
                break
                
        offspring1 = []
        offspring2 = []
        for i in range(len(new_gene_indices1)):
            selected = new_gene_indices1[i]
            if selected == 0:
                offspring1.append(gene1[i])
                offspring2.append(gene2[i])
            else:
                offspring1.append(gene2[i])
                offspring2.append(gene1[i])
                
        mutation_trace_1 = (MutationType.CROSSOVER, prompt1, prompt2, tuple(new_gene_indices1))
        mutation_trace_2 = (MutationType.CROSSOVER, prompt1, prompt2, tuple(new_gene_indices2))
        return Prompt(offspring1[0], offspring1[1], offspring1[2], offspring1[3], self.next_gene_id(), mutation_trace_1), Prompt(offspring2[0], offspring2[1], offspring2[2], offspring2[3], self.next_gene_id(), mutation_trace_2)
        
        