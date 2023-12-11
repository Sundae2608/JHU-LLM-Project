from enum import Enum
from model import Model
from prompt import Prompt
from constants import MUTATORS, THINKING_STYLES, SYSTEM_INSTRUCTIONS

import pickle
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
    
class PromptMutationParams:
    def __init__(self, max_num_examples=4, 
                 num_examples_mutate_prob=0.7,
                 thinking_styles=THINKING_STYLES, reselect_thinking_style_prob=0.1, mutate_thinking_style_prob=0.7,
                 mutators=MUTATORS, mutate_mutator_prob=0.7,
                 system_instructions=SYSTEM_INSTRUCTIONS, reselect_system_instruction_prob=0.1, mutate_system_instruction_prob=0.7):
        # Maximum examples
        self.max_num_examples = max_num_examples
        self.num_examples_mutate_prob = num_examples_mutate_prob
        
        # Assign thinking styles
        self.thinking_styles = thinking_styles
        self.reselect_thinking_style_prob = reselect_thinking_style_prob
        self.mutate_thinking_style_prob = mutate_thinking_style_prob
        
        # Assigning mutators
        self.mutators = mutators
        self.mutate_mutator_prob = mutate_mutator_prob
        
        # Assigning system instructions
        self.system_instructions = system_instructions
        self.reselect_system_instruction_prob = reselect_system_instruction_prob
        self.mutate_system_instruction_prob = mutate_system_instruction_prob

class PromptMutator:
    def __init__(self, model: Model, params=PromptMutationParams()):
        self.model = model
        
        # Maximum examples
        self.max_num_examples = params.max_num_examples
        self.num_examples_mutate_prob = params.num_examples_mutate_prob
        
        # Assign thinking styles
        self.thinking_style_id = 0
        self.thinking_styles = {}
        for thinking_style in params.thinking_styles:
            self._add_thinking_style(thinking_style)
            
        self.reselect_thinking_style_prob = params.reselect_thinking_style_prob
        self.mutate_thinking_style_prob = params.mutate_thinking_style_prob
        
        # Assigning mutators
        self.mutator_id = 0
        self.mutators = {}
        for mutator in params.mutators:
            self._add_mutator(mutator)
        self.mutate_mutator_prob = params.mutate_mutator_prob
        
        # Assigning system instructions
        self.system_instructions = {}
        self.system_instruction_id = 0
        for system_instruction in params.system_instructions:
            self._add_system_instruction(system_instruction)

        self.reselect_system_instruction_prob = params.reselect_system_instruction_prob
        self.mutate_system_instruction_prob = params.mutate_system_instruction_prob
        
        # Initialize gene ID to be 0
        self.curr_gene_id = 0
        
    def _add_thinking_style(self, thinking_style):
        if thinking_style not in self.thinking_styles:
            self.thinking_styles[thinking_style] = self.thinking_style_id
            self.thinking_style_id += 1
        
    def _add_mutator(self, mutator):
        if mutator not in self.mutators:
            self.mutators[mutator] = self.mutator_id
            self.mutator_id += 1
        
    def _add_system_instruction(self, system_instruction):
        if system_instruction not in self.system_instructions:
            self.system_instructions[system_instruction] = self.system_instruction_id
            self.system_instruction_id += 1
    
    def next_gene_id(self):
        return_gene = self.curr_gene_id
        self.curr_gene_id += 1
        return return_gene
    
    def _get_mutator(self):
        meta_mutator = random.choice(list(self.mutators.keys()))
        mutator = random.choice(list(self.mutators.keys()))
        thinking_style = random.choice(list(self.thinking_styles.keys()))
        if random.uniform(0, 1) < self.mutate_mutator_prob:
            mutate_prompt = random.choice([
                f"{meta_mutator}: \"<mutator>\".\nRewritten prompts:\n  1.",
                f"{thinking_style}, {meta_mutator}: \"<mutator>\".\nRewritten prompts:\n  1."])
            generated_text = self.model.generate(mutate_prompt.replace("<mutator>", mutator))
            variation = pick_random_variation("1. " + generated_text).strip()
            self._add_mutator(variation)
        return variation
    
    def _mutate_task(self, prompt: Prompt):
        mutator = random.choice(list(self.mutators.keys()))
        thinking_style = random.choice(list(self.thinking_styles.keys()))
        mutate_prompt = random.choice([
            f"{mutator}: \"<task>\".\nRewritten prompts:\n  1.",
            f"{thinking_style}, {mutator}: \"<task>\".\nRewritten prompts:\n  1."])
        generated_text = self.model.generate(mutate_prompt.replace("<task>", prompt.task))
        variation = pick_random_variation("1. " + generated_text).strip()
        return variation, mutate_prompt
    
    def _mutate_thinking_style(self, prompt: Prompt):
        mutator = random.choice(list(self.mutators.keys()))
        thinking_style = random.choice(list(self.thinking_styles.keys()))
        mutate_prompt = random.choice([
            f"{mutator}: \"<task>\".\nRewritten thinking styles:\n  1.",
            f"{thinking_style}, {mutator}: \"<task>\".\nRewritten thinking styles:\n  1."])
        generated_text = self.model.generate(mutate_prompt.replace("<thinking_style>", prompt.thinking_style))
        variation = pick_random_variation("1. " + generated_text).strip()
        self._add_thinking_style(variation)
        return variation, mutate_prompt
    
    def _mutate_num_examples(self):
        return random.choice(range(0, self.max_num_examples))
    
    def _mutate_system_instruction(self, prompt: Prompt):
        mutator = random.choice(list(self.mutators.keys()))
        thinking_style = random.choice(list(self.thinking_styles.keys()))
        mutate_prompt = random.choice([
            f"{mutator}: \"<system_instruction>\".\nRewritten system instructions:\n  1.",
            f"{thinking_style}, {mutator}: \"<system_instruction>\".\nRewritten system instructions:\n  1."])
        generated_text = self.model.generate(mutate_prompt.replace("<system_instruction>", prompt.system_instruction))
        variation = pick_random_variation("1. " + generated_text).strip()
        self._add_system_instruction(variation)
        return variation, mutate_prompt
    
    def save_ids(self, file_path):
        d = {}
        d["mutators"] = self.mutators
        d["thinking_styles"] = self.thinking_styles
        d["system_instructions"] = self.system_instructions
        # Pickle the dictionary and save it to a file
        with open(file_path, 'wb') as file:
            pickle.dump(d, file)

    def random_mutate(self, prompt: Prompt):
            
        # Mutation of tasks
        task, mutate_prompt = self._mutate_task(prompt)
            
        # Mutation of system instructions
        if random.uniform(0, 1) < self.mutate_system_instruction_prob:
            system_instruction, _ = self._mutate_system_instruction(prompt)
        elif random.uniform(0, 1) < self.reselect_system_instruction_prob:
            system_instruction = random.choice(THINKING_STYLES)
        else:
            system_instruction = prompt.system_instruction
            
        # Mutation of thinking style
        if random.uniform(0, 1) < self.mutate_thinking_style_prob:
            thinking_style, _ = self._mutate_thinking_style(prompt)
        elif random.uniform(0, 1) < self.reselect_thinking_style_prob:
            thinking_style = random.choice(THINKING_STYLES)
        else:
            thinking_style = prompt.thinking_style

        # Number of examples
        if random.uniform(0, 1) < self.num_examples_mutate_prob:
            num_examples = self._mutate_num_examples()
        else:
            num_examples = prompt.num_examples
        
        mutation_trace = (
            MutationType.PROMPT_MUTATE, mutate_prompt, prompt.task, system_instruction, thinking_style, num_examples,
        )
        return Prompt(task, system_instruction, thinking_style, num_examples, self.next_gene_id(), mutation_trace)
    
    def crossover(self, prompt1: Prompt, prompt2: Prompt):
        '''
        In a crossover scheme, basically we pick a random part from either gene
        '''
        gene1 = prompt1.gene()
        gene2 = prompt2.gene()
        
        def same_as_parent(indices):
            return len(set(indices)) <= 1 
        
        while True:
            new_gene_indices1 = []
            new_gene_indices2 = []
            for i in range(len(gene1)):
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
        return (
            Prompt(offspring1[0], offspring1[1], offspring1[2], offspring1[3], self.next_gene_id(), mutation_trace_1), 
            Prompt(offspring2[0], offspring2[1], offspring2[2], offspring2[3], self.next_gene_id(), mutation_trace_2))