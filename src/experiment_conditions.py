from src.dataset import Dataset
from src.llm import LLM

class Experiment_conditions:

    def __init__(self, dataset:Dataset, prompt_mutators:list[str], sys_prompts:list[str], 
                 instructions:list[str], llm:LLM, max_num_examples:int) -> None:
        
        self.dataset = dataset
        self.prompt_mutators = prompt_mutators
        self.sys_prompts = sys_prompts
        self. instructions = instructions
        self.llm = llm
        self.max_num_examples = max_num_examples

        self.examples = self.dataset.pick_random_examples(max_num_examples)


