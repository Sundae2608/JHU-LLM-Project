from datasets import load_dataset
import random

class Task:
    def __init__(self, dataset, initial_prompt, evaluator_func):
        self.dataset = dataset
        self.initial_prompt = initial_prompt
        self.evaluator_func = evaluator_func

    def train(self):
        # Return the training subset of the dataset
        return self.dataset['train']

    def test(self):
        # Return the test subset of the dataset
        return self.dataset['test']
    
    def train_samples(self, n):
        sampled_indices = random.sample(range(n), n)
        samples = [self.dataset['train'][i] for i in sampled_indices]
        return samples
    
    def test_samples(self, n):
        sampled_indices = random.sample(range(n), n)
        samples = [self.dataset['test'][i] for i in sampled_indices]
        return samples
        
    def train_sample(self):
        return random.choice(self.dataset['train'])
    
    def test_sample(self):
        return random.choice(self.dataset['test'])