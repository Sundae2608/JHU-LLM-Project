import random

class Prompt:
    def __init__(self, task):
        self.task = task

    def zero_shot_prompt(self, question):
        return f"{self.task}.\n\nQ: {question}\nA: "

    def one_shot_prompt(self, question, train_example):
        example_q = train_example["question"]
        example_a = train_example["answer"]
        return f"{self.task}.\n\nQ: {example_q}\nA: {example_a}\n\nQ: {question}\nA: "

    def few_shot_prompt(self, question, train_examples, n=3):
        n = min(n, len(train_examples))  # Ensure n doesn't exceed available examples
        sampled_train_indices = random.sample(range(len(train_examples)), n)
        sampled_train_examples = [(train_examples[i]["question"], train_examples[i]["answer"]) for i in sampled_train_indices]
        example_text = ''.join([f"\n\nQ: {q}\nA: {a}" for q, a in sampled_train_examples])
        return f"{self.task}. {example_text}\n\nQ: {question}\nA:"
    
    def __repr__(self):
        return self.task