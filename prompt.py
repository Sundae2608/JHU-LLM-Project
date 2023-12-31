import random

def remove_leading_spaces_and_lines(input_string):
    # Split the input string into lines
    lines = input_string.split('\n')
    
    # Remove leading spaces and tabs from each line
    cleaned_lines = [line.lstrip(' \t') for line in lines]
    
    # Go the the first non empty line
    for i in range(len(cleaned_lines)):
        if cleaned_lines[i] != '':
            break
    cleaned_lines = cleaned_lines[i:]
    
    # Join the cleaned lines back into a string
    output_string = '\n'.join(cleaned_lines)
    
    return output_string

class Prompt:
    def __init__(self, task, system_instruction, thinking_style, num_examples, gene_id, previous_mutation):
        # System instruction is similar to thinking style
        self.task = task
        self.system_instruction = system_instruction
        self.thinking_style = thinking_style
        self.num_examples = num_examples
        self.gene_id = gene_id
        self.mutation_trace = previous_mutation
        
        # Evaluation storage
        self.evaluated_indices = set()
        self.score = 0
        self.num_correct = 0
        
    def copy(self):
        """
        Return a new prompt but essentially reset the scoring of it
        """
        return Prompt(self.task, self.system_instruction, self.thinking_style, self.num_examples, self.gene_id, self.mutation_trace)
        
    def add_evaluation(self, index, correct, score):
        if index in self.evaluated_indices:
            return
        self.evaluated_indices.add(index)
        self.num_correct += 1 if correct else 0
        self.score += score
        
    def get_num_evals(self):
        return len(self.evaluated_indices)
        
    def get_accuracy(self):
        return self.num_correct / len(self.evaluated_indices)

    def _zero_shot_prompt(self, question):
        if self.thinking_style == '':
            user_prompt = f"{self.task}"
        else:
            user_prompt = f"{self.thinking_style}, {self.task}" 
        prompt = f"""
        SYSTEM: {self.system_instruction}
        USER: {user_prompt}
        QUESTION: {question}
        ANSWER: 
        """
        return remove_leading_spaces_and_lines(prompt)

    def _one_shot_prompt(self, question, train_example):
        example_q = train_example["question"]
        example_a = train_example["answer"]
        
        if self.thinking_style == '':
            user_prompt = f"{self.task}"
        else:
            user_prompt = f"{self.thinking_style}, {self.task}" 
        prompt = f"""
        SYSTEM: {self.system_instruction}
        USER: {user_prompt}
        QUESTION: {example_q}
        ANSWER: {example_a}
        QUESTION: {question}
        ANSWER: 
        """
        return remove_leading_spaces_and_lines(prompt)

    def _few_shot_prompt(self, question, train_examples, n=3):
        n = min(n, len(train_examples))  # Ensure n doesn't exceed available examples
        sampled_train_indices = random.sample(range(len(train_examples)), n)
        sampled_train_examples = [(train_examples[i]["question"], train_examples[i]["answer"]) for i in sampled_train_indices]
        example_text = ''.join([
            f"""
            QUESTION: {q}
            ANSWER: {a}""" for q, a in sampled_train_examples])
        
        if self.thinking_style == '':
            user_prompt = f"{self.task}"
        else:
            user_prompt = f"{self.thinking_style}, {self.task}" 
        prompt = f"""SYSTEM: {self.system_instruction}
            USER: {user_prompt}. {example_text}
            QUESTION: {question}
            ANSWER: """
        return remove_leading_spaces_and_lines(prompt)
    
    def get_prompt(self, question, train_examples):
        if self.num_examples == 0:
            return self._zero_shot_prompt(question)
        elif self.num_examples >= 1:
            return self._few_shot_prompt(question, train_examples, self.num_examples)
    
    def gene(self):
        return (
            self.task,
            self.system_instruction,
            self.thinking_style,
            self.num_examples,
        )
    
    def __repr__(self):
        return f"({self.task[:20]:20} - {self.system_instruction[:20]:20} - {self.thinking_style[:20]:20} - {self.num_examples})"