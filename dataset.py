from datasets import load_dataset
import numpy as np
import re


class Dataset:

    def __init__(self, name='gsm8k') -> None:

        self.dataset_name = name
        self.dataset = load_dataset(name, 'main')

    

    def pick_random_examples(self, max_num_examples:int):

        examples = []

        indexes = np.random.randint(len(self.dataset['train']), size=max_num_examples)

        for index in indexes:

            if self.dataset_name == 'gsm8k':

                examples.append('Q: ' + self.dataset['train'][int(index)]['question'] + '\nA: ' + self.dataset['train'][int(index)]['answer'])


        return examples


    def pick_random_problems(self, num_problems:int):

        problems, answers = [], []

        indexes = np.random.randint(len(self.dataset['train']), size=num_problems)

        for index in indexes:

            if self.dataset_name == 'gsm8k':

                problems.append('Q: ' + self.dataset['train'][int(index)]['question'] + '\nA: ')
                answers.append(self.dataset['train'][int(index)]['answer'])

        return problems, answers
    

    def evaluate(self, real_answer, predicted_answer):

        if self.dataset_name == 'gsm8k':

            if last_number(real_answer) == last_number(predicted_answer):
                return 1
            else:
                return 0


        


def last_number(s):
    numbers = re.findall(r'\d+', s)
    return int(numbers[-1]) if numbers else None