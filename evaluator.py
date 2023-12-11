import pickle
import random

class EvaluatorParams:
    def __init__(self, gene_storage_path, print_text_completion=True, store_text_completion=True, print_evaluation_steps=0):
        
        # Gene dictionary storage
        self.gene_storage_path = gene_storage_path
        
        # Print and storage
        self.print_text_completion = print_text_completion
        self.store_text_completion = store_text_completion
        self.print_evaluation_steps = print_evaluation_steps

class Evaluator:
    def __init__(self, params):
        self.params = params
        self.gene_dictionary = {}
        self.gene_answers = {}
        pass

    def evaluate(self, model, task, prompts, num_eval, test_indices=None):
        # Score of each prompt
        scores = []
        
        # Sample some test first
        if test_indices is None:
            test_indices, test_samples = task.test_samples(num_eval)
        else:
            test_samples = [task.test()[i] for i in test_indices]
        
        print(f"Evaluating test samples: {str(test_indices)}")
        
        for i, prompt in enumerate(prompts):
            print(f"Evaluating prompt {i}: {prompt}")
            
            # Store the prompt
            if prompt.gene_id not in self.gene_dictionary:
                self.gene_answers[prompt.gene_id] = set()
                self.gene_dictionary[prompt.gene_id] = prompt.gene()
            
            # Sample 10 test and test time
            score = 0
            for i in range(num_eval):
                # Get the sample
                test_index = test_indices[i]
                test_sample = test_samples[i]
                question = test_sample['question']
                correct_answer = test_sample['answer']
                
                examples = task.train()
                prompt_text = prompt.get_prompt(question, examples)

                # Generate the answer using the model
                generated_answer = model.generate(prompt_text)
                
                # Store the answer
                self.gene_answers[prompt.gene_id].add((question, generated_answer))

                # Check if the generated answer matches the correct answer
                if self.params.print_text_completion:
                    print("---")
                    print(correct_answer)
                    print(generated_answer)
                if task.evaluator_func(correct_answer, generated_answer):
                    correct = True
                    score = 1
                else:
                    correct = False
                    score = 0
                prompt.add_evaluation(test_index, correct, score)
                
                # Print current score so far
                if self.params.print_evaluation_steps > 0 and (i + 1) % self.params.print_evaluation_steps == 0:
                    print(f"Num evals: {prompt.get_num_evals()} - Accuracy: {prompt.get_accuracy() * 100:.2f}%")
                
            scores.append(prompt.get_accuracy())  # Append the total score for the current prompt
            
            # Store the genes
            filename = f"{self.params.gene_storage_path}/genes.pkl"
            data = {
                "gene_dict": self.gene_dictionary,
                "gene_answers": self.gene_answers
            }
            with open(filename, "wb") as file:
                pickle.dump(data , file)
            
            
        return scores