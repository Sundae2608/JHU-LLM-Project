import random

class Evaluator:
    def __init__(self, print_text_completion=False):
        self.print_text_completion = print_text_completion
        pass

    def evaluate(self, model, task, mode, prompts, num_eval):
        # Score of each prompt
        scores = []
        
        # Sample some test first
        test_indices, test_samples = task.test_samples(num_eval)
        print(f"Evaluating test samples: {str(test_indices)}")
        for i, prompt in enumerate(prompts):
            print(f"Evaluating prompt {i}: {prompt}")
            
            # Sample 10 test and test time
            score = 0
            for i in range(num_eval):
                # Get the sample
                test_index = test_indices[i]
                test_sample = test_samples[i]
                question = test_sample['question']
                correct_answer = test_sample['answer']
                
                # Create prompt text based on evaluation mode
                if mode == 'zero':
                    prompt_text = prompt.zero_shot_prompt(question)
                elif mode == 'one':
                    example = task.train_sample()
                    prompt_text = prompt.one_shot_prompt(question, example)
                else:  # few
                    examples = task.train_samples(3)
                    prompt_text = prompt.few_shot_prompt(question. examples)

                # Generate the answer using the model
                generated_answer = model.generate(prompt_text)

                # Check if the generated answer matches the correct answer
                if self.print_text_completion:
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
            scores.append(prompt.get_accuracy())  # Append the total score for the current prompt
        return scores