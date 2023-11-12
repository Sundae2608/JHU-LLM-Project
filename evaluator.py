import random

class Evaluator:
    def __init__(self):
        pass

    def evaluate(self, model, task, mode, prompts, num_eval):
        # Score of each prompt
        scores = []
        for i, prompt in enumerate(prompts):
            print(f"Evaluating prompt {i}: {prompt}")
            
            # Sample 10 test and test time
            score = 0
            test_samples = task.train_samples(num_eval)
            for i in range(num_eval):  # Test each prompt 10 times
                # Get the sample
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
                generated_answer = model.generate(prompt_text, max_tokens=500)

                # Check if the generated answer matches the correct answer
                print("---")
                print(correct_answer)
                print(generated_answer)
                if task.evaluator_func(correct_answer, generated_answer):
                    score += 1
            scores.append(score)  # Append the total score for the current prompt
        return scores