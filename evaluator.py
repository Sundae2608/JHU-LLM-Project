class Evaluator:
    def __init__(self, train_dataset):
        self.train_dataset = train_dataset

    def evaluate(self, model, prompts, test_dataset):
        scores = []
        for prompt in prompts:
            score = 0
            for _ in range(10):  # Test each prompt 10 times
                for test in test_dataset['test']:
                    question = test['question']
                    correct_answer = test['answer']
                    # Randomly choose evaluation mode
                    mode = random.choice(['zero', 'one', 'few'])
                    if mode == 'zero':
                        prompt_text = prompt.zero_shot_prompt()
                    elif mode == 'one':
                        example = random.choice(self.train_dataset['train'])['example']
                        prompt_text = prompt.one_shot_prompt(example)
                    else:  # few
                        examples = [item['example'] for item in random.sample(self.train_dataset['train'], k=3)]
                        prompt_text = prompt.few_shot_prompt(examples)

                    # Generate the answer using the model
                    generated_answer = model.complete_text(f"{prompt_text} {question}", max_length=50)

                    # Check if the generated answer matches the correct answer
                    if correct_answer.lower() in generated_answer.lower():
                        score += 1
            scores.append(score)  # Append the total score for the current prompt
        return scores