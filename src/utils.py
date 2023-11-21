def load_prompts(file_path):
    with open(file_path, 'r') as file:
        prompts = file.readlines()
    return [prompt.strip() for prompt in prompts]