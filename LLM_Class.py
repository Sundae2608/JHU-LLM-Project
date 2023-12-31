from transformers import pipeline
import openai  # for making OpenAI API requests


class LLM:
    def __init__(self, provider='openai', model='text-davinci-003', api_key=None):
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.pipeline = None
        if self.provider == 'openai':
            if not self.api_key:
                raise ValueError("API key is required for OpenAI provider")
            
            openai.api_key = self.api_key
        elif self.provider == 'huggingface':
            self.pipeline = pipeline('text-generation', model=self.model)

    def generate(self, prompt):
        if self.provider == 'openai':

            response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                        {"role": "system", "content": prompt['sys_instruction']},
                        {"role": "user", "content": prompt['instruction'] + '/n' + prompt['examples'] + '\n' + prompt['problem']},

                    ]
                    )
            return response['choices'][0]['message']['content']

        else:
            raise NotImplementedError("The specified provider is not supported")

# # Example usage:
# # For OpenAI (ensure you have the OpenAI API key set)
# llm_openai = LLM(provider='openai', model='text-davinci-003', api_key='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')
# print(llm_openai.generate("Translate the following English text to French: 'Hello, world!'"))

# # For Hugging Face (this assumes you have the necessary model locally or it's available to download)
# llm_huggingface = LLM(provider='huggingface', model='gpt2')
# print(llm_huggingface.generate("What is the capital of France?"))