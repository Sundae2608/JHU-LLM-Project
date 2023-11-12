import openai
import os
from transformers import pipeline
from ctransformers import AutoModelForCausalLM

class Model:
    def __init__(self, provider='openai', model_name='text-davinci-003'):
        """
        Initialize the Language Model Manager (LLM) class.

        Args:
            provider (str): The language model provider ('openai' or 'huggingface').
            model_name (str): The specific model to use (e.g., 'text-davinci-003' for OpenAI or a model name for Hugging Face).

        Raises:
            ValueError: If the provider is 'openai' but no API key is provided.
            NotImplementedError: If an unsupported provider is specified.
        """
        # Input function
        self.provider = provider
        self.model_name = model_name
        self.api_key = os.getenv("OPENAI_KEY")
        self.pipeline = None
        self.ctransformer_llm = None

        if self.provider == 'openai':
            # Initialize OpenAI API with the provided API key
            if not self.api_key:
                raise ValueError("API key is required for OpenAI provider")
            openai.api_key = os.getenv("OPENAI_KEY")
        elif self.provider == 'huggingface':
            # Initialize Hugging Face model pipeline
            self.pipeline = pipeline('text-generation', model=self.model_name)
        elif self.provider == 'quantized_llama':
            # Initialize C-transformer model
            self.ctransformer_llm = AutoModelForCausalLM.from_pretrained(self.model_name, model_type="llama", gpu_layers=50)

    def generate(self, prompt, **kwargs):
        """
        Generate text based on the specified provider and model.

        Args:
            prompt (str): The input text prompt to generate text from.
            **kwargs: Additional keyword arguments for generation.

        Returns:
            str: The generated text.

        Raises:
            ValueError: If the model pipeline is not initialized for Hugging Face.
            NotImplementedError: If an unsupported provider is specified.
        """
        if self.provider == 'openai':
            # Generate text using OpenAI's GPT-3 model
            response = openai.Completion.create(engine=self.model_name, prompt=prompt, **kwargs)
            return response.choices[0].text.strip()
        elif self.provider == 'huggingface':
            # Generate text using the Hugging Face model pipeline
            result = self.pipeline(prompt, **kwargs)
            return result[0]['generated_text']
        elif self.provider == 'quantized_llama':
            # Generate text using quantized version of the llama model
            return self.ctransformer_llm(prompt, **kwargs)
        else:
            raise NotImplementedError("The specified provider is not supported")
            
            
# if self.provider == 'openai':

#     response = openai.ChatCompletion.create(
#     model="gpt-3.5-turbo",
#     messages=[
#                 {"role": "system", "content": prompt['sys_instruction']},
#                 {"role": "user", "content": prompt['instruction'] + '/n' + prompt['examples'] + '\n' + prompt['problem']},

#             ]
#             )
#     return response['choices'][0]['message']['content']