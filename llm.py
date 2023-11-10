from transformers import pipeline
import openai
from transformers import pipeline

class LLM:
    def __init__(self, provider='openai', model='text-davinci-003', api_key=None):
        """
        Initialize the Language Model Manager (LLM) class.

        Args:
            provider (str): The language model provider ('openai' or 'huggingface').
            model (str): The specific model to use (e.g., 'text-davinci-003' for OpenAI or a model name for Hugging Face).
            api_key (str): The API key required for OpenAI provider.

        Raises:
            ValueError: If the provider is 'openai' but no API key is provided.
            NotImplementedError: If an unsupported provider is specified.
        """
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.pipeline = None

        if self.provider == 'openai':
            # Initialize OpenAI API with the provided API key
            if not self.api_key:
                raise ValueError("API key is required for OpenAI provider")
            openai.api_key = self.api_key
        elif self.provider == 'huggingface':
            # Initialize Hugging Face model pipeline
            self.pipeline = pipeline('text-generation', model=self.model)

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
            response = openai.Completion.create(engine=self.model, prompt=prompt, **kwargs)
            return response.choices[0].text.strip()
        elif self.provider == 'huggingface':
            if self.pipeline:
                # Generate text using the Hugging Face model pipeline
                result = self.pipeline(prompt, **kwargs)
                return result[0]['generated_text']
            else:
                raise ValueError("Model pipeline is not initialized for Hugging Face")
        else:
            raise NotImplementedError("The specified provider is not supported")