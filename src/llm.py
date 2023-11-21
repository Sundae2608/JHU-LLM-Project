from transformers import pipeline
from openai import OpenAI
import openai
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch


class LLM:
    def __init__(self, provider='openai', model='gpt-3.5-turbo', api_key=None):

        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.pipeline = None

        if self.provider == 'openai':
            # Initialize OpenAI API with the provided API key
            if not self.api_key:
                raise ValueError("API key is required for OpenAI provider")

            self.client = OpenAI(api_key=self.api_key)


        elif self.provider == 'huggingface':

            if self.model == 'falcon-7b-instruct':

                hf_model = "tiiuae/falcon-7b-instruct"

                self.tokenizer = AutoTokenizer.from_pretrained(hf_model)
                self.pipeline = transformers.pipeline(
                                "text-generation",
                                model=hf_model,
                                tokenizer=self.tokenizer,
                                torch_dtype=torch.bfloat16,
                                device_map="auto",
                            )
            

    def generate(self, sys_prompt:str, instruction:str, examples:str, problem:str, **kwargs):

        if self.provider == 'openai' and self.model in ['gpt-3.5-turbo']:
            done=False
            while not done:

                try:
                    response = self.client.chat.completions.create(model=self.model,
                                                                timeout=5,
                                                                messages=[{"role": "system", "content": sys_prompt},
                                                                            {"role": "user", "content": instruction + '\n' + examples + '\n' + problem}
                                                                            ])
                    done = True
                except:
                    time.sleep(5)
            return response.choices[0].message.content
        

        elif self.provider == 'huggingface' and self.model == 'falcon-7b-instruct':

            prompt = sys_prompt + ' \n' + instruction + ' \n' + examples + ' \n' + problem

            sequences = self.pipeline(prompt,
                                        max_length=1000,
                                        do_sample=True,
                                        top_k=10,
                                        num_return_sequences=1,
                                        eos_token_id=self.tokenizer.eos_token_id
)

            return sequences[0]['generated_text']
        

        else:
            raise NotImplementedError("The specified provider is not supported")
        
    def mutate(self, mutation_prompt:str, instruction:str):

        if self.provider == 'openai' and self.model in ['gpt-3.5-turbo']:

            response = self.client.chat.completions.create(model=self.model,
                                                           messages=[
                                                                     {"role": "user", "content": mutation_prompt + '\n' + instruction}
                                                                     ])
            return response.choices[0].message.content
        elif self.provider == 'huggingface':
            return None
        else:
            raise NotImplementedError("The specified provider is not supported")