from transformers import pipeline
from openai import OpenAI
import openai
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch
import fireworks.client as firew_client


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

        elif self.provider == 'fireworks':
            # Initialize OpenAI API with the provided API key
            if not self.api_key:
                raise ValueError("API key is required for Fireworks provider")

            firew_client.api_key = self.api_key


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

        if self.provider == 'openai' and self.model in ['gpt-3.5-turbo', 'gpt-4-1106-preview']:
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
                    time.sleep(10)
            return response.choices[0].message.content
        
        elif self.provider == 'fireworks' and self.model in ['llama-v2-70b-chat', 'falcon-7b', 'mistral-7b-instruct-4k', 'llama-v2-13b-chat']:
            done=False

            while not done:
                try:
                    completion = firew_client.ChatCompletion.create(
                                    model="accounts/fireworks/models/" + self.model,
                                    messages=[{"role": "system", "content": sys_prompt},
                                                                        {"role": "user", "content": instruction + '\n' + examples + '\n' + problem}
                                                                        ],
                                    n=1,
                                    max_tokens=800,
                                    temperature=0.1,
                                    top_p=0.9, 
                                )
                    done = True
                except:
                    time.sleep(10)
            return completion.choices[0].message.content        

        else:
            raise NotImplementedError("The specified provider or model is not supported")
        
    def mutate(self, mutation_prompt:str, instruction:str):
        done=False

        if self.provider == 'openai' and self.model in ['gpt-3.5-turbo']:

            while not done:
                try:

                    response = self.client.chat.completions.create(model=self.model,
                                                                messages=[
                                                                            {"role": "user", "content": mutation_prompt + '\n' + instruction}
                                                                            ])
                    done=True
                except:
                    time.sleep(10)
            return response.choices[0].message.content
        elif self.provider == 'fireworks':
            while not done:
                try:
                    completion = firew_client.ChatCompletion.create(
                                    model="accounts/fireworks/models/" + self.model,
                                    messages=[{"role": "user", "content": mutation_prompt + '\n' + instruction}],
                                    n=1,
                                    max_tokens=1000,
                                    temperature=0.1,
                                    top_p=0.9, 
                                )
                    done = True
                except:
                    time.sleep(10)
            return completion.choices[0].message.content 
        else:
            raise NotImplementedError("The specified provider is not supported")
