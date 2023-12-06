# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 15:48:57 2023

@author: sridhar sharma
"""
import re
import openai
import numpy as np
import time
from openai import OpenAI
import os
import fireworks.client

from datasets import load_dataset
def last_number(s):
    numbers = re.findall(r'\d+', s)
    return int(numbers[-1]) if numbers else None
       

def bring_problems_to_solve(dataset, max_num_problems):

    problems, answers = [], []

    #indexes = np.random.randint(len(dataset['train']), size=max_num_problems)
    indexes = np.random.randint(len(dataset['test']), size=max_num_problems)

    for index in indexes:

        problems.append('Q: ' + dataset['train'][int(index)]['question'] + '\nA: ')
        answers.append(dataset['train'][int(index)]['answer'])

    return problems, answers

def select_model(model_name):
    lcpp_llm = None
    openai_client=None
    if model_name == "GPT_3":
        print("Model selected is GPT_3")
        #from openai import OpenAI
        #import os
        #OPENAI_API_KEY=""
        # Access the API key from the environment variable
        api_key = os.environ.get('OPENAI_API_KEY')
        OPENAI_API_KEY=api_key        
        #openai_client = OpenAI(api_key=api_key)
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
    elif model_name == "LLAMA-2":
        print("Model selected is LLAMA-2")
        model_name_or_path = "TheBloke/Llama-2-13B-chat-GGML"
        model_basename = "llama-2-13b-chat.ggmlv3.q5_1.bin" # the model is in bin format
        from huggingface_hub import hf_hub_download
        model_path = hf_hub_download(repo_id=model_name_or_path, filename=model_basename)
        #setup machine parameters
        lcpp_llm = None
        from llama_cpp import Llama
        # GPU
        lcpp_llm = Llama(
                          model_path=model_path,
                          n_threads=2, # CPU cores
                          n_ctx=1024,
                          n_batch=512, # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
                          n_gpu_layers=32 # Change this value based on your model and your GPU VRAM pool.
                       )
    elif model_name == "DAVINCI":
        api_key = os.environ.get('OPENAI_API_KEY')
        OPENAI_API_KEY=api_key
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
    elif model_name == "FIREWORKS_LLAMA_13":
        fireworks.client.api_key = os.environ.get('FIREWORKS_API_KEY')
    elif model_name == "FIREWORKS_LLAMA_70":
        fireworks.client.api_key =os.environ.get('FIREWORKS_API_KEY')
    else:
        print("Unsupported model")
    return lcpp_llm,openai_client

class  PopulationMember():
    def __init__(self,problems_per_population_member,max_num_examples):
        print('Initializing Class Genetic Algorithm')
        self.sys_prompts = np.array(['Think like a graduate student and answer.',
                                     'Think like a mathematics teacher  and answer',
                                     'Approach the problem as a mathematics educator and provide your solution',
                                     'Demonstrate the problem-solving process with the approach of a math instructor',
                                     'Offer your answer using the teaching methodology of a mathematics educator',
                                     'Think pedagogically and present your solution as if you were teaching it in a classroom',
                                     'Approach the solution with the clarity and explanation of a mathematics teacher',
                                     'Share your answer, providing step-by-step guidance akin to a math educator',
                                     'Explain your solution as if you were guiding students through the problem as a math teacher',
                                     'Employ the strategies and clarity of explanation expected of a mathematics instructor in your response',
                                     'Solve the question while considering the method a math teacher would employ',
                                     'Think like a mathematics professor and solve this problem.',
                                     'Approach this problem with the precision and depth expected of a mathematics professor.',
                                     'Demonstrate your mastery by solving this problem through the lens of a mathematics professor',
                                     'Think deeply and analytically like a mathematics professor to resolve this problem.',
                                     'Utilize advanced mathematical theory and reasoning in your solution, akin to a math professor perspective.',
                                     'Approach this problem with the expertise and insight expected of a mathematics professor.',
                                     'Solve the problem with the rigor and sophistication typical of a mathematics professors approach.',
                                     'You are a helpful math solving assistant and helping someone with basic math skills.'
                              ])

        self.instructions = np.array(['Solve the math word problem, giving your answer as an arabic numeral.',
                                      'Provide the numerical solution to the math word problem using Arabic numerals.',
                                      'Find the solution and present it in the form of a numerical value written in Arabic numerals.',
                                      'Provide the solution to the word problem as a numerical figure in Arabic numerals',
                                      'Apply fundamental mathematical principles to solve the problem, and provide a clear numeric answer.',
                                      'Explain the problem and its solution using simple mathematical terms suitable for beginners.',
                                      'Present the problem and its resolution in a way that is easily graspable by those with elementary math understanding',
                                      'Simplify the problem explanation and solution for individuals with basic mathematical understanding.',
                                      'Adopt a beginner-friendly teaching style to elucidate the problem and its solution for individuals with limited math expertise.', 
                                      'Adjust your teaching approach to elucidate the problem and answer for those with fundamental math comprehension',
                                      'Teach the problem and its resolution as if guiding someone with elementary math skills.',
                                      'Explain the problem and its solution using simple mathematical terms suitable for beginners.',
                                      'Present the problem and solution in a manner accessible to someone with foundational math understanding.',
                                      'Adopt a beginner-friendly teaching style to elucidate the problem and its solution for individuals with limited math expertise.',
                                      'Adapt a teaching approach, explaining the problem and solution as if educating someone with basic mathematical knowledge.'
                                ])
        self.max_num_examples=max_num_examples
        self.num_problems=problems_per_population_member
        self.gene=[]
        
    
    def create_prompts_old(self,problems,dataset):
        self.prompt_template_base=f'''

          SYSTEM: '''
        sys_prompt_idx=np.random.randint(len(self.sys_prompts))
        self.gene.append(sys_prompt_idx)
        instruction_prompt_idx=np.random.randint(len(self.instructions))
        self.gene.append(instruction_prompt_idx)
        prompt_template=self.prompt_template_base+self.sys_prompts[sys_prompt_idx]+f'''
        '''
        prompt_template=prompt_template+self.prompt_template_base+self.instructions[instruction_prompt_idx]+f'''
        '''
        problem_with_prompt=[]

        self.example_prompt=self.add_examples(dataset)
        #First add example to prompt_template
        prompt_template=prompt_template+self.example_prompt
        print('New Prompt Template')
        prompt_template=prompt_template+f'''USER:Please answer this question
        '''
        for i in range(self.num_problems):
            #prompt=prompt_template+problems[i]
            prompt=prompt_template+problems[i]+f''' Assistant:
            '''
            problem_with_prompt.append(prompt)
        return problem_with_prompt
    
    def create_prompts(self,problems,dataset):
        sys_prompt_idx=np.random.randint(len(self.sys_prompts))
        self.gene.append(sys_prompt_idx)
        instruction_prompt_idx=np.random.randint(len(self.instructions))
        self.gene.append(instruction_prompt_idx)
        problem_with_prompt=[]

        self.example_prompt=self.add_examples(dataset)
        for i in range(self.num_problems):
            #prompt=prompt_template+problems[i]
            prompt=[{"role": "system", "content":self.sys_prompts[sys_prompt_idx] },
                                                                           {"role": "user", "content": self.instructions[instruction_prompt_idx] + '\n' + self.example_prompt + '\n' + problems[i]}
                                                                           ],
            problem_with_prompt.append(prompt)
        
        return problem_with_prompt
    
    def bring_examples(self, dataset, max_num_examples):

        examples = []

        indexes = np.random.randint(len(dataset['train']), size=max_num_examples)

        for index in indexes:

            examples.append('Q: ' + dataset['train'][int(index)]['question'] + '\nA: ' + dataset['train'][int(index)]['answer'])
            self.gene.append(index)
        return examples

    def add_examples(self, dataset):
         num_examples=np.random.randint(self.max_num_examples)
         self.gene.append(num_examples)
         if num_examples != 0:
              example_prompt=f''' "These are some examples of similar problems and how to solve them: 
              '''
              #examples=bring_examples(dataset, self.max_num_examples)
              examples=self.bring_examples(dataset, num_examples)
              for i in range(len(examples)):
                  s=str(examples[i])
                  example_prompt=example_prompt+s
         else:
              example_prompt=''''''
         return example_prompt

    def evaluate_problems(self, problem_prompts,answers, model_name,lcpp_llm,openai_client):
          evaluations = np.zeros(len(answers))
          model_ans = ['']*len(answers)

          for i in range(self.num_problems):
                prompt = problem_prompts[i]
                actual_answer= last_number(answers[i])
                print(prompt)
                if model_name == "GPT_3":
                     print("Model being evaluated  is GPT_3")
                     prompt_str=''.join(str(e) for e in prompt[0])
                     response=openai_client.completions.create(
                                         model="gpt-3.5-turbo-instruct",
                                         prompt=prompt_str,
                                         temperature=1,
                                         max_tokens=2048,
                                         top_p=1,
                                         frequency_penalty=0,
                                         presence_penalty=0
                                   )
                     print(response.choices[0].text)
                     generated_answer=last_number(response.choices[0].text)
                     time.sleep(25)
                elif model_name == "LLAMA-2":
                     print("Model selected is LLAMA-2")
                     response =lcpp_llm(prompt=prompt, max_tokens=1024, temperature=0.5, top_p=0.95,
                     repeat_penalty=1.2, top_k=150,
                     echo=True)
                     print(response['choices'][0]['text'])
                     generated_answer=last_number(response['choices'][0]['text'])
                elif model_name == "DAVINCI":
                     response = openai_client.completions.create(
                                      model="text-davinci-003",
                                      prompt=prompt,
                                      temperature=1,
                                      max_tokens=1024,
                                      top_p=1,
                                      frequency_penalty=0,
                                      presence_penalty=0
                                      )
                     print(response.choices[0].text)
                     generated_answer=last_number(response.choices[0].text)
                     time.sleep(25)
                elif model_name == "FIREWORKS_LLAMA_13":
                     done=False
                     while not done:
                        try:
                               completion = fireworks.client.ChatCompletion.create(
                                      model="accounts/fireworks/models/llama-v2-13b-chat",
                                      messages=prompt[0],
                                      stream=False,
                                      n=1,
                                      max_tokens=1024,
                                      temperature=0.1,
                                      top_p=0.9 
                                  )
                               print(completion.choices[0].message.content)
                               generated_answer=last_number(completion.choices[0].message.content)
                               done=True
                        except:
                               time.sleep(10)
                elif model_name == "FIREWORKS_LLAMA_70":
                     done=False
                     while not done:
                       try:
                                completion = fireworks.client.ChatCompletion.create(
                                    model="accounts/fireworks/models/llama-v2-70b-chat",
                                    messages=prompt[0],
                                    n=1,
                                    max_tokens=800,
                                    temperature=0.1,
                                    top_p=0.9 
                                )
                                done = True
                                print(completion.choices[0].message.content)
                                generated_answer=last_number(completion.choices[0].message.content)
                       except:
                                time.sleep(10)
                      
                else:
                     print("Unsupported model")        
                print('Actual Answer is ',actual_answer)
                print('Generated Answer is ',generated_answer)
                if actual_answer == generated_answer:
                    evaluations[i] = 1
                    model_ans[i]= generated_answer
          return evaluations
      
    def mutate_gene(self):
           #We can mutate only the sys_prompt, instruction_prompt and number of examples
           print('Mutating Gene')
           mutation_index=np.random.randint(3)
           if mutation_index ==0:
               #does not cover the case when the new index is same as old
               #Needs fixing
               sys_prompt_idx=np.random.randint(len(self.sys_prompts))
               self.gene[0]=sys_prompt_idx
           elif mutation_index ==1: 
               instruction_prompt_idx=np.random.randint(len(self.instructions))
               self.gene[1]=instruction_prompt_idx
           else: 
               #Need to add or delete examples
               num_examples=np.random.randint(self.max_num_examples)
               if num_examples <= self.gene[2]:
                   num_remove=self.gene[2]-num_examples
                   print("Need to remove ",num_remove," examples")
                   idx_removal=np.random.randint(low=0,high=self.gene[2],size=num_remove)
                   for i in range(num_remove):
                       removal_index=idx_removal[i]+3
                       print("Removed  Example ",removal_index)
                       try:
                           del self.gene[removal_index]
                           self.gene[2]=self.gene[2]-1
                        # If a ValueError is raised, the element is not in the list
                       except IndexError:                    
                           self.gene[2]=self.gene[2]   
               else:
                   num_add=num_examples-self.gene[2]
                   print("Need to add ",num_add," examples")
                   indexes = np.random.randint(len(dataset['train']), size=num_add)
                   for index in indexes:
                       self.gene.append(index)
                   self.gene[2]=num_examples

    def reconstruct_examples(self,dataset):
        num_examples=self.gene[2]
        if num_examples != 0:
             example_prompt=f''' "These are some examples of similar problems and how to solve them: 
             '''
             examples = []

             indexes = self.gene[3:3+self.gene[2]]

             for index in indexes:
                 examples.append('Q: ' + dataset['train'][int(index)]['question'] + '\nA: ' + dataset['train'][int(index)]['answer'])

             for i in range(len(examples)):
                 s=str(examples[i])
                 example_prompt=example_prompt+s
        else:
             example_prompt=''''''
        return example_prompt

    def reconstruct_prompt(self,problems,dataset):        
        sys_prompt_idx=self.gene[0]
        instruction_prompt_idx=self.gene[1]
        problem_with_prompt=[]

        self.example_prompt=self.reconstruct_examples(dataset)
       
        for i in range(self.num_problems):
            prompt=[{"role": "system", "content":self.sys_prompts[sys_prompt_idx] },
                                                                           {"role": "user", "content": self.instructions[instruction_prompt_idx] + '\n' + self.example_prompt + '\n' + problems[i]}
                                                                           ],
            problem_with_prompt.append(prompt)
        return problem_with_prompt
  
class GeneticAlgo():
     def __init__(self,population_size,carry_forward_next_gen):
         self.population_size=population_size
         #self.fitness_score=np.zeros((population_size,1), dtype=float)
         self.population=[]
         self.carry_forward_next_gen=carry_forward_next_gen
         #Garbage value of fitness score to test w/o calling API
         #self.fitness_score=[20.0,30.0,10.0,40.0,0.0,60.0,20.0,80.0,50.0,70.0]
         self.fitness_score=np.random.rand(population_size,1)*100
         self.problems_solved=np.zeros((population_size,1), dtype=int)
         self.num_corrected=np.zeros((population_size,1), dtype=int)
    
     def add_member(self,member):
         self.population.append(member)
    
     def compute_fitness_score(self,generation,problems,dataset):
         for i in range(self.population_size):
             if generation == 0:
                 problems_prompt=self.population[i].create_prompts(problems,dataset)
             else:
                problems_prompt=self.population[i].reconstruct_prompt(problems,dataset)
             evaluations=self.population[i].evaluate_problems(problems_prompt,answers,model_name,lcpp_llm,openai_client)
             self.problems_solved+=len(evaluations)
             self.num_corrected+=np.sum(evaluations)
             accuracy=np.sum(evaluations)/len(evaluations)*100
             self.fitness_score[i]=accuracy
     
     def crossover(self,parent1,parent2):
         #Splice point can be one of the first 
         splice_point=np.random.randint(3)
         gene1=self.population[parent1].gene[0:splice_point]+self.population[parent2].gene[splice_point:]
         gene2=self.population[parent2].gene[0:splice_point]+self.population[parent1].gene[splice_point:]
         return gene1,gene2 

    
     def select_next_population(self):
         number_kill=int((1-self.carry_forward_next_gen)*self.population_size)
         print('Number kill ',number_kill)
         #determine which members of the population had low fitness score 
         #and hence eliminated
         #idx = np.argpartition(self.fitness_score,number_kill)
         #idx = np.argsort(self.fitness_score)
         idx = np.argsort(np.transpose(self.fitness_score))[0]
         print('Index ',idx)
         print('Population Members being eliminated are ',idx[0:number_kill])
         output_string='Population Members being eliminated are '+str(idx[0:number_kill])
         fo.write(output_string)
         fo.flush()
         print('Best Performing population members are ',idx[self.population_size-2:self.population_size])
         output_string='Crossover candidates are '+str(idx[self.population_size-2])+' and '+str(idx[self.population_size-1])+' Replace members '+str(idx[2])+' and '+str(idx[3])+'\n'
         fo.write(output_string)
         fo.flush()
         #Mutation of 2 random members 
         #and replacement of the worst 2 in the previous population        
         mutation_idx1=np.random.randint(self.population_size)
         self.population[idx[0]]=self.population[mutation_idx1]
         self.population[idx[0]].mutate_gene()
         output_string='Mutating member '+str(mutation_idx1)+' and updating member '+str(idx[0])+'\n'
         fo.write(output_string)
         fo.flush()
         mutation_idx2=np.random.randint(self.population_size)
         self.population[idx[1]]=self.population[mutation_idx2]
         self.population[idx[1]].mutate_gene()
         output_string='Mutating member '+str(mutation_idx2)+' and updating member '+str(idx[1])+'\n'
         fo.write(output_string)
         fo.flush()
         # Now do a crossover for the best 2 candidate members
         #Best member from prev generation
         crossover_parent1=idx[self.population_size-1]
         #2nd Best member from prev generation
         crossover_parent2=idx[self.population_size-2]
         gene1,gene2=self.crossover(crossover_parent1,crossover_parent2)
         self.population[idx[2]].gene=gene1
         self.population[idx[3]].gene=gene2


model_name="GPT_3" 
#model_name="DAVINCI"       
#select_model("GPT_3")
#select_model(model_name)
#model_name="FIREWORKS_LLAMA_13"
#model_name="FIREWORKS_LLAMA_70"

dataset = load_dataset("gsm8k", 'main')
train_data = dataset["train"]
test_data = dataset["test"]       

print('Length of Training Dataset is ',len(train_data))
print('Length of Test Dataset is ',len(test_data))
#model_name="LLAMA-2"

lcpp_llm,openai_client=select_model(model_name)

problems_per_population_member=10
max_num_examples=8
population_size=10
#Indicates the fraction of the population that will survive to the next generation
#unfortunately hard coded for now
carry_forward_next_gen=0.6
num_generations=10
#Enable to reuse problems for all generations
#[problems,answers]=bring_problems_to_solve(dataset,problems_per_population_member)

genetic=GeneticAlgo(population_size,carry_forward_next_gen)

#Construct the initial population and compute the fitness score
for i in range(population_size):
       member=PopulationMember(problems_per_population_member,max_num_examples)
       genetic.add_member(member)

# Start time
start_time = time.time()
fo = open("results.txt", "w")
for generation in range(num_generations):
#    Enable to use different problems each generation 
     [problems,answers]=bring_problems_to_solve(dataset,problems_per_population_member)
     output_string='Generation '+str(generation)+'\n'
     fo.write(output_string)
     fo.flush()
     genetic.compute_fitness_score(generation,problems,dataset)
     average_accuracy=np.average(genetic.fitness_score)
     print('Accuracy of  population is ',np.transpose(genetic.fitness_score),'%')

     output_string='Accuracy of  population is '+str(genetic.fitness_score)+'%'+' Average Accuracy is '+str(average_accuracy)+'\n'
     fo.write(output_string)
     fo.flush()
     genetic.select_next_population()    
# End time
end_time = time.time()
fo.close()

# Calculate elapsed time
elapsed_time = end_time - start_time




print('Elapsed Time for evaluating  is ',elapsed_time)
