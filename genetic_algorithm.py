from experiment_conditions import Experiment_conditions
import numpy as np
from prompt import Prompt, encode_to_binary, decode_from_binary
import random
import pickle
import os
import datetime
import time

class GeneticAlgorithm:

    def __init__(self, experiment_conditions:Experiment_conditions, num_individuals:int, num_generatios:int) -> None:
        
        self.experiment_conditions = experiment_conditions

        if num_individuals % 2 != 0:
            raise Exception('Population must have an even number of individuals')

        self.num_individuals = num_individuals
        self.num_generations = num_generatios

        self.generations = {0:[]}
        self.fitness_per_gen = {0:[]}
        self.run_time = 0

    def save_experiment(self):

        path = self.experiment_conditions.llm.model + '--' + str(self.experiment_conditions.dataset.dataset_name) + '--' + str(datetime.datetime.now() )

        os.mkdir(path)

        exp_cond = {}
        exp_cond['llm'] = {'provider': self.experiment_conditions.llm.provider, 'model': self.experiment_conditions.llm.model}
        exp_cond['mutators'] = self.experiment_conditions.prompt_mutators
        exp_cond['sys_prompts'] = self.experiment_conditions.sys_prompts
        exp_cond['instructions'] = self.experiment_conditions.instructions
        exp_cond['examples'] = self.experiment_conditions.examples
        exp_cond['dataset'] = self.experiment_conditions.dataset.dataset_name

        with open(path + '/exp_conditions.pkl', 'wb') as f:
            pickle.dump(exp_cond, f)

        exp_run = {'run_time': self.run_time}

        for i in range(self.num_generations):

            individuals = {}

            for j in range(self.num_individuals):

                individuals[j] = {'sys_prompt_idx': self.generations[i][j].sys_prompt_idx,
                                  'instruction_idx': self.generations[i][j].instruction_idx,
                                  'num_examples': self.generations[i][j].num_examples,
                                  'fitness': self.fitness_per_gen[i][j]}
                
            exp_run[i] = individuals


        with open(path + '/ga_run.pkl', 'wb') as f:
            pickle.dump(exp_run, f)
    
    def init_population(self):

        for i in range(self.num_individuals):

            num_examples = np.random.randint(self.experiment_conditions.max_num_examples)
            sys_prompt_idx = np.random.randint(len(self.experiment_conditions.sys_prompts))
            instruction_idx = np.random.randint(len(self.experiment_conditions.instructions))

            individual = Prompt(self.experiment_conditions, sys_prompt_idx, instruction_idx, num_examples)

            self.generations[0].append(individual)

    
    def crossover(self, parent_1:Prompt, parent_2:Prompt):

        mutation_type = np.random.randint(2)

        cutoff = np.random.randint(len(parent_1.encoded_prompt))

        if mutation_type == 0:

            child_1 = np.concatenate((parent_1.encoded_prompt[0:cutoff], parent_2.encoded_prompt[cutoff:]))
            child_2 = np.concatenate((parent_2.encoded_prompt[0:cutoff], parent_1.encoded_prompt[cutoff:]))

        elif mutation_type == 1:

            child_1 = np.concatenate((parent_1.encoded_prompt[0:cutoff], parent_2.encoded_prompt[cutoff:]))
            child_2 = np.concatenate((parent_2.encoded_prompt[0:cutoff], parent_1.encoded_prompt[cutoff:]))

            i = np.random.randint(len(self.experiment_conditions.prompt_mutators))
            j = np.random.randint(len(self.experiment_conditions.prompt_mutators))

            inst_child_1_idx = decode_from_binary(child_1[12:])
            inst_child_2_idx = decode_from_binary(child_2[12:])

            new_instruct_child_1 = self.experiment_conditions.llm.mutate(self.experiment_conditions.prompt_mutators[i], 
                                                                           self.experiment_conditions.instructions[inst_child_1_idx])
            new_instruct_child_2 = self.experiment_conditions.llm.mutate(self.experiment_conditions.prompt_mutators[j],
                                                                           self.experiment_conditions.instructions[inst_child_2_idx])
            
            self.experiment_conditions.instructions.append(new_instruct_child_1)
            self.experiment_conditions.instructions.append(new_instruct_child_2)

            encoded_new_instruct_1 = encode_to_binary(len(self.experiment_conditions.instructions)-2, 6)
            encoded_new_instruct_2 = encode_to_binary(len(self.experiment_conditions.instructions)-1, 6)

            child_1 = np.concatenate((child_1[0:12], encoded_new_instruct_1))
            child_2 = np.concatenate((child_2[0:12], encoded_new_instruct_2))

        else:

            child_1 = parent_1.econded_prompt
            child_2 = parent_2.econded_prompt
        
        return child_1, child_2
    

    def create_next_generation(self, generation, encoded_children):

        self.generations[generation] = []

        for encoded_child in encoded_children:

            individual = Prompt(self.experiment_conditions, None, None, None, binary_encoding=encoded_child)

            self.generations[generation].append(individual)
        
        self.fitness_per_gen[generation] = []
    
    
    def selection_process(self, generation):

        encoded_children = []

        individuals_idx = range(self.num_individuals)

        selected_parents_idx = random.choices(individuals_idx, weights=self.fitness_per_gen[generation], k=self.num_individuals)

        i=0
        while i < len(selected_parents_idx):

            parent_1 = self.generations[generation][selected_parents_idx[i]]
            parent_2 = self.generations[generation][selected_parents_idx[i+1]]

            child_1, child_2 = self.crossover(parent_1, parent_2)

            encoded_children.append(child_1)
            encoded_children.append(child_2)

            i+=2
        
        return encoded_children
    

    def calculate_generation_fitness(self, generation):

        for prompt in self.generations[generation]:

            self.fitness_per_gen[generation].append(prompt.evaluate())


    
    def run_algorithm(self):

        start = time.time()

        for i in range(self.num_generations):

            self.calculate_generation_fitness(i)
    
            if i != self.num_generations-1:

                encoded_children = self.selection_process(i)

                self.create_next_generation(i+1, encoded_children)
        
        self.run_time = time.time() - start










