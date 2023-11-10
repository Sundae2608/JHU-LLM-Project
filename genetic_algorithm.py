from utils import calculate_bits, encode_to_binary, decode_from_binary, last_number
import numpy as np

class GeneticAlgorithm():

    def __init__(self, num_generations_, sys_instructions_, instructions_, examples_, pool_problems_, pool_answers_, num_individuals, p) -> None:
    
        self.num_generations = num_generations_
        self.sys_instructions = sys_instructions_
        self.instructions = instructions_
        self.examples = examples_
        self.pool_problems = pool_problems_
        self.pool_answ_ = pool_answers_

        self.max_num_examples = len(examples_)
        self.num_sys_instructions = len(self.sys_instructions)
        self.num_instructions = len(self.instructions)

        self.individuals_per_generation = num_individuals

        self.current_population, self.problem_indexes = [], []

        self.p = p


    def create_population(self):

        for i in range(self.individuals_per_generation):

            j = np.random.randint(self.num_sys_instructions)
            k = np.random.randint(self.num_instructions)
            l = np.random.randint(self.max_num_examples)
            m = np.random.randint(len(self.pool_problems))

            prompt_examples = ''
            for x in range(l):
                prompt_examples += self.examples[x] + ' \n'

            prompt = self.sys_instructions[j] + '\n' + self.instructions[k] + '\n' + prompt_examples + ' \n' + self.pool_problems[m]

            self.problem_indexes.append(m)
            self.current_population.append(prompt)



    def encode_prompt(self, num_examples, sys_instruction_index, instruction_index):

        num_examples_encoded = encode_to_binary(value=num_examples, min_value=0, max_value=self.max_num_examples)

        sys_instruction_encoded = encode_to_binary(value=sys_instruction_index, min_value=0, max_value=self.num_sys_instructions)

        instruction_encoded = encode_to_binary(value=instruction_index, min_value=0, max_value=self.num_instructions)

        return np.concatenate((num_examples_encoded, sys_instruction_encoded, instruction_encoded))


    def decode_prompt(self, encoded_array):
        
        num_examples_bits = calculate_bits(0, self.max_num_examples)
        sys_instruction_bits = calculate_bits(0, self.num_sys_instructions)

        # Split the array back into its original components
        num_examples_encoded = encoded_array[:num_examples_bits]
        sys_instruction_encoded = encoded_array[num_examples_bits:num_examples_bits + sys_instruction_bits]
        instruction_encoded = encoded_array[num_examples_bits + sys_instruction_bits:]

        # Decode 
        num_examples = decode_from_binary(num_examples_encoded, 0, self.max_num_examples)
        sys_instruction_index = decode_from_binary(sys_instruction_encoded, 0, self.num_sys_instructions)
        instruction_index = decode_from_binary(instruction_encoded, 0, self.num_instructions)

        num_examples = self.max_num_examples if num_examples > self.max_num_examples else num_examples
        sys_instruction_index = self.num_sys_instructions if sys_instruction_index > self.num_sys_instructions else sys_instruction_index
        instruction_index = self.num_instructions if instruction_index > self.num_instructions else instruction_index

        return num_examples, sys_instruction_index, instruction_index
    

    def crossover_1(self, parent1, parent2):

        cutoff = np.random.randint(len(parent1))

        child_1 = np.concatenate((parent1[0:cutoff], parent2[cutoff:]))
        child_2 = np.concatenate((parent2[0:cutoff], parent1[cutoff:]))

        return child_1, child_2


    def crossover_2(self, parent1, parent2, llm, mutation_prompt):

        cutoff = np.random.randint(len(parent1))

        child_1 = np.concatenate((parent1[0:cutoff], parent2[cutoff:]))
        child_2 = np.concatenate((parent2[0:cutoff], parent1[cutoff:]))

        num_examples_bits = calculate_bits(0, self.max_num_examples)
        num_sys_instr_bits = calculate_bits(0, self.num_sys_instructions)

        instruct_child_1 = decode_from_binary(child_1[num_examples_bits + num_sys_instr_bits:], 0, self.num_instructions)
        instruct_child_2 = decode_from_binary(child_2[num_examples_bits + num_sys_instr_bits:], 0, self.num_instructions)

        new_instruct_child_1 = llm.generate(mutation_prompt + '/n' + self.instructions[instruct_child_1])
        new_instruct_child_2 = llm.generate(mutation_prompt + '/n' + self.instructions[instruct_child_2])

        new_instructions_array = np.concatenate((self.instructions,np.array([new_instruct_child_1, new_instruct_child_2])))
        self.instructions = new_instructions_array
        self.num_instructions += 2

        encoded_new_instruct_1 = encode_to_binary(self.num_instructions-2, 0, self.num_instructions)
        encoded_new_instruct_2 = encode_to_binary(self.num_instructions-1, 0, self.num_instructions)

        child_1 = np.concatenate((child_1[0:num_examples_bits + num_sys_instr_bits], encoded_new_instruct_1))
        child_2 = np.concatenate((child_2[0:num_examples_bits + num_sys_instr_bits], encoded_new_instruct_2))

        return child_1, child_2
    

    def evaluate(self, prompts, answers, llm):

        evaluations = np.zeros(len(answers))

        model_ans = ['']*len(answers)

        for i in range(len(prompts)):

            model_ans[i] = llm.generate(prompts[i])

            if last_number(answers[i]) == last_number(model_ans[i] ):

                evaluations[i] = 1
        
        return evaluations, model_ans
    
    def pick_parents(self, good_individual_indexes, p):
        
        parents = []
        i = 0

        while i < self.individuals_per_generation:

            j_temp = np.random.randint(len(good_individual_indexes))
            k_temp = np.random.randint(len(good_individual_indexes))

            j = good_individual_indexes[j_temp]
            k = good_individual_indexes[k_temp]

            parent_1 = self.current_population[j]
            parent_2 = self.current_population[k]

            mating_prob = np.random.rand()

            if mating_prob > p:
                parents.append(parent_1)
                parents.append(parent_2)
                i+=2

        return parents

    def run(self):

        for i in range(self.num_generations):

            evals, model_answ = self.evaluate(self.current_population, self.pool_answ_[np.array(self.problem_indexes)])

            generation_fitness = np.mean(evals)

            good_individual_indexes = np.where(evals==1)[0]
            bad_individual_indexes = np.where(evals==0)[0]

            parents = self.pick_parents(good_individual_indexes, self.p)

            j=0
            while j < self.individuals_per_generation:

                pass
            ## TODO

