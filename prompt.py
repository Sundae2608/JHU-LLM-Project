from experiment_conditions import Experiment_conditions
import numpy as np


class Prompt:

    def __init__(self, experiment_conditions:Experiment_conditions, sys_prompt_idx:int, 
                 instruction_idx:int, num_examples:int, binary_encoding=[]) -> None:
        
        self.experiment_conditions = experiment_conditions
        
        if len(binary_encoding) > 0:

            self.num_examples, self.sys_prompt_idx, self.instruction_idx = self.decode_prompt(binary_encoding)

        else:
        
            self.sys_prompt_idx = sys_prompt_idx
            self.instruction_idx = instruction_idx
            self.num_examples = num_examples
            
        self.encoded_prompt = self.encode_prompt()


    def encode_prompt(self):

        num_examples_encoded = encode_to_binary(value=self.num_examples, bits_to_use=6)

        sys_instruction_encoded = encode_to_binary(value=self.sys_prompt_idx, bits_to_use=6)

        instruction_encoded = encode_to_binary(value=self.instruction_idx, bits_to_use=6)

        return np.concatenate((num_examples_encoded, sys_instruction_encoded, instruction_encoded))
    

    def decode_prompt(self, encoded_array):
        

        # Split the array back into its original components
        num_examples_encoded = encoded_array[:6]
        sys_instruction_encoded = encoded_array[6:12]
        instruction_encoded = encoded_array[12:]

        # Decode 
        num_examples = decode_from_binary(num_examples_encoded)
        sys_instruction_index = decode_from_binary(sys_instruction_encoded)
        instruction_index = decode_from_binary(instruction_encoded)

        num_examples = self.experiment_conditions.max_num_examples if num_examples > self.experiment_conditions.max_num_examples  else num_examples
        sys_instruction_index = len(self.experiment_conditions.sys_prompts) if sys_instruction_index > len(self.experiment_conditions.sys_prompts) else sys_instruction_index
        instruction_index = len(self.experiment_conditions.instructions) if instruction_index > len(self.experiment_conditions.instructions)  else instruction_index

        return num_examples, sys_instruction_index, instruction_index
    

    def evaluate(self):

        fitness = 0

        problems, answers = self.experiment_conditions.dataset.pick_random_problems(5)

        for i in range(len(problems)):

            examples = ''

            for j in range(self.num_examples):

                examples += self.experiment_conditions.examples[j] + '\n'


            sys_prompt = self.experiment_conditions.sys_prompts[self.sys_prompt_idx]

            instruction = self.experiment_conditions.instructions[self.instruction_idx]
            
            predicted_ans = self.experiment_conditions.llm.generate(sys_prompt, instruction, examples, problems[i])

            fitness += self.experiment_conditions.dataset.evaluate(answers[i], predicted_ans)
        
        fitness /= len(problems)

        return fitness 




def calculate_bits(min_value, max_value):
    range_size = max_value - min_value
    bits = int(np.floor(np.log2(range_size)) + 1)
    return bits

def encode_to_binary(value, bits_to_use):
    
    # Format the normalized value as a binary string
    binary_string = format(value, f'0{bits_to_use}b')
    
    # Convert the binary string to a numpy array of integers 
    return np.array(list(binary_string), dtype=int)

def decode_from_binary(binary_array):

    # Check if the array is a numpy array
    if not isinstance(binary_array, np.ndarray):
        raise ValueError("Input must be a numpy array")
    
    # Convert the numpy array of 0s and 1s to a string and then to an integer
    binary_string = ''.join(binary_array.astype(str))
    
    # Convert the binary string to an integer
    value = int(binary_string, 2)
     
    return value