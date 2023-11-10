import numpy as np
import re

def last_number(s):
    numbers = re.findall(r'\d+', s)
    return int(numbers[-1]) if numbers else None

def calculate_bits(min_value, max_value):
    range_size = max_value - min_value
    bits = int(np.floor(np.log2(range_size)) + 1)
    return bits

def encode_to_binary(value, min_value, max_value):
    # Check if the value is within the range
    if value < min_value or value > max_value:
        raise ValueError(f"Value must be between {min_value} and {max_value}")
    
    # Calculate the number of bits needed for the range
    bits = calculate_bits(min_value, max_value)
    
    # Subtract the minimum to normalize the value to zero-based range
    normalized_value = value - min_value
    
    # Format the normalized value as a binary string
    binary_string = format(normalized_value, f'0{bits}b')
    
    # Convert the binary string to a numpy array of integers 
    return np.array(list(binary_string), dtype=int)

def decode_from_binary(binary_array, min_value, max_value):
    # Check if the array is a numpy array
    if not isinstance(binary_array, np.ndarray):
        raise ValueError("Input must be a numpy array")
    
    # Convert the numpy array of 0s and 1s to a string and then to an integer
    binary_string = ''.join(binary_array.astype(str))
    
    # Calculate the number of bits needed for the range
    bits = calculate_bits(min_value, max_value)
    
    # Check if the binary string has the correct length
    if len(binary_string) != bits:
        raise ValueError(f"Binary string must be {bits} bits long")
    
    # Convert the binary string to an integer
    normalized_value = int(binary_string, 2)
    
    # Re-add the minimum value to get the actual value
    actual_value = normalized_value + min_value
    return actual_value



def bring_examples_gsm8k(dataset, max_num_examples):

    examples = []

    indexes = np.random.randint(len(dataset['train']), size=max_num_examples)

    for index in indexes:

        examples.append('Q: ' + dataset['train'][int(index)]['question'] + '\nA: ' + dataset['train'][int(index)]['answer'])

    return examples


def bring_problems_to_solve_gsm8k(dataset, max_num_examples):

    problems, answers = [], []

    indexes = np.random.randint(len(dataset['train']), size=max_num_examples)

    for index in indexes:

        problems.append('Q: ' + dataset['train'][int(index)]['question'] + '\nA: ')
        answers.append(dataset['train'][int(index)]['answer'])

    return problems, answers

#bring_examples_gsm8k(dataset, 3)