import matplotlib.pyplot as plt
import numpy as np

def load_prompts(file_path):
    with open(file_path, 'r') as file:
        prompts = file.readlines()
    return [prompt.strip() for prompt in prompts]


def plot_gene_frequency(results, cutoff):

    generations = len(results.keys())-1

    individuals = len(results[0].keys())

    sys_prompt_genes = []
    instruct_genes = []
    num_examples_genes = []

    for i in range(generations):
        for j in range(individuals):

            if results[i][j]['fitness'] >= cutoff:

                sys_prompt_genes.append(results[i][j]['sys_prompt_idx'])
                instruct_genes.append(results[i][j]['instruction_idx'])
                num_examples_genes.append(results[i][j]['num_examples'])

    sys_prompt_genes = np.array(sys_prompt_genes)
    instruct_genes = np.array(instruct_genes)
    num_examples_genes = np.array(num_examples_genes)

    plt.figure(figsize=(18, 7))

    counts = np.bincount(sys_prompt_genes)
    numbers = np.arange(len(counts))

    # First subplot - Bar plot for array1
    plt.subplot(1, 3, 1)
    plt.bar(numbers, counts, color='skyblue')
    plt.title('Bar Plot of Array 1')
    plt.xlabel('Index')
    plt.ylabel('Value')

    counts = np.bincount(instruct_genes)
    numbers = np.arange(len(counts))

    # Second subplot - Line plot for array2
    plt.subplot(1, 3, 2)
    plt.bar(numbers, counts, color='skyblue')
    plt.title('Line Plot of Array 2')
    plt.xlabel('Index')
    plt.ylabel('Value')

    counts = np.bincount(num_examples_genes)
    numbers = np.arange(len(counts))

    # Third subplot - Scatter plot for array3
    plt.subplot(1, 3, 3)
    plt.bar(numbers, counts, color='skyblue')
    plt.title('Scatter Plot of Array 3')
    plt.xlabel('Index')
    plt.ylabel('Value')

    # Displaying the plot
    plt.tight_layout()
    plt.show()


def plot_fitness_per_generation(results):
    generations = len(results.keys()) - 1
    individuals = len(results[0].keys())

    median = np.zeros(generations)
    min_ = np.zeros(generations)
    max_ = np.zeros(generations)

    for i in range(generations):
        fitness_per_gen = np.zeros(individuals)
        for j in range(individuals):
            fitness_per_gen[j] = results[i][j]['fitness']

        median[i] = np.median(fitness_per_gen)
        min_[i] = np.min(fitness_per_gen)
        max_[i] = np.max(fitness_per_gen)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(range(generations), median, label='Median Fitness', color='blue', marker='o')
    plt.plot(range(generations), min_, label='Min Fitness', color='green', marker='v')
    plt.plot(range(generations), max_, label='Max Fitness', color='red', marker='^')
    
    plt.title('Fitness Metrics Per Generation')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.legend()
    plt.grid(True)
    plt.show()



def plot_gene_frequency_per_gen(results, gen):

    individuals = len(results[0].keys())

    sys_prompt_genes = []
    instruct_genes = []
    num_examples_genes = []

    for j in range(individuals):
        sys_prompt_genes.append(results[gen][j]['sys_prompt_idx'])
        instruct_genes.append(results[gen][j]['instruction_idx'])
        num_examples_genes.append(results[gen][j]['num_examples'])

    sys_prompt_genes = np.array(sys_prompt_genes)
    instruct_genes = np.array(instruct_genes)
    num_examples_genes = np.array(num_examples_genes)

    plt.figure(figsize=(18, 7))

    counts = np.bincount(sys_prompt_genes)
    numbers = np.arange(len(counts))

    # First subplot - Bar plot for array1
    plt.subplot(1, 3, 1)
    plt.bar(numbers, counts, color='skyblue')
    plt.title('Bar Plot of Array 1')
    plt.xlabel('Index')
    plt.ylabel('Value')

    counts = np.bincount(instruct_genes)
    numbers = np.arange(len(counts))

    # Second subplot - Line plot for array2
    plt.subplot(1, 3, 2)
    plt.bar(numbers, counts, color='skyblue')
    plt.title('Line Plot of Array 2')
    plt.xlabel('Index')
    plt.ylabel('Value')

    counts = np.bincount(num_examples_genes)
    numbers = np.arange(len(counts))

    # Third subplot - Scatter plot for array3
    plt.subplot(1, 3, 3)
    plt.bar(numbers, counts, color='skyblue')
    plt.title('Scatter Plot of Array 3')
    plt.xlabel('Index')
    plt.ylabel('Value')

    # Displaying the plot
    plt.tight_layout()
    plt.show()