from individual import Individual
import problem_mnist as problem

train_data = problem.x_train
train_labels = problem.y_train

individual = Individual(problem.skeleton_genome)
individual.evaluate(problem.x_train)
print(individual.genome_outputs)
