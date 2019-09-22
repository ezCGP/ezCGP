from individual import Individual
import problem
import numpy as np

train_data = problem.x_train
train_labels = problem.y_train

individual = Individual(problem.skeleton_genome)
individual.evaluate(problem.x_train, problem.y_train, (problem.x_test, problem.y_test))

individual.fitness.values = problem.scoreFunction(actual=problem.y_val, predict=individual.genome_outputs)
print('individual has fitness: ', individual.fitness.values)

