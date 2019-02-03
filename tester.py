from individual import Individual
import problem_mnist as problem

individual = Individual(problem.skeleton_genome)
individual.evaluate(problem.x_train, problem.y_train)
print(individual.genome_outputs)