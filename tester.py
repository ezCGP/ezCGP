from individual import Individual
import problem_mnist as problem

train_data = problem.x_train
train_labels = problem.y_train

individual = Individual(problem.skeleton_genome)
individual.evaluate(problem.x_train, problem.y_train)
print(individual.genome_outputs)
# print(individual.dead)
print('labels: {} have shape: {}\ngenome_outputs: {} have shape: {}'\
    .format(train_labels, train_labels.shape, individual.genome_outputs, individual.genome_outputs.shape))
individual.fitness.values = problem.scoreFunction(actual=train_labels, predict=individual.genome_outputs)
print('individual has fitness: ', individual.fitness.values)
