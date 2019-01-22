### mate_methods.py

# my scripts
from genome import Genome

class Mate(Genome):

    def __init__(self, genome_input_dtypes, genome_output_dtypes,
                       genome_main_count, genome_arg_count):
        Genome.__init__(self, genome_input_dtypes, genome_output_dtypes,
                              genome_main_count, genome_arg_count)
        self.mate_methods = [] #list(mate_dict.keys())
        self.mate_weights = [] #self.buildWeights('mate_methods', mate_dict)
        #self.mate_dict = mate_dict


    # honstly, don't mate. all of this is way too experimental to introduce now. stick with 4 mutant offspring and then get mate off the ground later
    def dont_mate(self, other):
        # I MADE THIS DUMMY FTN JUST SO WE CAN GAVE A setup_ftn_dict... SET mate_prob=0
        return 0

    """
    def mate(self, other, crossover, keep_both_offspring):
        '''
        Mating is skipped for CGP but we are testing the implementation of a new crossover method called
        single point subgraph crossover which picks a random active node to crossover the genome.
        Once the crossover occurs, nodes need to be reconnected to matching data types and active nodes are
        attempted to be linked to other active nodes.
        There is also the option to return both offsprings instead of the recommended one offspring as
        outlined in the subgraph paper.
        '''
        offsprings = []
        numArgs=self.numArgs
        

        if crossover=="uniform":
            '''
            Execute uniform crossover for both genome and arguments.
            The genome needs to be 'fixed' after crossover so that the data types match.
            Arguments shouldn't have this problem because all individuals should have the same
            argument types for each index in the arguments list.
            CODE IS NOT FINISHED...prob won't be needed and can get tossed later
            '''

            # initialize offspring
            offspring = deepcopy(self)

            # genome first
            for node_index in range(numMain+numOutputs):
                if random.random() <= .5:
                    #offspring.genome[node_index] = other.genome[node_index]
                    expected_types = operDict[offspring.genome[node_index]["ftn"]]["inputs"]
                    #if 

                else:
                    pass

            # now fix nodes so that datatypes match


            # then arguments list
            for arg_index in range(numArgs):
                if random.random() <= .5:
                    offspring.arguments[arg_index] = other.arguments[arg_index]
                else:
                    pass

            return self


        elif crossover=="subgraph":
            '''
            perform subgraph crossover as defined by Kalkreuth, Rudolph, Droschinsky
            '''
            
            # collect the active nodes
            M1 = self.active_inputs
            M2 = other.active_inputs

            # filter M1 and M2 to only include numbers 0->numMain
            M1 = [i for i in M1 if i>=0 and i<numMain]
            M2 = [i for i in M2 if i>=0 and i<numMain]

            # choose an active node at random from each and get the min of the two
            if len(M1) == 0 and len(M2) == 0: # shouldn't happen if output nodes can't mutate and change to be directly the input nodes
                print("rare error, rerun with new random number", flush=True)
            elif len(M1) == 0:
                M1=M2
            elif len(M2) == 0:
                M2=M1
            else:
                #all good
                pass

            C1 = random.choice(M1)
            C2 = random.choice(M2)

            # start building offspring
            if C1 < C2:
                C = C1
                offspring = deepcopy(other) #the second half of offspring will dominate the genome so keep all things half2
                offspring.genome = self.genome[0:C+1] + other.genome[C+1:]
                half1 = M1
                half2 = M2
            else:
                C = C2
                offspring = deepcopy(self)
                offspring.genome = other.genome[0:C+1] + self.genome[C+1:]
                half1 = M2
                half2 = M1

            # go through all nodes in half2 and see which have inputs < C
            # make sure that the inputs and outputs match.
            # and if that node is active switch the inputs to active nodes
            # get active nodes in the first half of half2 that are not in part1.
            # those are likely the nodes that are disconnected to the orignal part

            for node_index in range(C+1, numMain):
                for input_index in range(1, len(offspring.genome[node_index]["inputs"])):
                    incoming_node = offspring.genome[node_index]["inputs"][input_index]
                    incoming_type = offspring.getNodeType(node_index, input_index=input_index, this_node=False)
                    ftn_type = offspring.getNodeType(node_index, input_index=input_index, this_node=True)

                    # check if it's an active node
                    if incoming_node <= C and incoming_node in half2:# and incoming_node not in half1:
                        # nodes <= C and are supposed to be active
                        if incoming_type != ftn_type or incoming_node not in half1:
                            #if the active node types don't match or if they're both not active
                            half1_activeC = [i for i in half1 if i<=C]
                            no_match = True
                            counter = 0
                            while no_match:
                                if len(half1_activeC) > 0:
                                    temp_node = random.choice(half1_activeC)
                                    temp_type = offspring.getNodeType(temp_node)
                                    if temp_type == ftn_type:
                                        # it is a match. add the node
                                        offspring.genome[node_index]["inputs"][input_index] = temp_node
                                        no_match=False
                                    else:
                                        counter += 1
                                    if counter > 3*len(half1_activeC):
                                        offspring.genome[node_index]["inputs"][input_index] = offspring.randmInput(node_index, ftn_type)
                                        no_match = False
                                    else:
                                        pass
                                else:
                                    offspring.genome[node_index]["inputs"][input_index] = offspring.randmInput(node_index, ftn_type)
                                    no_match = False
                        else:
                            #then the data types match AND both are active
                            pass

                    elif incoming_node <= C:
                        # any non active node less than C
                        if incoming_type != ftn_type:
                            offspring.genome[node_index]["inputs"][input_index] = offspring.randmInput(node_index, ftn_type)
                        else:
                            # they match
                            pass

                    else:
                        #they should match
                        if incoming_type != ftn_type:
                            print("ERROR!!!! THIS SHOULD BE A MATCH", flush=True)
                            print(node_index, C, incoming_node, flush=True)

            # now get output nodes
            for node_index in range(numMain, numMain+numOutputs):
                incoming_node = offspring.genome[node_index]
                incoming_type = offspring.getNodeType(node_index, this_node=False)
                ftn_type = offspring.getNodeType(node_index, this_node=True)

                # try and connect to an active node
                if incoming_node <= C:
                    if incoming_node not in half1 or incoming_type != ftn_type:
                        half1_activeC = [i for i in half1 if i<=C]
                        no_match = True
                        counter = 0
                        while no_match:
                            if len(half1_activeC) > 0:
                                temp_node = random.choice(half1_activeC)
                                temp_type = offspring.getNodeType(temp_node)
                                if temp_type == ftn_type:
                                    # it is a match. add the node
                                    offspring.genome[node_index] = temp_node
                                    no_match=False
                                else:
                                    counter += 1
                                if counter > 3*len(half1_activeC):
                                    offspring.genome[node_index] = offspring.randmInput(numMain, ftn_type, min_range=0)
                                    no_match = False
                                else:
                                    pass
                            else:
                                offspring.genome[node_index] = offspring.randmInput(numMain, ftn_type, min_range=0)
                                no_match = False
                    else:
                        pass
                else:
                    pass

            # sooo arguments are already crossed over early on, but maybe give new option for how to do it
            # (1) crossover the genes from the first half into the offspring
            # (2) go through each arg_gene and flip coin to crossover or not
            if arg_mate_type == 'single point':

                if C==C1:
                    half1arg = []
                    for node in range(C+1):
                        half1arg.extend(self.genome[node]["args"])
                    half1arg = set(half1arg)
                    for arg_index in half1arg:
                        offspring.arguments[arg_index] = self.arguments[arg_index]
                else:
                    # C==C2
                    half1arg = []
                    for node in range(C+1):
                        half1arg.extend(other.genome[node]["args"])
                    half1arg = set(half1arg)
                    for arg_index in half1arg:
                        offspring.arguments[arg_index] = other.arguments[arg_index]

            elif arg_mate_type == 'uniform':

                for arg_index in range(numArgs):
                    if random.random() <= .5:
                        # crossover args
                        if C==C1:
                            offspring.arguments[arg_index] = self.arguments[arg_index]
                        else:
                            #C==C2
                            offspring.arguments[arg_index] = other.arguments[arg_index]
                    else: #don't crossover
                        pass
            else: # there are no other crossover types
                pass

            offsprings.append(offspring)


            # Offspring2:
            if keep_both_offspring:

                # start building offspring
                if C1 < C2:
                    C = C1
                    offspring = deepcopy(self)
                    offspring.genome = other.genome[0:C+1] + self.genome[C+1:]
                    half1 = M2
                    half2 = M1

                else:
                    C = C2
                    offspring = deepcopy(other) #the second half of offspring will dominate the genome so keep all things half2
                    offspring.genome = self.genome[0:C+1] + other.genome[C+1:]
                    half1 = M1
                    half2 = M2

                for node_index in range(C+1, numMain):
                    for input_index in range(1, len(offspring.genome[node_index]["inputs"])):
                        incoming_node = offspring.genome[node_index]["inputs"][input_index]
                        incoming_type = offspring.getNodeType(node_index, input_index=input_index, this_node=False)
                        ftn_type = offspring.getNodeType(node_index, input_index=input_index, this_node=True)

                        # check if it's an active node
                        if incoming_node <= C and incoming_node in half2:
                            # nodes <= C and are supposed to be active
                            if incoming_type != ftn_type or incoming_node not in half1:
                                #if the active node types don't match or if they're both not active
                                half1_activeC = [i for i in half1 if i<=C]
                                no_match = True
                                counter = 0
                                while no_match:
                                    if len(half1_activeC) > 0:
                                        temp_node = random.choice(half1_activeC)
                                        temp_type = offspring.getNodeType(temp_node)
                                        if temp_type == ftn_type:
                                            # it is a match. add the node
                                            offspring.genome[node_index]["inputs"][input_index] = temp_node
                                            no_match=False
                                        else:
                                            counter += 1
                                        if counter > 3*len(half1_activeC):
                                            offspring.genome[node_index]["inputs"][input_index] = offspring.randmInput(node_index, ftn_type)
                                            no_match = False
                                        else:
                                            pass
                                    else:
                                        offspring.genome[node_index]["inputs"][input_index] = offspring.randmInput(node_index, ftn_type)
                                        no_match = False
                            else: #then the data types match AND both are active
                                pass

                        elif incoming_node <= C:
                            # any non active node less than C
                            if incoming_type != ftn_type:
                                offspring.genome[node_index]["inputs"][input_index] = offspring.randmInput(node_index, ftn_type)
                            else: # they match
                                pass

                        else: #they should match
                            if incoming_type != ftn_type:
                                print("ERROR!!!! THIS SHOULD BE A MATCH", flush=True)
                                print(node_index, C, incoming_node, flush=True)
                # now get output nodes
                for node_index in range(numMain, numMain+numOutputs):
                    incoming_node = offspring.genome[node_index]
                    incoming_type = offspring.getNodeType(node_index, this_node=False)
                    ftn_type = offspring.getNodeType(node_index, this_node=True)

                    # try and connect to an active node
                    if incoming_node <= C:
                        if incoming_node not in half1 or incoming_type != ftn_type:
                            half1_activeC = [i for i in half1 if i<=C]
                            no_match = True
                            counter = 0
                            while no_match:
                                if len(half1_activeC) > 0:
                                    temp_node = random.choice(half1_activeC)
                                    temp_type = offspring.getNodeType(temp_node)
                                    if temp_type == ftn_type:
                                        # it is a match. add the node
                                        offspring.genome[node_index] = temp_node
                                        no_match=False
                                    else:
                                        counter += 1
                                    if counter > 3*len(half1_activeC):
                                        offspring.genome[node_index] = offspring.randmInput(numMain, ftn_type, min_range=0)
                                        no_match = False
                                    else:
                                        pass
                                else:
                                    offspring.genome[node_index] = offspring.randmInput(numMain, ftn_type, min_range=0)
                                    no_match = False
                        else:
                            pass
                    else:
                        pass


                if arg_mate_type == 'single point':

                    if C==C2:#<-this is what changed from offspring1
                        half1arg = []
                        for node in range(C+1):
                            half1arg.extend(self.genome[node]["args"])
                        half1arg = set(half1arg)
                        for arg_index in half1arg:
                            offspring.arguments[arg_index] = self.arguments[arg_index]
                    else:
                        # C==C1
                        half1arg = []
                        for node in range(C+1):
                            half1arg.extend(other.genome[node]["args"])
                        half1arg = set(half1arg)
                        for arg_index in half1arg:
                            offspring.arguments[arg_index] = other.arguments[arg_index]

                elif arg_mate_type == 'uniform':

                    for arg_index in range(numArgs):
                        if random.random() <= .5:
                            # crossover args
                            if C==C2:
                                offspring.arguments[arg_index] = self.arguments[arg_index]
                            else:
                                #C==C1
                                offspring.arguments[arg_index] = other.arguments[arg_index]
                        else: #don't crossover
                            pass
                else: # there are no other crossover types
                    pass

                offsprings.append(offspring)

            return offsprings

        else:
            '''
            Don't do crossover or any mating
            '''
            offsprings = [self, other]
            return offsprings
    """