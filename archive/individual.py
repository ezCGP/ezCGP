### individual.py

# packages:
from collections import defaultdict
from copy import deepcopy
import random
import numpy as np
import os, sys
import gc

# other python scripts:
from configuration import *
from operators import *
from arguments import *
from problem import Problem
from evaluation_log import my_logging


# define the 'Individual' class below
class Individual(Problem):
    '''
    Each population will be populated by individual objects with class
    'Individual' which also inherits class 'Problem' and Fitness
    
    Define all mutation + mating, evaluation of output nodes, calculation of fitness
    and other evolutionary strategy components here.
    
    Anything specific to the problem should be defined in 'Problem.py'
    '''
    
    def __init__(self, argument_skeleton, numArgs):
        '''
        Initialize the 'parent' individual with a genome structure and
        then randomly fill it. NOTE the genome structure is a list of lists
        in this order:
        
        MAIN nodes -> OUTPUT nodes -> reversed INPUT nodes 
        '''
        
        # inherit the 'Problem' class
        super().__init__()

        # Arguments:
        self.numArgs = numArgs
        self.arguments = deepcopy(argument_skeleton)
        for arg_index in range(numArgs):
            self.arguments[arg_index].mutate()

        # Genome - Empty:
        self.genome = [None] * (numInputs + numMain + numOutputs)
        
        # Genome - Input Nodes
        for i in range(numInputs):
            #self.genome[-(i+1)] = self.inputs[i] #inputs in reverse
            self.genome[-(i+1)] = "inputs"
        # DON'T WANT TO COPY HAVE ALL INDIVIDUALS POINT TO SAME LOCATION...KEEP SIMPLE
        # if calling inputs, just do inputs[-i] or something


        # Genoem - Main Nodes
        for node_index in range(numMain):
            # create dictionary that will hold all of node's info
            temp_dict = {}

            # Function
            temp_dict["ftn"] = self.randmFtn()

            # Inputs
            inputs = []
            for input_type in operDict[temp_dict["ftn"]]["inputs"]:
                inputs.append(self.randmInput(node_index, input_type))
            temp_dict["inputs"] = inputs

            # Args
            args = []
            for arg_type in operDict[temp_dict["ftn"]]["args"]:
                args.append(self.randmArg(arg_type.__name__))
            temp_dict["args"] = args

            # save dictionary to node in genome
            self.genome[node_index] = temp_dict

        # Genome - Output Nodes
        for i in range(numOutputs):
            self.genome[numMain+i] = self.randmInput(numMain, operDict["output"][i], min_range=0)
            # ^min_range=0 to prevent output node referencing directly to an input node

        # Fitness() class inherited
        self.fitness = self.Fitness()

         
    
    def getNodeType(self, node_index, input_index=None, arg_index=None, this_node=True):
        '''
        Since the genome is self referencing but also contains different structures (main nodes vs input/output nodes),
        we created this function to simplify the process of getting the data types of the nodes.

        If node_index is only given, it returns the data type of the output of the operator of that node.
        If input_index is given and this_node=True, it returns the expected data type of that index for the operator of that node.
        If input_index is givenand this_node=False, it pulls the node_index that is stored in that gene, and returns the data type of
        the outout of the operator of that node_index.
        Similarly with arg_index.
        '''
        numArgs=self.numArgs

        # Input Nodes:
        if node_index < 0 :
            return operDict["input"][node_index]

        # Output Nodes:
        elif node_index >= numMain:
            # just get data type of output
            if this_node:
                return operDict["output"][node_index-numMain]
            # get the data type of the node that feeds into the output
            else:
                input_node = self.genome[node_index]
                if input_node < 0: #if it is an InputNode
                    return operDict["input"][input_node]
                else:
                    return operDict[self.genome[input_node]["ftn"]]["outputs"]

        # Main Nodes:
        elif input_index==None and arg_index==None:
            # dealing with main node and the function
            #so find the output
            return operDict[self.genome[node_index]["ftn"]]["outputs"]

        else:#if gene_index > 0:
            # main node and input-gene that feeds data to the function-gene
            if this_node:
                # get data type for this node and gene
                if input_index != None:
                    return operDict[self.genome[node_index]["ftn"]]["inputs"][input_index]
                elif arg_index != None:
                    return operDict[self.genome[node_index]["ftn"]]["args"][arg_index].__name__
                else:
                    print("ERROR. Change values fed into function...", flush=True)
                    for key in list(locals().keys()):
                        print(key, locals()[key])
                    exit()
            else:
                # get data type for the node that feeds to this one
                if input_index != None:
                    input_node = self.genome[node_index]["inputs"][input_index]
                    if input_node >= 0: # main node node input
                        return operDict[self.genome[input_node]["ftn"]]['outputs']
                    else: # input node
                        return operDict["input"][input_node]
                elif arg_index != None:
                    return operDict[self.genome[node_index]["ftn"]]["args"][arg_index].__name__
                else:
                    print("Error. Change values fed into function.", flush=True)
                    for key in list(locals().keys()):
                        print(key, locals()[key], flush=True)
                    exit()

 
    def checkTypes(self):
        '''
        METHOD FOR TESTING

        After an a new individual is built, mutated, or mated, we can run its genome through this
        to check every single node and gene to make sure that the incoming data type matches the
        expected data type as defined by the primitive operator of the node.

        This should be run when first introducing new operators, arguments, or changing the
        evolution process of an individual just to make sure no bug was introduced.
        We'll use 'check_types' from configuration.py to see whether this should be run everytime
        or not. If False, it will just return nothing.
        '''
        if check_types:
            numArgs=self.numArgs

            for node_index in range(0,numMain): # all Main Nodes
                #if type(self.genome[node_index]["inputs"])!=list:
                #    print("\n\n\n\n\n\nCHECKTYPE",self.genome[node_index]["inputs"], flush=True)
                #else:
                #    pass
                for input_index in range(len(self.genome[node_index]["inputs"])): # check inputs
                    ftn_type = operDict[self.genome[node_index]["ftn"]]["inputs"][input_index]
                    input_node = self.genome[node_index]["inputs"][input_index]
                    if input_node < 0:
                        input_type = operDict["input"][input_node]
                    else:
                        input_type = operDict[self.genome[input_node]["ftn"]]["outputs"]
                    if ftn_type != input_type:
                        print("GENOME CHECK ERROR inputs:", flush=True)
                        print("genome for node", node_index, flush=True)
                        print(self.genome[node_index], flush=True)
                        print("input index and node and parent node:", input_index, input_node, self.genome[input_node], flush=True)
                        print("need:", ftn_type, "got:", input_type, flush=True)
                        print("\n\n\n\n\n", flush=True)
                    else:
                        pass
                for arg_index in range(len(self.genome[node_index]["args"])): # check args
                    ftn_type = operDict[self.genome[node_index]["ftn"]]["args"][arg_index].__name__ #str
                    arg_node = self.genome[node_index]["args"][arg_index]
                    input_type = type(self.arguments[arg_node]).__name__ #str
                    if ftn_type != input_type:
                        print("GENOME CHECK ERROR args:", flush=True)
                        print("genome for node", node_index, flush=True)
                        print(self.genome[node_index], flush=True)
                        print("args index and node and value:", arg_index, arg_node, self.arguments[arg_node], flush=True)
                        print("need:", ftn_type, "got:", input_type, flush=True)
                        print("\n\n\n\n\n", flush=True)
                    else:
                        pass      


            for node_index in range(numMain, numMain+numOutputs):
                output_type = operDict["output"][node_index-numMain]
                input_node = self.genome[node_index]
                if input_node < 0:
                    input_type = operDict["input"][input_node]
                else:
                    input_type = operDict[self.genome[input_node]["ftn"]]["outputs"]
                if output_type != input_type:
                    print("GENOME CHECK ERROR output:", flush=True)
                    print("at output node:", node_index, "input node:", input_node, flush=True)
                    print("need:", output_type, "got:", input_type, flush=True)
                    print("\n\n\n\n\n", flush=True)
                else:
                    pass

        else:
            pass



    def randmFtn(self, previousFtn=None):
        '''
        Randomly choose an operator or function from the operDict dictionary object.
        
        Pass through previousFtn for the case when mutating (as opposed to initializing)
        to make sure the same function/operator isn't returned.
        This will also check to make sure that the input and output data types of the new
        function will match that of the previousFtn so that the node is still valid in
        terms of its ability to be evaluated.
        '''

        temp_count = 0
        temp_oper = list(operDict.keys())
        temp_oper.remove("output")
        temp_oper.remove("input")
        while True:

            # introduce operator bias for single_learner()
            if random.random() <= learner_bias:
                chosenFtn = single_learner
            else:
                chosenFtn = random.choice(temp_oper)
            if chosenFtn not in ["output", "input"]:
                # data types need to match the previousFtn input+output+args
                # BUT if 'None' then we are initializing so it doesn't matter
                if (previousFtn==None or
                    (operDict[chosenFtn]["inputs"]==operDict[previousFtn]["inputs"] and
                        operDict[chosenFtn]["outputs"]==operDict[previousFtn]["outputs"] and
                        operDict[chosenFtn]["args"]==operDict[previousFtn]["args"])):
                    if chosenFtn != previousFtn:
                        return chosenFtn
                    else:
                        temp_count+=1
                else:
                    if temp_count >= 2*len(operDict):#this count is overkill but shouldn't take too long anyways
                        return previousFtn
                        # no options for mutating
                    else:
                        temp_count+=1
            else:
                pass
    

    def randmArg(self, input_type, previousInput=None):
        '''
        Randomly choose an argument index that matches the expected data type
        '''
        numArgs=self.numArgs

        while True:
            chosenArg = random.randrange(numArgs)

            # get data type as a string
            arg_type = type(self.arguments[chosenArg]).__name__


            if arg_type == input_type and chosenArg != previousInput:
                # it is a match, so return it
                return chosenArg
            else: # keep looking
                pass
            

    
    def randmInput(self, index, input_type, previousInput=None, min_range=-numInputs):
        '''
        Randomly choose a node_index such that it falls in the range of 'min_range' and
        'index'.
        
        'min_range' is usually the negated number of inputs, but in the case of
        the current node being an output node, we don't want to link directly to an input
        node so min_range=0 in that case. 'index' is the node_index of the current node.
        Using 'index' as a roof for our random index insures that we don't create a cycle
        in the genome grid.
        
        The data types must also match, such that the required input ('input_type') for
        the current node is the same data type of the output for operator of the randomly
        generated node_index.
        
        And finally, we want to make sure that the previous node_index, 'previousInput',
        before mutation, is not randomly selected and returned.
        
        A peculiar thing has to be done for nodes toward the beginning of the genome. We
        have to identify the cases where a node has no possible mutation options because
        there is only one node with the proper data type, and that happens to be the same
        node in 'previousInput', so the 'while loop' will never end. This usually happens
        when there are only a few input nodes and if the current node_index is 0 or is
        close to 0
        '''
        
        while True:
            
            # check to make sure that more than one node with that outputtype exists
            if previousInput != None: # make sure this isn't an initialization step
                temp_count = 0
                for node in range(min_range, index):
                    temp_type = self.getNodeType(node, input_index=None, arg_index=None, this_node=True)
                    if temp_type == input_type:
                        temp_count += 1
                        if temp_count > 1:
                            # we are good. only need to check that there is more than 1
                            break
                        else:
                            pass
                    else:
                        pass
                
                if temp_count <= 1:
                    # so instead of returning a node_index, it'll return
                    #this string. so we have to check for that after it is returned.
                    #we pass this to make sure we don't count this as a mutation and
                    #we continue to try and mutate another node
                    return "skip mutation"
                else:
                    # we are good to mutate on this node
                    pass
            else:
                # if we are initializing, then we don't need to worry about this
                pass
            
            # randomly pick a node_index
            chosenInput = random.randrange(min_range, index)
            chosenOutputType = self.getNodeType(chosenInput)
            if chosenOutputType == input_type and chosenInput != previousInput:
                # data type matches. return index
                return chosenInput
            else:
                pass
    

    def reorder(self):
        '''
        CGP can limit the model's ability to connect to different nodes because of the directed
        acyclical nature of CGP. To encourage diversification of mutation possibilities, we can use this
        method to reshuffle nodes without  changing node behavior.
        
        The key to this method is to document how nodes are connected and dependent on each other,
        and to then iteratively move through the genome, from start to finish, until you find a node
        that is completely independent of anything after the iterating cursor. At that moment, we know
        we can move that node.
        
        The majority of this code is derived from the paper "Analysis of Cartesian Genetic Programming's
        Evolutionary Methods" by Brian W. Goldman. He created this 'reorder' function, and I adapted it
        to the way that I organize the genome.
        '''
        
        # create dictionaries such that for each node we'll note what other nodes feed into
        #the current node (depends_on), and what nodes do the current node feed into (feeds_to)
        depends_on = defaultdict(set)
        feeds_to = defaultdict(set)
        
        # fill the dictionaries from above
        for node_index in range(numMain):
            for input_index in self.genome[node_index]["inputs"]:
            #for gene in self.genome[node][1:]:
                #find the genes that feed into the node
                depends_on[node_index].add(input_index)
                feeds_to[input_index].add(node_index)
        
        # create a dictionary to document where the old and new node_indeces are
        # fill with the Input Nodes
        map_to_new_order = {inpt: inpt for inpt in range(-numInputs,0)}
        
        # now create a list that we'll update throug the process that are 'in the process'
        #of being mapped to a new index
        map_queue = list(map_to_new_order.keys())
        new_index = 0
        while map_queue:
            #random.seed(seed)
            map_this_one = random.choice(map_queue)
            map_queue.remove(map_this_one)
            
            # don't want to map an input location
            if map_this_one >= 0:
                map_to_new_order[map_this_one] = new_index
                new_index += 1
            else:
                pass
            
            # update what can now be mapped
            if map_this_one < numMain: #dont map an output node
                for add_to_queue in feeds_to[map_this_one]:
                    #print("feeds to check", map_this_one, feeds_to[map_this_one], add_to_queue)
                    depends_on[add_to_queue].remove(map_this_one)
                    if len(depends_on[add_to_queue]) == 0:
                        map_queue.append(add_to_queue)
                    else:
                        pass
            else:
                pass
        # mapping dictionary done
        # now assign the new mapping
        # first create a copy so it inherits all the same attributes
        new_genome = deepcopy(self.genome)
        
        # map the Main Nodes first
        for old_index in range(numMain):
            if old_index in map_to_new_order:
                # then a mapping exists for this node
                new_index = map_to_new_order[old_index]
                new_genome[new_index] = self.genome[old_index]
                
                # now the node has been moved, but we have to check and see if
                #any of the node_indices listed within that node need to also be mapped
                for input_index in range(len(new_genome[new_index]["inputs"])):
                    input_node = new_genome[new_index]["inputs"][input_index]
                    if input_node in map_to_new_order:
                        new_input = map_to_new_order[input_node]
                        new_genome[new_index]["inputs"][input_index] = new_input
                    else:
                        pass
            else:
                pass
        
        # map the Output Nodes now
        for output_node in range(numMain, numMain+numOutputs):
            old_input = self.genome[output_node]
            if old_input in map_to_new_order:
                new_input = map_to_new_order[old_input]
                new_genome[output_node] = new_input
            else:
                pass
        
        # done mapping old to new, so save new_genome to the proper genome
        self.genome = new_genome
        del new_genome
        
        # genome sequence did change so we will have to get a new active list
        #BUT since the actual process doesn't change node behavior, the fitness value
        #shouldn't have changed.
        #NOTE that evaluated list will be out of order but it should never be called at this point
        self.findActive()


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
            if len(M1) == 0 and len(M2) == 0:
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

    
    def mutate(self):
        '''
        Now that we have the functions to mutate operator genes and input genes,
        we can define the larger mutation process below.
        
        After a deep copy is created off a 'parent', we can mutate that copy and
        call it an offspring.
        
        NOTE the different mutation processes.
        
        SINGLE: doesn't mutate each gene with a fixed probability 'mut_rate',
        rather it picks a single gene at random. NOTE since our genes are tucked
        into lists within nodes that are in a list within genome, we have to get
        a count for total number of genes used in the Main Nodes and Output Nodes,
        to properly pick a gene at random. The process is not too clean imo but works.
        
        SKIP + ACCUMULATE: mutate each gene with probability 'mut_rate'
        
        NOTE that Main Nodes and Output Nodes are mutated seperately because
        their nodes are populated in different ways.

        After the genome is mutated, go to mutate the argument values
        '''

        # mutate the indices in the genome
        self.active_node_changed = False
        if duplicate == "skip":
            self.mutate_skip()
        elif duplicate == "accumulate":
            self.mutate_accumulate()
        else:
            # "single"
            self.mutate_single()

        # mutate the list of arguments
        #self.mutate_args()


    def mutate_skip(self):
        '''
        This is the standard way to mutate, in that each gene is mutated with probability mut_rate.

        If no active node is mutated then genome evaluation is skipped and the old fitness is passed.
        '''
        #mutated_args = set()
        #active_args = set()

        # only mutate Main+Output Nodes
        # start with main nodes
        for node_index in range(numMain):
            # start with the function gene
            if random.random() <= ftn_mut_rate:
                current_ftn = self.genome[node_index]["ftn"]
                self.genome[node_index]["ftn"] = self.randmFtn(current_ftn)

                # see if we changed an active node
                if node_index in self.active_ftns:
                    self.active_node_changed = True
                else:
                    pass
            else:
                pass

            # next mutate input genes
            for input_index in range(len(self.genome[node_index]["inputs"])):
                if random.random() <= input_mut_rate:
                    current_input = self.genome[node_index]["inputs"][input_index]
                    current_input_type = self.getNodeType(node_index, input_index=input_index, this_node=True)

                    new_input = self.randmInput(node_index, current_input_type, current_input)
                    if new_input != "skip mutation":
                        self.genome[node_index]["inputs"][input_index] = new_input

                        # check if active node
                        if node_index in self.active_inputs:
                            self.active_node_changed = True
                        else:
                            pass
                    else:
                        # don't mutate since no place to mutate to
                        pass
                else:
                    # don't mutate since coin flipped against it
                    pass

            # next mutate args
            for arg_index in range(len(self.genome[node_index]["args"])):

                # mutate arg indices
                if random.random() <= argIndex_mut_rate:
                    current_arg = self.genome[node_index]["args"][arg_index]
                    current_arg_type = self.getNodeType(node_index, arg_index=arg_index)

                    self.genome[node_index]["args"][arg_index] = self.randmArg(current_arg_type, current_arg)

                    # check if active
                    if current_arg in self.active_args:
                        self.active_node_changed = True
                    else:
                        pass
                else:
                    #don't mutate
                    pass

                # mutate arg values:
                arg = self.genome[node_index]["args"][arg_index]
                if random.random() <= argValue_mut_rate: #then mutate
                    self.arguments[arg].mutate()
                    if arg in self.active_args:
                        if type(self.arguments[arg]).__name__=="TriState" and streamData == False:
                            # then there is only one value for TriState...doesn't affect phenotype
                            pass
                        else:
                            self.active_node_changed = True
                    else: #not active
                        pass
                else: #don't mutate
                    pass


                #if node_index in self.active_args:
                #    active_args.add(arg)
                #    print("\n\nnode active, added to list\n", self.genome[node_index], "\n")
                #else:
                #    pass
                #if random.random() <= argValue_mut_rate:
                #    print("maybe mutate:", arg, type(self.arguments[arg]).__name__)
                #    print("mutated args:\n", mutated_args)
                #    print("active args:\n", active_args)
                #    if arg not in mutated_args:
                #        # hasn't already been mutated so mutate now
                #        self.arguments[arg].mutate()
                #        mutated_args.add(arg)
                #        print("haven't mutated this arg, so mutate and add\n", mutated_args)
                #    else: # don't mutate
                #        print("already mutated...")
                #        pass
                #    if arg in active_args:
                #        if streamData == False and type(self.arguments[arg]).__name__=="TriState":
                #            # don't say active node changed...
                #            print("active node mutated but tristate so no one cares")
                #            pass
                #        else:
                #            print("active node mutated!\n")
                #            self.active_node_changed = True
                #    else:
                #        pass
                #    wait = input("so we mutated...\n\n")
                #else:
                #    print("DONT mutate\n")
                #    pass

                        

        # finish with the Output Nodes
        for node_index in range(numMain, numMain+numOutputs):
            if random.random() <= input_mut_rate:
                current_node = self.genome[node_index]
                current_node_type = self.getNodeType(node_index, this_node=True)

                self.genome[node_index] = self.randmInput(numMain, current_node_type, current_node, min_range=0)
                self.active_node_changed = True #Output Node is always active
            else:
                pass


    def mutate_accumulate(self):
        '''
        This method continues to mutate_skip() until an active node is mutated.

        There are 2 options of who to return.
        1) if the mutant with the active node mutated, is dominated and strictly worse than the parent
        then the previous genertion of mutant without the mutated active node is returned
        2) else if the mutant is not dominated by the parent, that mutant is returned
        '''

        mutant = deepcopy(self)
        while mutant.active_node_changed==False:
            previous_mutant = deepcopy(mutant)
            mutant.mutate_skip()

        if self.fitness.dominates(mutant.fitness):
            # new mutant is strictly worse
            self = deepcopy(previous_mutant)
        else:
            self = deepcopy(mutant)
        del mutant

    def mutate_single(self):
        '''
        This mutation method randomly picks a gene one at a time until an active node is mutated.

        Since all the genes are built into dictionaries within a list, we have to count all possible
        mutatable genes so that we can apply the random number generator properly to pick the gene to
        mutate.
        '''
        # count all possible mutatable spaces
        gene_count = 0
        for node_index in range(numMain):
            gene_count += 1 #one function
            gene_count += len(self.genome[node_index]["inputs"]) # number of inputs
            gene_count += len(self.genome[node_index]["args"]) # number of arguments to mutate index
            gene_count += len(self.genome[node_index]["args"]) # and again for mutating values 
        gene_count += numOutputs #number of outputs

        while self.active_node_changed==False:

            #pick random gene to mutate
            mutate_pointer = random.randrange(1, gene_count+1) #points to a 'count' not an index so it starts at 1 not 0

            current_pointer = 0
            for node_index in range(numMain+numOutputs):

                # looking at a Main Node
                if node_index < numMain:

                    # check the function first
                    current_pointer += 1
                    
                    if current_pointer == mutate_pointer:
                        # mutate this function
                        current_ftn = self.genome[node_index]["ftn"]
                        self.genome[node_index]["ftn"] = self.randmFtn(current_ftn)

                        # see if we changed an active node
                        if node_index in self.active_ftns:
                            self.active_node_changed = True
                        else:
                            pass

                        # mutate_pointer found so break loop
                        break
                    else:
                        pass

                    current_pointer += len(self.genome[node_index]["inputs"])

                    if mutate_pointer <= current_pointer:
                        input_index = (len(self.genome[node_index]["inputs"]) - 1) - (current_pointer - mutate_pointer)
                        current_input = self.genome[node_index]["inputs"][input_index]
                        current_input_type = self.getNodeType(node_index, input_index=input_index, this_node=True)

                        new_input = self.randmInput(node_index, current_input_type, current_input)
                        if new_input != "skip mutation":
                            self.genome[node_index]["inputs"][input_index] = new_input

                            # check if active node
                            if node_index in self.active_inputs:
                                self.active_node_changed = True
                            else:
                                pass
                        else:
                            # don't mutate since no place to mutate to
                            pass

                        # mutate_pointer found
                        break
                    else:
                        pass
                    

                    current_pointer += len(self.genome[node_index]["args"])
                    
                    if mutate_pointer <= current_pointer: # mutate the arg INDEX
                        arg_index = (len(self.genome[node_index]["args"]) - 1) - (current_pointer - mutate_pointer)
                        current_arg = self.genome[node_index]["args"][arg_index]
                        current_arg_type = self.getNodeType(node_index, arg_index=arg_index)
                        self.genome[node_index]["args"][arg_index] = self.randmArg(current_arg_type, current_arg)

                        if current_arg in self.active_args:
                            self.active_node_changed = True
                        else:
                            pass
                        # muate pointer found
                        break
                    else:
                        pass
                        #haven't found, go to next node


                    current_pointer += len(self.genome[node_index]["args"])
                    
                    if mutate_pointer <= current_pointer: # mutate the arg VALUE
                        arg_index = (len(self.genome[node_index]["args"]) - 1) - (current_pointer - mutate_pointer)
                        arg = self.genome[node_index]["args"][arg_index]
                        if arg in self.active_args:
                            self.active_node_changed = True
                        else:
                            pass
                        # muate pointer found
                        break
                    else:
                        pass
                        #haven't found, go to next node

                else:
                    # node_index must be in output
                    node_index = (numOutputs - 1) - (gene_count - mutate_pointer) + numMain
                    current_node = self.genome[node_index]
                    current_node_type = self.getNodeType(node_index, this_node=True)

                    self.genome[node_index] = self.randmInput(numMain, current_node_type, current_node, min_range=0)
                    self.active_node_changed = True

                    # mutate pointer has to be in output == found
                    break



    def mutate_args(self): # CAN PROB DELETE
        '''
        The previous mutate functions change which node indices feed into the genes. This goes directly
        to the individual's argument list and mutates the values directly.

        Note that each argument class has a mutate() function defined directly inside the class so that
        handles all the actual mutation...just needs to be called.

        *still unsure how to handle active arguments*
        '''

        #self.active_node_changed = False # see to put this outside of the function

        for arg in self.arguments:
            if random.random() <= argValue_mut_rate:
                #the mutate
                arg.mutate()

                if arg in self.active_args:
                    self.active_index = True
                else:
                    #not active
                    pass
            else:
                #don't mutate
                pass


    def findActive(self):
        '''
        Start from the Output Nodes and identify all the nodes that actively contribute to
        the Output Node.
        
        Store the node_indices in a list and sort.

        Also store the argument indices and store those in self.active_args ...that list can be unsorted
        
        NOTE: the 'active' list includes both input and output nodes in addition to the active main nodes.


        NEW!!!!!
        4 things stored and found
        active_inputs: all node indices that are fed into the genome output (including input nodes)
        active_ftns: node indices that fall before and include the LAST single_learner in the genome (no genome-input node)
        active_args: all argument indices that are used by the ftns in active_ftns
        has_learned: Boolean for whether or not the individual learns or not

        if no single_learner, active_ftns and active_args will be empty but active_inputs will still include all

        '''

        # add ouptput node_indices to active set
        self.active_inputs = set(range(numMain, numMain+numOutputs))
        self.active_ftns = set()
        self.active_args = set()
        self.has_learned = False
        
        # add node_indices that feed to output nodes
        self.active_inputs.update(node_index for node_index in self.genome[numMain : numMain+numOutputs])
        for node_index in reversed(range(numMain)):
            if node_index in self.active_inputs:
                self.active_inputs.update(self.genome[node_index]["inputs"])
                if node_index >= 0:
                    if self.has_learned==True or self.genome[node_index]["ftn"].__name__=="single_learner":
                        self.has_learned = True
                        self.active_ftns.update([node_index])
                        if len(self.genome[node_index]["args"]) > 0:
                            self.active_args.update(self.genome[node_index]["args"])
                        else: #no args to add
                            pass
                    else: # no learner ftn yet so won't contribute to phenotype
                        pass
                else: #nothing to add if input node other than input
                    pass
            else: # not an active node, don't care
                pass


        #for node_index in self.genome[numMain : numMain+numOutputs]:
        #    if node_index >= 0:
        #        if learned == True or self.genome[node_index]["ftn"].__name__=="single_learner":
        #            learned = True
        #            self.active_ftns.update(node_index)
        #            if len(self.genome[node_index]["args"]) > 0:
        #                self.active_args.update(self.genome[node_index]["args"])
        #            else: #don't add args
        #                pass
        #        else: #don't add args or ftns
        #            pass
        #    else: # node index is an input node...that gets add from self.genome[node_index]["inputs"]
        #        pass


            #if node_index > 0 and len(self.genome[node_index]["args"]) > 0:
            #    self.active_args.update(self.genome[node_index]["args"])
            #else:
            #    pass

        #for node_index in reversed(range(numMain)):
        #    if node_index in self.active:
        #        self.active.update(self.genome[node_index]["inputs"])
        #        self.active_args.update(self.genome[node_index]["args"])
        #    else:
        #        pass
        self.active_inputs = sorted(list(self.active_inputs))
        self.active_ftns = sorted(list(self.active_ftns))
        self.active_args = sorted(list(self.active_args))


    class Fitness(object):
        '''
        the NSGA taken from deap requires a Fitness class to hold the values.
        so this attempts to recreate the bare minimums of that so that NSGA
        or (hopefully) any other deap mutli obj ftn handles this Individual class

        http://deap.readthedocs.io/en/master/api/base.html#fitness
        '''

        def __init__(self):
            self.values = () #empty tuple

        # check dominates
        def dominates(self, other):
            a = np.array(self.values)
            b = np.array(other.values)
            # 'self' must be at least as good as 'other' for all objective fnts (np.all(a>=b))
            # and strictly better in at least one (np.any(a>b))
            return np.any(a < b) and np.all(a <= b)






from psutil import virtual_memory # mem_limit = _inst.mem_limit/100.0*virtual_memory().total/(1024*1024)    #Puts in MB
#import resource # mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
from multiprocessing import Process, Manager
import time
import socket

def process_evaluator(ftn, inputs, args, return_dict):
    try:
        return_dict['result'] = ftn(*inputs, *args)
    except:
        return_dict['error'] = str(sys.exc_info()[1])

def ram_check():
    free = os.popen('free -m').read()
    words = free.split()
    ind = [i for i,x in enumerate(words) if x=="Swap:"][0]-1
    avail_ram = int(words[ind])
    return avail_ram


def evaluate(indv, global_inputs, memory_limit=memory_limit, time_limit=time_limit):
    '''
    Go through all active Main Nodes and evaluate the node after extracting
    the function/operator from gene_index=0, and after extracting the arguments
    or inputs from the remaining genes.
    
    Store the output for each node's evaluation in a list where the index of that
    evaluated list is the same as the respective node_index in the genome.
        
    NOTE: we will also store the Input Nodes and Output Nodes, and any inactive
    node will be filled with None. That way, the evaluated list retains the same
    length as genome and makes for convenient indexing.
    '''

        
    evaluated = [None] * (numInputs+numMain+numOutputs)
    #indv.has_learned = False
    indv.dead = False

    if compute == "scoop":
        eval_id = os.getpid()
        process_file = "/tmp/rt_%s_%s_0_Started" % (indv.id, eval_id)
        open(process_file, 'a').close()
    else:
        eval_id = "local"


    ### quickly check for a singler_learner operator...findActive() SHOULD BE DOING THIS NOW!
    #ftn_str = ""
    #final_learner = 0
    #for i, active_index in enumerate(indv.active):
    #    if active_index >= 0 and active_index < numMain:
    #        function = indv.genome[active_index]["ftn"]
    #        ftn_str += function.__name__ #returns str
    #        ftn_str += "\n"
    #        if function == single_learner:
    #            indv.has_learned = True
    #            final_learner = i
    #            #break
    #        else:
    #            pass
    #    else:
    #        pass

    if indv.has_learned:
        #indv.ftns = ""
        start_time = time.time()
            
        # Input Nodes
        for i in range(numInputs):
            evaluated[-(i+1)] = global_inputs[i] #make sure inputs isn't in reverse

        # Main Nodes
        for active_index in indv.active_ftns:#[:final_learner+1]:
            if active_index >= 0 and active_index < numMain:
                function = indv.genome[active_index]["ftn"]
                input_indices = indv.genome[active_index]["inputs"]
                    
                # follow the args_index in genome to find the evaluated output at those node_indeces
                inputs = []
                for i, input_index in enumerate(input_indices):
                    inputs.append(evaluated[input_index])

                    if check_types: 
                        got_type = type(evaluated[input_index])
                        if input_index < 0:
                            alleg_type = operDict["input"][input_index]
                        else:
                            alleg_type = operDict[indv.genome[input_index]["ftn"]]["outputs"]
                        need_type = operDict[function]["inputs"][i]
                        if got_type != need_type:
                            print("\n...EVALUATION ERROR...", flush=True)
                            print("at node", active_index, "expected data type", need_type, "for function", function, flush=True)
                            print("got fed input index", input_index, "with actual data type", got_type, "and supposed type", alleg_type, flush=True)
                            print("value:", evaluated[input_index], flush=True)
                            exit()
                        else:
                            pass
                    else: # don't checkTypes
                        pass
                    
                # now gather the args
                arg_indices = indv.genome[active_index]["args"]
                args = []
                for i, arg_index in enumerate(arg_indices):
                    args.append(indv.arguments[arg_index].value)

                    if check_types:
                        got_type = type(indv.arguments[arg_index]).__name__ #str
                        need_type = operDict[function]["args"][i].__name__
                        if got_type != need_type:
                            print("\n...EVALUATION ERROR...", flush=True)
                            print("at node", active_index, "expected arg type", need_type, "for function", function, flush=True)
                            print("got fed arg index", arg_index, "with actual arg type", got_type, flush=True)
                            print("value:", indv.arguments[arg_index].value, flush=True)
                            exit()
                        else:
                            pass
                    else:
                        pass



                    # check to see if it is a machine learner
                    #if type(indv.arguments[arg_index]).__name__=="LearnerType":
                    #    indv.has_learned = True #used when calc fitness to see if machine learning was used at least once to cluster
                    #else:
                    #    pass

                # check if there is a computational error
                if compute == "scoop":
                    # Jason's code on tracking RAM and time
                    socket.setdefaulttimeout(None) # what does this even do? https://docs.python.org/3/library/socket.html#socket.setdefaulttimeout
                    manager = Manager()
                    return_dict = manager.dict()
                    return_dict['result'] = []
                    return_dict['error'] = None
                    

                    # ram check to avoid Error12 cannot allocate memmory
                    mem_limit = min(memory_limit[0]/100.0*virtual_memory().total/(1024.*1024.), memory_limit[1])    #Puts in MB
                    count_loops = 0
                    while True:
                        #avail_ram = ram_check()
                        avail_ram = virtual_memory().available/(1024*1024)
                        if avail_ram > mem_limit + 100:
                            break
                        elif count_loops <= 5:
                            count_loops += 1
                            process_file = "/tmp/rt_%s_%s_11notEnoughRam_%s_%s" % (indv.id, eval_id, avail_ram, mem_limit)
                            open(process_file, 'a').close()
                            time.sleep(60)
                            gc.collect()
                        else: #loop count > 5
                            mem_limit = avail_ram - 100
                            break
                    my_process = Process(target=process_evaluator, args=(function, inputs, args, return_dict))

                    my_process.daemon = True
                    my_process.start()
                    #process_file = "/tmp/rt_%sstarted" % (my_process.pid)
                    #open(process_file, 'a').close()
                    #mem_limit = memory_limit/100.0*virtual_memory().total/(1024*1024)    #Puts in MB
                    #should_kill_mem = False
                    #should_kill_time = False

                    #ii = 0
                    while my_process.is_alive():
                        total_mem = int(os.popen('ps -p %d -o %s | tail -1' % (
                            my_process.pid, 'rss')).read()) / 1024.0  # KB
                        time.sleep(1)






                        # rodd testing
                        #ii+=1
                        process_file = "/tmp/rt_%s_%s_%s_0ProcessEval_%s_%s" % (indv.id, eval_id, my_process.pid, function, str(args))
                        open(process_file, 'a').close()
                        #with open(process_file, 'w') as p: #p = file(front_file, 'wb')
                        #    info = str(os.popen('ps -e -o uname,pid,pcpu,pmem,etime,comm --sort=-pcpu | head -15').read())
                        #    p.write(info)
                        #    p.write()
                            #pickle.dump(info, p)
                            #for i,line in enumerate(info):
                            #    pickle.dump(line, p)
                        #p.close()







                        if total_mem > mem_limit:
                            indv.dead = True
                            node_stat = [active_index, function.__name__, input_indices, args]
                            error_string = 'Memory usage ' + str( total_mem / (1024.0)) + ' GB exceeds ' + str(mem_limit / (1024.0)) +'GB'
                            indv.to_write = my_logging("mem overflow", [indv.to_write, node_stat, error_string])
                            process_file = "/tmp/rt_%s_%s_%s_1MemError" % (indv.id, eval_id, my_process.pid)
                            open(process_file, 'a').close()
                            my_process.terminate()

                        elif time.time() - start_time > time_limit:
                            indv.dead = True
                            node_stat = [active_index, function.__name__, input_indices, args]
                            error_string = 'Time usage ' + str( time.time() - start_time ) + ' exceeds ' + str(time_limit)
                            indv.to_write = my_logging("time overflow", [indv.to_write, node_stat, error_string])
                            process_file = "/tmp/rt_%s__%s_%s_1TmError" % (indv.id, eval_id, my_process.pid)
                            open(process_file, 'a').close()
                            my_process.terminate()

                        else: # no reason to kill indv yet
                            pass
                    # process has terminated
                    my_process.join()
                    if return_dict['error'] is not None: # died from function error
                        indv.dead = True
                        node_stat = [active_index, function.__name__, input_indices, args]
                        indv.to_write = my_logging("node died", [indv.to_write, node_stat, return_dict['error']])
                    elif indv.dead==False:
                        evaluated[active_index] = return_dict['result']
                        node_stat = [active_index, function.__name__, input_indices, args, evaluated[active_index].get_train_data().get_numpy().shape]
                        indv.to_write = my_logging("node", [indv.to_write, node_stat])
                    else: #indv died from mem or time usage
                        pass

                elif compute == "local":
                    try:
                        evaluated[active_index] = function(*inputs, *args)
                        node_stat = [active_index, function.__name__, input_indices, args, evaluated[active_index].get_train_data().get_numpy().shape]
                        indv.to_write = my_logging("node", [indv.to_write, node_stat])
                    except:
                        indv.dead = True
                        node_stat = [active_index, function.__name__, input_indices, args]
                        indv.to_write = my_logging("node died", [indv.to_write, node_stat, str(sys.exc_info()[1])])


                else:
                    pass
                #except:
                #    #this should never be reached
                #    process_file = "/tmp/rt_zz_WAIT_WHYAREYOUHERE_%s_%s_%s_%s_%s" % (indv.id, eval_id, function, str(args), str(sys.exc_info()[1]))
                #    open(process_file, 'a').close()

                #    indv.dead = True
                #    node_stat = [active_index, function.__name__, input_indices, args]
                #    indv.to_write = my_logging("node died", [indv.to_write, node_stat, str(sys.exc_info()[1])])
                #    break

                # if indv died, kill the for loop
                if indv.dead:
                    break
                else:
                    pass
            else:
                pass
    else: #indv.has_learned == False...dead
        pass
        
    # Output Nodes
    if indv.dead:
        if compute == "scoop":
            process_file = "/tmp/rt_%s_%s_z_died" % (indv.id, eval_id)
            open(process_file, 'a').close()
        else:
            pass
        indv.to_write = my_logging("individual died", [indv.to_write])
        final_evaluated = "dead"
        pass

    elif indv.has_learned == False:
        # get list of ftns used
        ftn_str = ""
        for active_index in indv.active_inputs:
            if active_index >= 0 and active_index < numMain:
                function = indv.genome[active_index]["ftn"]
                ftn_str += function.__name__ #returns str
                ftn_str += "\n"
            else:
                pass
        if compute == "scoop":
            process_file = "/tmp/rt_%s_%s_z_noLearn" % (indv.id, eval_id)
            open(process_file, 'a').close()
        else:
            pass
        indv.to_write = my_logging("individual no learner", [indv.to_write, ftn_str])
        final_evaluated = "dead_noLearn"
        #pass

    else:
        for output_node in range(numMain, numMain+numOutputs):
            if compute == "scoop":
                process_file = "/tmp/rt_%s_%s_z_survive" % (indv.id, eval_id)
                open(process_file, 'a').close()
            else:
                pass
            #evaluated[output_node] = evaluated[indv.active[final_learner]]#[indv.genome[output_node]]
            #final_evaluated = evaluated[indv.active[final_learner]]
            final_evaluated = evaluated[max(indv.active_ftns)]
            indv.to_write = my_logging("individual survived", [indv.to_write])


    # now get fitness
    del evaluated
    indv.getFitness(global_inputs, final_evaluated)
    del final_evaluated
    gc.collect()
    return indv
    

def testEvaluate(indv, global_inputs): # NEEED TO CLEAN THIS UP
    '''
        Similar to Evaluate() but it takes in the test data instead of train data
    '''

    evaluated = [None] * (numInputs+numMain+numOutputs)
    indv.to_write = "FINAL EVALUATION -- TESTING OF MODEL\n"

    # Input Nodes
    for i in range(numInputs):
        evaluated[-(i+1)] = global_inputs[i] #make sure inputs isn't in reverse

    # Main Nodes
    for active_index in indv.active_ftns:#[:final_learner+1]:
        if active_index >= 0 and active_index < numMain:
            # get function
            function = indv.genome[active_index]["ftn"]
            
            # get inputs
            input_indices = indv.genome[active_index]["inputs"]
            inputs = []
            for input_index in input_indices:
                inputs.append(evaluated[input_index])
                    
            # now gather the args
            arg_indices = indv.genome[active_index]["args"]
            args = []
            for arg_index in arg_indices:
                args.append(indv.arguments[arg_index].value)

            # now evaluate
            if compute == "scoop":
                # Jason's code on tracking RAM and time
                socket.setdefaulttimeout(None) # what does this even do? https://docs.python.org/3/library/socket.html#socket.setdefaulttimeout
                manager = Manager()
                return_dict = manager.dict()
                return_dict['result'] = []
                return_dict['error'] = None
                
                # ram check to avoid Error12 cannot allocate memmory
                mem_limit = min(memory_limit[0]/100.0*virtual_memory().total/(1024*1024), memory_limit[1])    #Puts in MB
                while True:
                    #avail_ram = ram_check()
                    avail_ram = virtual_memory().available/(1024*1024)
                    if avail_ram > mem_limit + 100:
                        break
                    else:
                        time.sleep(60)
                        gc.collect()
                my_process = Process(target=process_evaluator, args=(function, inputs, args, return_dict))
                my_process.daemon = True
                my_process.start()

                # DON'T INCLUDE ANY REASON TO KILL INDIVIDUAL

                # process has terminated
                my_process.join()
                if return_dict['error'] is not None: # died from function error
                    indv.dead = True
                    node_stat = [active_index, function.__name__, input_indices, args]
                    indv.to_write = my_logging("node died", [indv.to_write, node_stat, return_dict['error']])
                elif indv.dead==False:
                    evaluated[active_index] = return_dict['result']
                    node_stat = [active_index, function.__name__, input_indices, args, evaluated[active_index].get_train_data().get_numpy().shape]
                    indv.to_write = my_logging("node", [indv.to_write, node_stat])
                else: #indv died from mem or time usage
                    pass

            elif compute == "local":
                try:
                    evaluated[active_index] = function(*inputs, *args)
                    node_stat = [active_index, function.__name__, input_indices, args, evaluated[active_index].get_train_data().get_numpy().shape]
                    indv.to_write = my_logging("node", [indv.to_write, node_stat])
                except:
                    indv.dead = True
                    node_stat = [active_index, function.__name__, input_indices, args]
                    indv.to_write = my_logging("node died", [indv.to_write, node_stat, str(sys.exc_info()[1])])

            else:
                pass

            # if indv died, kill the for loop
            if indv.dead:
                break
            else:
                pass
        else:
            pass

        
    # Output Nodes
    if indv.dead:
        indv.to_write = my_logging("individual died", [indv.to_write])
        final_evaluated = "dead"

    else:
        for output_node in range(numMain, numMain+numOutputs):
            final_evaluated = evaluated[max(indv.active_ftns)]
            indv.to_write = my_logging("individual survived", [indv.to_write])


    # now get fitness
    del evaluated
    indv.getFitness(global_inputs, final_evaluated)
    del final_evaluated
    gc.collect()
    return indv.fitness.values, indv.to_write