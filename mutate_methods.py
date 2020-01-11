### mutate_methods.py

# external packages
import numpy as np

# my scripts
from genome import Genome

class Mutate(Genome):

    def __init__(self, genome_input_dtypes, genome_output_dtypes,
                       genome_main_count, genome_arg_count):
        Genome.__init__(self, genome_input_dtypes, genome_output_dtypes,
                              genome_main_count, genome_arg_count)
        self.mut_methods = []
        self.mut_weights = []


    def mutate_singleGene(self):
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
            gene_count += len(self[node_index]["inputs"]) # number of inputs
            gene_count += len(self[node_index]["args"]) # number of arguments to mutate index
            gene_count += len(self[node_index]["args"]) # and again for mutating values
        gene_count += numOutputs #number of outputs
        self.active_node_unchanged = True
        while self.active_node_unchanged:
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
                        current_ftn = self[node_index]["ftn"]
                        self[node_index]["ftn"] = self.randomFtn(only_one=True, exclude=[current_ftn], \
                                                                 output_dtype=self.operator_dict[current_ftn]['outputs'])
                        # see if we changed an active node
                        if node_index in self.active_nodes:
                            self.active_node_unchanged = False
                        else:
                            pass
                        # mutate_pointer found so break loop
                        break
                    else:
                        pass

                    current_pointer += len(self[node_index]["inputs"])
                    if mutate_pointer <= current_pointer:
                        input_index = (len(self[node_index]["inputs"]) - 1) - (current_pointer - mutate_pointer)
                        current_input = self.genome[node_index]["inputs"][input_index]
                        input_dtypes = self.getNodeType(node_index, input_dtype=True)
                        new_input = self.randomInput(dtype=input_dtypes[input_index], max_=node_index, exclude=[current_input])
                        self[node_index]["inputs"][input_index] = new_input
                        # check if active node
                        if node_index in self.active_nodes:
                            self.active_node_unchanged = False
                        else:
                            pass
                        # mutate_pointer found
                        break
                    else:
                        pass

                    current_pointer += len(self[node_index]["args"])
                    if mutate_pointer <= current_pointer: # mutate the arg INDEX
                        arg_index = (len(self[node_index]["args"]) - 1) - (current_pointer - mutate_pointer)
                        current_arg = self[node_index]["args"][arg_index]
                        arg_dtypes = self.getNodeType(node_index, arg_dtype=True)
                        self[node_index]["args"][arg_index] = self.randmArg(dtype=arg_dtypes[arg_index], exclude=[current_arg])
                        if node_index in self.active_nodes:
                            self.active_node_unchanged = False
                        else:
                            pass
                        # muate pointer found
                        break
                    else:
                        pass

                    current_pointer += len(self[node_index]["args"])
                    if mutate_pointer <= current_pointer: # mutate the arg VALUE
                        arg_index = (len(self[node_index]["args"]) - 1) - (current_pointer - mutate_pointer)
                        current_arg = self[node_index]["args"][arg_index]
                        self.mutate_argType(current_arg)
                        if current_arg in self.active_args:
                            self.active_node_unchanged = False
                        else:
                            pass
                        # muate pointer found
                        break
                    else:
                        pass

                else:
                    # node_index must be in output
                    node_index = (numOutputs - 1) - (gene_count - mutate_pointer) + numMain
                    current_node = self[node_index]
                    node_dtype = self.getNodeType(node_index)
                    self[node_index] = self.randomInput(dtype=node_dtype, min_=0, max_=self.genome_main_count, exclude=[current_node])
                    self.active_node_unchanged = False
                    # mutate pointer has to be in output == found
                    break


    def mutate_singleArgIndex(self):
        if len(self.active_args) == 0:
            pass
        else:
            # pick a different arg index value
            choices = []
            for node_index in range(self.genome_main_count):
                if len(self[node_index]["args"]) > 0:
                    choices.append(node_index)
            if len(choices) == 0:
                pass
            else:
                self.active_node_unchanged = True
                while self.active_node_unchanged:
                    node_index = np.random.choice(a=choices)
                    arg_index = np.random.choice(a=np.arange(len(self[node_index]["args"])))
                    current_arg = self[node_index]["args"][arg_index]
                    arg_dtypes = self.getNodeType(node_index, arg_dtype=True)
                    new_arg = self.randomArg(dtype=arg_dtypes[arg_index], exclude=[current_arg])
                    self[node_index]["args"][arg_index] = new_arg
                    if node_index in self.active_nodes:
                        self.active_node_unchanged = False
                    else:
                        pass



    def mutate_singleArgValue(self):
        # make sure that there are active_args
        if len(self.active_args) == 0:
            pass
        else:
            choices = np.arange(self.args_count)
            self.active_node_unchanged = True
            while self.active_node_unchanged:
                arg_index = np.random.choice(a=choices)
                self.args[arg_index].mutate()
                if arg_index in self.active_args:
                    self.active_node_unchanged = False
                else:
                    pass


    def mutate_singleInput(self):
        if self.genome_main_count == 1:
            return # cannot swap gene if only one
        choices = np.arange(self.genome_main_count+self.genome_output_count)
        self.active_node_unchanged = True
        while self.active_node_unchanged:
            node_index = np.random.choice(a=choices)
            if node_index < self.genome_main_count:
                input_index = np.random.choice(a=np.arange(len(self[node_index]["inputs"])))
                current_input = self[node_index]["inputs"][input_index]
                input_dtypes = self.getNodeType(node_index, input_dtype=True)
                new_input = self.randomInput(dtype=input_dtypes[input_index], max_=node_index, exclude=[current_input])
                if new_input is not None:
                    self[node_index]["inputs"][input_index] = new_input
                else: # cant mutate this node input
                    continue
                if node_index in self.active_nodes:
                    self.active_node_unchanged = False
                else:
                    pass
            else: #output node
                current_node = self[node_index]
                node_dtype = self.getNodeType(node_index)
                new_node = self.randomInput(dtype=node_dtype, min_=0, max_=self.genome_main_count, exclude=[current_node])
                if new_node is not None:
                    self[node_index] = new_node
                else:
                    continue
                self.active_node_unchanged = False


    def mutate_singleFtn(self):
        """
        if we mutate a function, then we have to verify that all the args and inputs also match
        """
        choices = np.arange(self.genome_main_count)
        self.active_node_unchanged = True
        while self.active_node_unchanged:
            node_index = np.random.choice(a=choices)
            current_ftn = self[node_index]["ftn"]
            self[node_index]["ftn"] = self.randomFtn(only_one=True, exclude=[current_ftn], \
                                                     output_dtype=self.operator_dict[current_ftn]['outputs'])

            # get all inputs to match new required datatype
            # find which already connected inputs match with the required datatypes
            required_dtypes = self.getNodeType(node_index, input_dtype=True)
            new_inputs = [None]*len(required_dtypes)
            for input_index, input_node in enumerate(self[node_index]["inputs"]):
                current_dtype = self.getNodeType(input_node, output_dtype=True)
                for i, dtype in enumerate(required_dtypes):
                    if (current_dtype==dtype) and (new_inputs[i] is None):
                        new_inputs[i] = input_node

            # now see which input node was not filled in
            for i, dtype in enumerate(required_dtypes):
                if new_inputs[i] is None:
                    new_inputs[i] = self.randomInput(dtype=dtype, max_=node_index, exclude=None)
            self[node_index]["inputs"] = new_inputs

            # same for the args
            required_dtypes = self.getNodeType(node_index, arg_dtype=True)
            new_args = [None]*len(required_dtypes)
            for arg_index, arg_node in enumerate(self[node_index]["args"]):
                current_dtype = type(self.args[arg_node]).__name__
                for i, dtype in enumerate(required_dtypes):
                    if (current_dtype==dtype) and (new_args[i] is None):
                        new_args[i] = arg_node

            # which has not been filled
            for i, dtype in enumerate(required_dtypes):
                if new_args[i] is None:
                    new_args[i] = self.randomArg(dtype=dtype, exclude=None)
            self[node_index]["args"] = new_args

            # check if it's active
            if node_index in self.active_nodes:
                self.active_node_unchanged = False
            else:
                pass

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
            del previous_mutant
        else:
            self = deepcopy(mutant)
        del mutant

    def mutate_singlegene(self):
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

                    # check the inputs
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

                    # check the argument connections
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

                    # check the argument values
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
