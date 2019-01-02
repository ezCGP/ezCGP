### individual.py

# external packages

# my scripts
from blocks import Block

class Individual(): # Block not inherited in...rather just instanciate to an attribute or something
    """
    Individual genome composed of blocks of smaller genome.
    Here we define those blocks into the full genome
    """
    def __init__(self, skeleton):
        # TODO eventually make it so we can read in an xml/json/yaml file
        # the user will be able to outline the individual.genome there
        # instead of having to edit this file every time
        
        #self.genome = [] #maybe a dict instead?
        """
        self.skeleton = {
            'input': [datatype, datatype],
            'output': [datatype],
            1: {**kwargs},
            2: {},
            3: {}
        }


        [
        block1 = {"inputs"=[inputs0,inputs1],"args","outputs"},
        block2 = {"inputs"=block1.outputs,"args","outputs"},
        output0 = block2.output[0],
        output1 = block2.output[1],
        inputs1,
        inputs0
        ]
        """
        self.skeleton = skeleton
        self.blocks = list(self.skeleton.keys())
        self.blocks.remove('input')
        self.blocks.remove('output')
        self.num_blocks = len(self.blocks)
        # verify that the number of blocks matches the dictionary
        for i in range(1,self.num_blocks+1):
            if i not in skeleton.keys():
                print("UserError: skeleton keys do not have a consistent count of blocks")
                exit()
            else:
                # now build out the block if it exists
                self.skeleton[i]["block_object"] = Block(**self.skeleton[i])
        """
        self.preprocessing = Block(
                                nickname="preprocessing",
                                ftn_dict={"ftnName": {"prob"1:}},
                                arg_dict={"argumentDataType": {"prob":1}},
                                mate_dict={"mateMethod": {"prob":1, "args":[]}},
                                mut_dict={"mutateMethod": {"prob":1, "args":[]}},
                                gene_dict={ftnName: {"inputs":[], "args":[], "output":_}},
                                block_inputs=[inputDataType, inputDataType],
                                block_outputs=[outputDataType, outputDataType],
                                block_main_count=num_main,
                                block_arg_count=num_args)
        """


    def evaluate(self, data):
        for i in range(1,self.num_blocks+1):
            self.skeleton[i]["block_object"].evaluate(block_inputs=data)
            data = self.skeleton[i]["block_object"].genome_output_values
        self.genome_outputs = data
        #return genome_outputs


    def score_fitness(self, labels):
        #self.fitness = 
        pass


    def mutate(self, block=None):
        if block is "random selection":
            roll = np.random.random()
            for i in range(1,self.num_blocks):
                if roll <= i/self.num_blocks:
                    self.skeleton[i]["block_object"].mutate()
                    break
                else:
                    continue
        elif block is None:
            for i in range(1,self.num_blocks):
                self.skeleton[i]['block_object'].mutate()
        else:
            self.skeleton[block]["block_object"].mutate()


    def mate(self, other, block=None):
        if block is "random selection":
            roll = np.random.random()
            for i in range(1,self.num_blocks):
                if roll <= i/self.num_blocks:
                    offspring_list = self.skeleton[i]["block_object"].mate(
                                            other.skeleton[i]["block_object"])
                    break
                else:
                    continue
        elif block is None:
            for i in range(1,self.num_blocks):
                offspring_list = self.skeleton[i]["block_object"].mate(
                                        other.skeleton[i]["block_object"])
        else:
            offspring_list = self.skeleton[i]["block_object"].mate(
                                    other.skeleton[i]["block_object"])
        return offspring_list