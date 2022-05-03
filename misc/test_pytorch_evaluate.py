'''
see how well i built the evaluate and primitives for simgan
'''

### packages
import torch

### sys relative to root dir
import sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(realpath(__file__))))

### absolute imports wrt root
from codes.block_definitions.evaluate.block_evaluate_pytorch import BlockEvaluate_PyTorch_Abstract
import codes.block_definitions.utilities.operators_pytorch as opPytorch


class SimGAN_Test(BlockEvaluate_PyTorch_Abstract):
    def __init__(self):
        print("initializing...")
        super().__init__()
        self.final_module_dicts.append({"module": opPytorch.flatten_layer,
                                        "args": []})
        self.final_module_dicts.append({"module": opPytorch.linear_layer,
                                        "args": [3]})
        self.final_module_dicts.append({"module": opPytorch.softmax_layer,
                                        "args": []})

    def build_graph(self, *args):
        print("building...")
        self.standard_build_graph(*args)

    def evaluate(self,
                 block_material,#: BlockMaterial,
                 block_def,#: BlockDefinition,
                 training_datalist,#: ezData,
                 validating_datalist,#: ezData,
                 supplements=None):
        print("evaluating...")
        self.build_graph(block_material, block_def, training_datalist)
        return self.graph


class BlockDef_Fake():
    def __init__(self, genome):
        self.genome_count = len(genome)
        for i, node in enumerate(genome):
            if type(node)==int:
                self.main_count = i
                break


class BlockMaterial_Fake():
    def __init__(self, genome, num_inputs=1):
        self.id = "Poop-69"
        self.genome = genome
        active_nodes = []
        last_node = genome[-(num_inputs+1)]
        assert(type(last_node)==int)
        active_nodes.append(last_node)
        for node_index in reversed(range(last_node+1)):
            if node_index in active_nodes:
                node_dict = self.genome[node_index]
                active_nodes+=node_dict["inputs"]

        self.active_nodes = sorted(list(set(active_nodes)))


    def __getitem__(self, node_index):
        return self.genome[node_index]


### Build a genome
genome = [None]*10
genome[0] = {"function": opPytorch.linear_layer,
             "inputs": [-1],
             "args": [69]}

genome[1] = {"function": opPytorch.linear_layer,
             "inputs": [-1],
             "args": [50]}

genome[2] = {"function": opPytorch.linear_layer,
             "inputs": [0],
             "args": [50]}

genome[5] = {"function": opPytorch.pytorch_concat,
             "inputs": [1, 2],
             "args": [1]}

genome[7] = {"function": opPytorch.conv1d_layer,
             "inputs": [5],
             "args": [6]}

genome[8] = 7

block_def = BlockDef_Fake(genome)
block_material = BlockMaterial_Fake(genome)


### Build Data
fake_data_X = torch.randn(512, 1, 92)
fake_data_Y = torch.randn(512, 3)


### Run Build + Forward
block_evaluate = SimGAN_Test()
graph = block_evaluate.evaluate(block_material, block_def, [fake_data_X], None, None)
output = graph(fake_data_X)
print(graph)
print("Done:", output.shape, graph.final_layer_shape)
print()


### Run Training
loss_fn = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.RMSprop(graph.parameters(),
                                lr=1e-3)
print(graph._modules['node_1'].bias) # value before training
y_pred = graph(fake_data_X)
loss = loss_fn(y_pred, fake_data_Y)
optimizer.zero_grad()
loss.backward()
optimizer.step()
print(graph._modules['node_1'].bias) # value after training