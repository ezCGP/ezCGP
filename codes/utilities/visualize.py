"""
This class takes in an individual and writes it to a .csv for draw.io to process

Use:
    Instantiate object, then pass an individual to create_csv

Make Viz:
    Go to draw.io or app.diagrams.net
    Navigate to Arrange->Inser->Advanced->CSV...
    Delete all contents in the pop-up text box
    Copy-Paste all contents from the saved csv into the text box
    Select 'Import'
"""
import glob
import os
import pickle
import random
import string
import numpy as np


HEADER = '## Hello World \
            \n# label: %step%<br><i style="color:gray;">%text%</i> \
            \n# style: html=1;shape=rectangle;rounded=1;fillColor=%fill%;strokeColor=%fill% \
            \n# namespace: csvimport- \
            \n# connect: {\"from\":\"refs\", \"to\":\"id\", \"invert\":true, \"style\":\"curved=0;endArrow=blockThin;endFill=1;fillColor=#1500ff;strokeColor=#1500ff;\"} \
            \n# width: auto \
            \n# height: auto \
            \n# padding: 10 \
            \n# ignore: id,fill,refs \
            \n# nodespacing: 10 \
            \n# levelspacing: 30 \
            \n# edgespacing: 40 \
            \n# layout: horizontalflow \
            \n## CSV starts under this line \
            \nid,step,text,fill,refs,arrow_color \
            \n '
INPUT_ROW = '{}{}{},{},\"nickname = {}\",{},\"{}\",{}'
NORMAL_ROW = '{}{}{},{},\"node #{}\",{},\"{}\",{}'
OUTPUT_ROW = '{0}{1}{2},Output,,{3},\"{0}{1}{4}\",{5}'
END_ROW = 'END,\"Fitness: ({})\",,#ffe6cc,\"{}\",'
INACTIVE_NODE_COLOR = '#f8cecc'


class Visualizer:
    def __init__(self, output_path):
        # Limit on number of block is letters of alphabet
        self.output_path = output_path
        self.shifts = list(string.ascii_lowercase)
        self.colors = ['#dae8fc', '#f8cecc', "#d5e8d4"] * 9
        self.header = HEADER
        self.arrow_color = '#1500ff'
        #self.csv_rows = self.header.split('\n')
        self.individual_num = 0


    def visualize(self, individual_material, individual_def, individual_name, print_entire_genome):
        self.individual_num += 1
        self.csv_rows = self.header.split('\n')
        prev_output = ''
        for block_num, (block_material, block_def) in enumerate(zip(individual_material.blocks, individual_def.block_defs)):
            print("\nBlock %i - %s" % (block_num, block_def.nickname))
            shift = self.shifts[block_num]
            indices = block_material.active_nodes
            if print_entire_genome:
                # Rodd is unsure this is right...
                indices = range(indices[0], len(block.genome) + indices[0])
            
            for index in indices:
                color = self.colors[block_num] if index in block_material.active_nodes else INACTIVE_NODE_COLOR
                node_dict = block_material.genome[index]
                print("Index %i with contents %s" % (index, node_dict))
                if index < 0:  # Input
                    print("...is genome input")
                    out = INPUT_ROW.format(self.individual_num, shift, index, node_dict, block_material.block_nickname, color, prev_output, self.arrow_color)
                elif type(node_dict) == np.int64:
                    # ^has to be np.int64, won't work for int
                    print("...is genome output")
                    out = OUTPUT_ROW.format(self.individual_num, shift, index, color, node_dict, self.arrow_color)
                    prev_output = f'{self.individual_num}{shift}{index}'
                else:
                    try:
                        ftn = node_dict['ftn']
                        ftn_name = ftn.__name__
                        arg_txt = []
                        for arg_type, arg_index in zip(block_def.operator_dict[ftn]["args"], node_dict["args"]):
                            arg_name = arg_type.__name__
                            arg_val = block_material.args[arg_index].value
                            arg_txt.append(f'{arg_name}: {arg_val}')

                    except Exception as err:
                        print("Hit an error when trying to parse the node.\n%s" % err)
                        import pdb; pdb.set_trace()
                    
                    # now combine items and format
                    arg_txt = '<br>'.join(arg_txt)
                    index_n_args = "{}<br>{}".format(index, arg_txt)
                    inputs = ','.join([f'{self.individual_num}{shift}{x}' for x in node_dict['inputs']])
                    out = NORMAL_ROW.format(self.individual_num, shift, index, ftn_name, index_n_args, color, inputs, self.arrow_color)
        
                # add index info to list to be eventually written out
                self.csv_rows.append(out + "")
        
        # make fitness scores into list of strings
        scores = [str(x) for x in individual_material.fitness.values]
        # now join into single string
        scores = ', '.join(scores)
        self.csv_rows.append(END_ROW.format(scores, prev_output))
        self.write_csv(individual_name)


    def write_csv(self, indiv_id):
        with open(os.path.join(self.output_path, "%s_viz.csv" % indiv_id), 'w') as f:
            for row in self.csv_rows:
                f.write(row + '\n')


if __name__ == '__main__':
    import sys
    import argparse
    parser = argparse.ArgumentParser(description="Given the output path for a given universe, this script will attempt "
                                                 "to visualize any pickled individual's genetic material via draw.io")
    parser.add_argument('--output_universe', '-o', help='File path to the parent folder that stores the pickled individuals')
    parser.add_argument('--individual', '-in', help='Base name of the individual that you want to visualize. If not provided, '
                                               'all individuals are visualized.',
                                               nargs='?',
                                               default='*.pkl')
    parser.add_argument('-i', help='Show inactive nodes.',
                              action='store_true')
    args = parser.parse_args()

    
    # Check validitiy of output folder
    output_folder = args.output_universe
    assert(os.path.isdir(output_folder)), "Given Output Universe folder does not exist or is not a directory: %s" % output_folder
    if output_folder.endswith("/"):
        # will help later when we do os.path.dirname if we remove the trailing '/' to get parent of this dir
        output_folder = output_folder[:-1]

    # Load our visualizing class defined above
    viz = Visualizer(output_folder)

    # Look for pickled Problem instance.
    # In main.py (where we pickle.dump the Problem instance) we import our specific problem file after we add 'problems' to the
    # sys.path list; so if we want to load the pickled Problem instance, we have to similarly add 'problems' to our sys.path or
    # else the load will fail
    sys.path.append("../../problems")
    pickled_problem = os.path.join(os.path.dirname(output_folder), "problem_def.pkl")
    assert(os.path.exists(pickled_problem)), "Couldn't find the pickled problem_def in the parent dir: %s" % pickled_problem
    with open(pickled_problem, 'rb') as f:
        problem = pickle.load(f)
    
    # What we really want from the Problem instance is the IndividualDefinition
    individual_def = problem.indiv_def

    # Look for pickled Individuals
    individuals = glob.glob(os.path.join(output_folder, args.individual))
    print("\nWe found %i pickled individuals in the given output directory" % len(individuals))
    arbitrary_max_count = 20
    if len(individuals) > arbitrary_max_count:
        print("...going to only look at %i individuals though" % arbitrary_max_count)
        individuals = individuals[-1*arbitrary_max_count:]

    for individual in individuals:
        individual_basename = os.path.basename(individual)[:-4] # strip ".pkl"
        print("\nVisualizing %s" % individual_basename)
        with open(individual, 'rb') as f:
            individual_material = pickle.load(f)
            viz.visualize(individual_material, individual_def, individual_basename, args.i)
        print("...done")

    print("\nCSV successfully created!")
