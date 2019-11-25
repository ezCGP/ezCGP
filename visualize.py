"""
This class takes in an individual and writes it to a .csv for draw.io to process

Use: Instantiate object, then pass an individual to create_csv
"""
import string
import random

header = '## Hello World \
            \n# label: %step%<br><i style="color:gray;">%text%</i> \
            \n# style: html=1;shape=rectangle;rounded=1;fillColor=%fill%;strokeColor=#000000 \
            \n# namespace: csvimport- \
            \n# connect: {\"from\":\"refs\", \"to\":\"id\", \"invert\":true, \"style\":\"curved=0;endArrow=blockThin;endFill=1;\"} \
            \n# width: auto \
            \n# height: auto \
            \n# padding: 10 \
            \n# ignore: id,fill,refs \
            \n# nodespacing: 10 \
            \n# levelspacing: 30 \
            \n# edgespacing: 40 \
            \n# layout: horizontalflow \
            \n## CSV starts under this line \
            \nid,step,text,fill,refs \
            \n '

class Visualizer():
    def __init__(self, individual, output_path='vis.csv'):
        # Limit on number of block is letters of alphabet
        self.individual = individual
        self.output_path = output_path
        self.shifts = list(string.ascii_lowercase)
        self.colors = ['#dae8fc', '#f8cecc', "#d5e8d4"] * 9

    def create_csv(self):
        csv_rows = header.split("\n")

        prev_output = ""
        for block_num in range(self.individual.num_blocks):
            block = self.individual.skeleton[block_num+1]
            curr_block = block["block_object"]

            print("sadsadsadasd", block)
            shift = self.shifts[block_num]
            color = self.colors[block_num]
            for active_node in curr_block.active_nodes:
                fn = curr_block[active_node]
                if active_node < 0:  #Input
                    layer_info = "batch_size Size= {} <br>n_epochs= {} <br>large_dataset= {} <br>nickname= {}".format(
                                    block['batch_size'], block['n_epochs'], block['large_dataset'],block['nickname'])
                    out = "{}{},{},\"{}\",{},\"{}\"".format(shift, active_node, fn, layer_info, color, prev_output)
                elif active_node >= curr_block.genome_main_count:
                    out = "{}{},Output,,{},\"{}\"".format(shift, active_node, color, shift + str(fn))
                    prev_output = shift + str(active_node)
                else:
                    out = "{}{},{},,{},\"{}\"".format(shift, active_node, fn['ftn'].__name__, color, ','.join(map(lambda x: shift + str(x), fn['inputs'])))
                csv_rows.append(out + "")

        csv_rows.append("END,\"Fitness: {}\",,{},\"{}\"".format(self.individual.fitness.values, '#ffe6cc', prev_output))

        with open(self.output_path, 'w') as csv:
            for row in csv_rows:
                csv.write(row + '\n')

