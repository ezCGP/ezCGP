#This is going to be the file that converts to csv and uploads to draw.io
'''
This class takes in an individual and writes it to a .csv for draw.io to process

Use: Instantiate object, then pass an individual to create_csv
'''
import numpy as np

# curr_block[active_node] = InputPlaceholder
# 


class Visualizer():
    def __init__(self):
        pass

    def create_csv(self, individual, output_path='vis.csv'):
        header = "## Hello World\n# label: %step%\n# style: shape=rectangle;rounded=1;fillColor=%fill%;strokeColor=#000000\n# namespace: csvimport-\n# connect: {\"from\":\"refs\", \"to\":\"id\", \"invert\":true, \"style\":\"curved=0;endArrow=blockThin;endFill=1;\"}\n# width: auto\n# height: auto\n# padding: 10\n# ignore: id,fill,refs\n# nodespacing: 10\n# levelspacing: 30\n# edgespacing: 40\n# layout: horizontalflow\n## CSV starts under this line\nid,step,fill, refs"
        csv_rows = []
        for x in header.split("\n"):
            csv_rows.append(x)
        csv_rows.append("")
        # csv_rows.append(header.split("\n"))
        shift = 'a'
        color = ['#dae8fc', '#f8cecc', '#ffe6cc']
        c = 0
        prev_output = ""
        for i in range(1,individual.num_blocks+1):
            row = []
            curr_block = individual.skeleton[i]["block_object"]
            arg_values = np.array(curr_block.args)
            for active_node in curr_block.active_nodes:
                fn = curr_block[active_node]
                id = str(active_node) + shift
                if active_node < 0:
                    if c > 0:
                        out = "{},{},{},\"{}\"".format(id,fn,color[c],prev_output)
                    else:
                        out = "{},{},{},".format(id,fn,color[c])
                elif active_node >= curr_block.genome_main_count:
                    out = "{},output,{},\"{}\"".format(id,color[c],str(fn)+shift)
                    prev_output = id
                else:
                    out = "{},{},{},\"{}\"".format(id, fn['ftn'].__name__, color[c], ','.join(map(str,[str(x) + shift for x in fn['inputs']])))
                # if 0 < active_node < curr_block.genome_main_count:
                    # out += ", {}".format( arg_values[fn['args']])
                csv_rows.append(out + "")
            c += 1
            shift = chr(ord(shift) + 1) 
            csv_rows.append("")
        self.write_file(csv_rows, output_path)

    def write_file(self, csv_rows, output_path):
        csv = open(output_path, 'w')
        #TODO label the columns and add colors and stuff idk
        for i, row in enumerate(csv_rows):
            for j, item in enumerate(row):
                csv.write(item)

            if i < len(csv_rows)-1:
                csv.write('\n')
        csv.close()


    def get_args(self, block):
        #TODO this should definitely be in the block itself
        args = '{}'.format(block.args)
        return args.replace(',','|')
