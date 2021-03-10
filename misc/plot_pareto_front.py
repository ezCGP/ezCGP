'''
take in a list of directories from which to grab the npz files for ploting
'''
### packages
import os
import glob
import matplotlib.pyplot as plt


### sys relative to root dir
import sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(realpath(__file__))))

### absolute imports wrt root
from post_process import plot_things


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--minimization",
                        action = "store_const",
                        const = True,
                        default = False,
                        help = "plot a maximization pareto front instead of minimization")
    parser.add_argument("-d",'--dirs',
                        nargs='+', # 1 or more
                        help='List directories for each universe run',
                        required=True)
    args = parser.parse_args()
    minimization = args.minimization

    fig, axis = plt.subplots(1, 1, figsize=(12,10))
    generation = 0
    for dir_path in args.dirs:
        npzs = glob.glob(os.path.join(dir_path, "gen*_fitness.npz"))
        for npz in npzs:
            print(npz)
            assert(os.path.exists(npz)), "npz doesn't exist"
            plot_things.plot_pareto_front_from_fitness_npz(axis,
                                                           npz,
                                                           minimization=args.minimization,
                                                           color=plot_things.matplotlib_colors[generation%len(plot_things.matplotlib_colors)],
                                                           label="Gen %2d" % generation,
                                                          )
            generation+=1

    plot_things.plot_legend(fig)
    plot_things.square_figure(fig)
    plt.savefig("temp.png")
    #plt.show()
