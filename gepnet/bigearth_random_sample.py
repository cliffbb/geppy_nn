from gepcore.utils import convolution, cell_graph
from gepcore.entity import Gene, Chromosome
from gepcore.symbol import PrimitiveSet
from fastai.vision import np, random, pickle
import argparse

# import evolutionary tools from DEAP and GEPPY
from deap import creator, base, tools
from geppy import Toolbox

parser = argparse.ArgumentParser(description='random search')
parser.add_argument('--head_length', type=int, default=2, help='gene head')
parser.add_argument('--num_genes', type=int, default=3, help='num of genes')
parser.add_argument('--pop_size', type=int, default=20, help='population')
parser.add_argument('--dir', type=str, default='mlj_experiments/3_2/seed_8')
parser.add_argument('--seed', type=int, default=328)
args = parser.parse_args()

# define primitive set
pset = PrimitiveSet('cnn')
# add cellular encoding program symbols
pset.add_program_symbol(cell_graph.end)
pset.add_program_symbol(cell_graph.seq)
pset.add_program_symbol(cell_graph.cpo)
pset.add_program_symbol(cell_graph.cpi)

# add convolutional operations symbols
conv_symbol = convolution.get_symbol()
pset.add_cell_symbol(conv_symbol.conv1x1)
pset.add_cell_symbol(conv_symbol.conv3x3)
pset.add_cell_symbol(conv_symbol.dwconv3x3)
#pset.add_cell_symbol(conv_symbol.conv1x3)
#pset.add_cell_symbol(conv_symbol.conv3x1)
#pset.add_cell_symbol(conv_symbol.maxpool3x3)

random.seed(args.seed)
np.random.seed(args.seed)

def random_sample():
    # define fitness and individual type
    creator.create('FitnessMax', base.Fitness, weights=(1,))
    creator.create('Individual', Chromosome, fitness=creator.FitnessMax)

    # create individuals (genotype space)
    toolbox = Toolbox()
    toolbox.register('gene', Gene, pset, args.head_length)
    toolbox.register('individual', creator.Individual, toolbox.gene, args.num_genes)
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)

    # translate individuals into computation graph (phenotype space)
    toolbox.register('comp_graph', cell_graph.generate_comp_graph)

    # sample population
    pop = toolbox.population(n=args.pop_size)

    # save population graphs
    for i, p in enumerate(pop):
        agraph, comp_graph = cell_graph.generate_comp_graph(p)
        cell_graph.save_graph(agraph, args.dir+'/indv_{}'.format(i))
        cell_graph.draw_graph(agraph, args.dir+'/indv_{}'.format(i))
        with open(args.dir+'/indv_{}/code.pkl'.format(i), 'wb') as f:
            pickle.dump(repr(p), f)


# start sampling process
if __name__=='__main__':
    random_sample()



