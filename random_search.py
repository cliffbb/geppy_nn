from gepcore.utils import convolution, cell_graph
from gepcore.entity import Gene, Chromosome
from gepcore.symbol import PrimitiveSet
# from gepcore.operators import *
from gepnet.model_v2 import get_gepnet, arch_config
# from gepnet.utils import count_parameters
# from scipy.special import expit
from fastai.vision import *
#from fastai.callbacks import SaveModelCallback
# from fastai.utils.mod_display import *
import argparse

# import evolutionary tools from DEAP and GEPPY
from deap import creator, base, tools
from geppy import Toolbox

parser = argparse.ArgumentParser(description='random search')
# evolutionary algorithm hyperparameter
parser.add_argument('--head_length', type=int, default=2, help='gene head')
parser.add_argument('--num_genes', type=int, default=4, help='num of genes')
parser.add_argument('--pop_size', type=int, default=10, help='population')
parser.add_argument('--hof', type=int, default=2, help='best individuals')
parser.add_argument('--dir', type=str, default='mlj_experiments')

# architecture config
parser.add_argument('--depth_coeff', type=float, default=1.0, help='layer scalar')
parser.add_argument('--width_coeff', type=float, default=1.0, help='channel scalar')
parser.add_argument('--channels', type=int, default=16, help='initial out channels')
parser.add_argument('--repeat_list', type=list, default=[1, 1, 1, 1], help='cells repetitions list')
parser.add_argument('--classes', type=int, default=45, help='num of labels')

# training search architecture
parser.add_argument('--seed', type=int, default=300, help='training seed')
parser.add_argument('--max_lr', type=float, default=3e-3, help='max learning rate')
parser.add_argument('--wd', type=float, default=4e-4, help='weight decay')
parser.add_argument('--epochs', type=int, default=2, help='training epochs')
parser.add_argument('--bs', type=int, default=128, help='batch size')

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


def random_search():
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

    # evaluation function
    def evaluate(indv):
        _, comp_graph = toolbox.comp_graph(indv)
        net = build_model(comp_graph)
        acc = train_model(net)
        fit = acc[1].item()
        return fit,
    toolbox.register('evaluate', evaluate)

    # population size and best individual
    pop = toolbox.population(n=args.pop_size)
    hof = tools.HallOfFame(args.hof)

    # simple_random algorithm
    pop, log = simple_random(pop, toolbox, hall_of_fame=hof, verbose=False)

    # save and draw best individual
    for i, best in enumerate(hof):
        agraph, comp_graph = cell_graph.generate_comp_graph(best)
        cell_graph.save_graph(agraph, args.dir+'/comp_graphs/best/indv_{}'.format(i))
        cell_graph.draw_graph(agraph, args.dir+'/comp_graphs/best/indv_{}'.format(i))
        with open(args.dir+'/comp_graphs/indv_{}/code.pkl'.format(i), 'wb') as f:
            pickle.dump(repr(best), f)

    # save population graphs
    for i, p in enumerate(pop):
        agraph, comp_graph = cell_graph.generate_comp_graph(p)
        cell_graph.save_graph(agraph, args.dir+'/comp_graphs/pop/indv_{}'.format(i))
        cell_graph.draw_graph(agraph, args.dir+'/comp_graphs/pop/indv_{}'.format(i))
        with open(args.dir+'/comp_graphs/indv_{}/code.pkl'.format(i), 'wb') as f:
            pickle.dump(repr(p), f)

    # save accuracy record
    with open(args.dir+'/comp_graphs/best/record.pkl', 'wb') as f:
        pickle.dump(log, f)


def simple_random(population, toolbox, hall_of_fame, verbose=True):
    logbook = tools.Logbook()
    logbook.header = ['indv', 'acc']
    fitnesses = toolbox.map(toolbox.evaluate, population)
    for indv, acc in zip(population, fitnesses):
        indv.fitness.values = acc
        record = {'indv':repr(indv), 'acc':acc[0]}
        logbook.record(**record)

    hall_of_fame.update(population)
    if verbose:
        print(logbook.stream)
    return population, logbook


def build_model(comp_graph):
    conf = arch_config(comp_graph=comp_graph,
                       depth_coeff=args.depth_coeff,
                       width_coeff=args.width_coeff,
                       channels=args.channels,
                       repeat_list=args.repeat_list,
                       classes=args.classes)
    return get_gepnet(conf)


def train_model(net):
    if not torch.cuda.is_available():
        log.info('no gpu device available')
        sys.exit(1)

    # set random seeds for all individuals
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    # enable torch backends
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    path = Path("/home/cliff/NWPU-RESISC45")
    tfms = get_transforms(flip_vert=True, max_lighting=0.1, max_zoom=1.05, max_warp=0.)
    data = (ImageList.from_folder(path/'train')
            .split_by_rand_pct(valid_pct=0.2, seed=args.seed)
            .label_from_folder()
            .transform(size=32)
            .databunch(bs=args.bs, num_workers=num_cpus())
            .normalize())

    learn = Learner(data, net, metrics=accuracy).to_fp16()
    learn.fit_one_cycle(args.epochs, args.max_lr, wd=args.wd)
    acc = learn.validate()
    return acc


def duration(sec):
    days, sec = sec//(24*3600), sec%(24*3600)
    hrs, sec = sec//(3600), sec%3600
    mins, sec = sec//60, sec%60
    print()
    print('Duration: %d:%02d:%02d:%02d' %(days, hrs, mins, sec))


# start search process
if __name__=='__main__':
    import time
    start = time.time()
    random_search()
    duration(time.time() - start)
