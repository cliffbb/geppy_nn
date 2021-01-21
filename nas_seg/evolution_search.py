from gepcore.utils import convolution, cell_graph
from gepcore.entity import Gene, Chromosome
from gepcore.symbol import PrimitiveSet
from gepcore.algorithm import gep_EA
from gepcore.operators import *
from nas_seg.seg_model import *
# from gepnet.utils import count_parameters
# from scipy.special import expit
from fastai.vision.all import *

import argparse

# import evolutionary tools from DEAP and GEPPY
from deap import creator, base, tools
from geppy import Toolbox

parser = argparse.ArgumentParser(description='evolutionary architectural search')
# evolutionary algorithm hyperparameter
parser.add_argument('--head_length', type=int, default=2, help='length of gene head')
parser.add_argument('--num_genes', type=int, default=3, help='num of genes per chromosome')
parser.add_argument('--num_gen', type=int, default=20, help='num of generations')
parser.add_argument('--pop_size', type=int, default=20, help='size of population')
parser.add_argument('--cx_pb', type=list, default=[0.1, 0.6], help='crossover probability')
parser.add_argument('--mu_pb', type=list, default=[0.044, 0.1], help='mutation probability')
parser.add_argument('--elites', type=int, default=1, help='num of elites selected')
parser.add_argument('--hof', type=int, default=2, help='hall of fame (record best individuals)')
parser.add_argument('--path', type=str, default='comp_graphs/experiment_1', help='path to save individuals')

# architecture config
parser.add_argument('--channels', type=int, default=16, help='initial out channels')
parser.add_argument('--input_size', type=int, default=128, help='input image size')
parser.add_argument('--classes', type=int, default=6, help='num of labels')

# training search architecture
parser.add_argument('--seed', type=int, default=200, help='training seed')
parser.add_argument('--max_lr', type=float, default=1e-2, help='max learning rate')
parser.add_argument('--wd', type=float, default=1e-4, help='weight decay')
parser.add_argument('--epochs', type=int, default=10, help='training epochs')
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
pset.add_cell_symbol(conv_symbol.sepconv3x3)
pset.add_cell_symbol(conv_symbol.sepconv5x5)
pset.add_cell_symbol(conv_symbol.dilconv3x3)
pset.add_cell_symbol(conv_symbol.dilconv5x5)
pset.add_cell_symbol(conv_symbol.maxpool3x3)
pset.add_cell_symbol(conv_symbol.avgpool3x3)


def build_model(comp_graphs):
    conf = arch_config(comp_graphs=comp_graphs,
                       channels=args.channels,
                       input_size=args.input_size,
                       classes=args.classes)
    return Network(conf)


def train_model(net):
    if not torch.cuda.is_available():
        sys.exit(1)
    # set random seeds for all individuals
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    # enable torch backends
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    def overall_acc(preds, target):
        target = target.squeeze(1)
        return (preds.argmax(dim=1) == target).float().mean()

    # dataset path
    def get_mask(x):
        dset = x.parent.name
        path = x.parent.parent.parent/'masks'/dset
        name = x.name
        return path/name

    dataset_dir = Path.home()/'rs_imagery/ISPRS_DATASETS/Vaihingen'
    data_path = dataset_dir/'vaihingen_{}'.format(args.input_size)
    img_path = data_path/'images/train'
    labels = np.array(["imp. surf.", "buildings", "low veg.", "trees", "cars", "clutter"])
    data = DataBlock(blocks=(ImageBlock, MaskBlock(codes=labels)),
                     get_items=get_image_files,
                     get_y=get_mask,
                     splitter=RandomSplitter(seed=42),
                     batch_tfms=[*aug_transforms(flip_vert=True, size=args.input_size),
                                 Normalize.from_stats([0.4769, 0.3227, 0.3191], [0.1967, 0.1358, 0.1300])])

    dls = data.dataloaders(img_path, bs=20)
    metrics = overall_acc
    learn = Learner(dls, net, wd=args.wd, metrics=metrics)
    learn.fit_one_cycle(args.epochs, args.max_lr)
    acc = learn.validate()
    return acc


def search_model():
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

    # build, train and evaluate an individual network
    def evaluate(indv):
        _, comp_graph = toolbox.comp_graph(indv)
        net = build_model(comp_graph)
        acc = train_model(net)
        fit = acc[1].item()
        return fit,

    # evaluate and select
    toolbox.register('evaluate', evaluate)
    toolbox.register('select', tools.selRoulette)

    # recombine and mutate
    toolbox.register('cx_gene', cross_gene, pb=args.cx_pb[0])
    toolbox.register('cx_2p', cross_two_point, pb=args.cx_pb[1])
    toolbox.register('cx_1p', cross_one_point, pb=args.cx_pb[1])
    toolbox.register('mut_uniform', mutate_uniform, pset=pset, pb=args.mu_pb[0])
    toolbox.register('mut_inversion', mutate_inversion, pb=args.mu_pb[1])
    toolbox.register('mut_transposition', mutate_transposition, pb=args.mu_pb[1])

    # evolution statistics
    stats = tools.Statistics(key=lambda ind: ind.fitness.values[0])
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # individual history
    hist = tools.History()
    # decorate the variation operators
    toolbox.decorate('cx_gene', hist.decorator)
    toolbox.decorate('cx_2p', hist.decorator)
    toolbox.decorate('cx_1p', hist.decorator)
    toolbox.decorate("mut_uniform", hist.decorator)
    toolbox.decorate("mut_inversion", hist.decorator)
    toolbox.decorate("mut_transposition", hist.decorator)

    # population size and best individual (hall of fame)
    pop = toolbox.population(n=args.pop_size)
    hof = tools.HallOfFame(args.hof)

    # call gep_simple evolutionary algorithm
    pop, log, his = gep_EA(pop=pop,
                      toolbox=toolbox,
                      gen_days=args.num_gen,
                      n_elites=args.elites,
                      stats=stats,
                      hof=hof,
                      history=hist,
                      verbose=True)
    print('\n', log)

    # save and draw best individuals graphs
    for i, best in enumerate(hof):
        graph, _ = cell_graph.generate_comp_graph(best)
        cell_graph.save_graph(graph, args.path+'/best/indv_{}'.format(i))
        cell_graph.draw_graph(graph, args.path+'/best/indv_{}'.format(i))

    # save population graphs
    for i, p in enumerate(pop):
        graph, _ = cell_graph.generate_comp_graph(p)
        cell_graph.save_graph(graph, args.path+'/pop/indv_{}'.format(i))

    # save stats record
    with open(args.path+'/best/stats.pkl', 'wb') as f:
        pickle.dump(log, f)

    # save history of individuals
    with open(args.path + '/best/history.pkl', 'wb') as f:
        pickle.dump(his, f)

# start evolution process
if __name__=='__main__':
    from datetime import datetime
    start = datetime.now()
    search_model()
    tm = datetime.now() - start
    print(tm)