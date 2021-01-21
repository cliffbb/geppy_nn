from fastai.vision.all import *
from fastai.distributed import *
from fastai.callback.tracker import SaveModelCallback
from fastprogress import fastprogress
from fastai.callback.cutmix import CutMix
from fastai.callback.mixup import MixUp

# from gepcore.utils import cell_graph
from gepnet.utils import count_parameters
# from gepnet.model_v2 import get_net, arch_config
# from pygraphviz import AGraph
# import glob

import warnings
warnings.filterwarnings('ignore')

torch.backends.cudnn.benchmark = True
fastprogress.MAX_COLS = 120

# graph = [AGraph(g) for g in glob.glob('comp_graphs/bigearthnet/*.dot')]
# _, comp_graphs = cell_graph.generate_comp_graph(graph)
#
# conf = arch_config(comp_graphs=comp_graphs,
#                    channels=40,
#                    repeat_list=[4, 4, 4, 4],
#                    classes=19)
# net = get_net(conf)


def splitter(df):
    train = df.index[~df['is_valid']].tolist()
    valid = df.index[df['is_valid']].tolist()
    return train, valid


ds_path = Path('/home/atsumilab/Clifford/rs_imagery/BigEarthNet19/')
def get_x(r): return ds_path/'train'/r['keys']
def get_y(r): return r['labels'].split('|')


def get_data(bs, ds_path):
    rgb_mean = [0.28820102, 0.29991056, 0.20993312]
    rgb_std = [0.33002318, 0.28460911, 0.27950019]
    train_df = pd.read_csv(ds_path/'train.csv', sep=';', header=0, names=['keys', 'labels', 'is_valid'])
    dblock = DataBlock(blocks=(ImageBlock, MultiCategoryBlock),
                       splitter=splitter,
                       get_x=get_x,
                       get_y=get_y,
                       batch_tfms=[*aug_transforms(), Normalize.from_stats(rgb_mean, rgb_std)])
    return dblock.dataloaders(train_df, bs=bs, num_workers=num_cpus())


@call_parse
def main(gpu:  Param("GPU to run on", int)=None,
        bs:    Param("Batch size", int)=128,
        runs:  Param("Number of times to repeat training", int)=1):

    # gpu = setup_distrib(gpu)
    if gpu is not None: torch.cuda.set_device(gpu)

    data = get_data(bs, ds_path)
    for run in range(runs):
        print(f'Run: {run}')

        mixup = MixUp()
        model_dir = '/home/atsumilab/Clifford/geppy_nn/comp_graphs/bigearthnet/322/'
        save_best = SaveModelCallback(monitor='accuracy_multi', fname='best_model')
        net = load_learner(model_dir+'model.pkl', cpu=False)
        # print(count_parameters(net.model))
        model_learner = Learner(data, net.model, metrics=partial(accuracy_multi, thresh=0.5),
                                cbs=[save_best, mixup], model_dir=model_dir).load('best_model')#.to_native_fp16()
        model_learner.export(model_dir + 'best_model1.pkl')
        # n_gpu = torch.cuda.device_count()
        #
        # # the context manager way of dp/ddp, both can handle single GPU base case.
        # ctx = model_learner.parallel_ctx if gpu is None and n_gpu else model_learner.distrib_ctx
        #
        # with partial(ctx, gpu)(): # distributed traing requires "-m fastai2.launch"
        #     print(f"Training in {ctx.__name__} context on GPU {gpu if gpu is not None else list(range(n_gpu))}")
        #     model_learner.fit_one_cycle(200, 1e-2, wd=1e-4)
        # model_learner.export(model_dir+'best_model.pkl')
