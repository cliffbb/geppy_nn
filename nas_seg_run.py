from fastai.vision.all import *
from fastai.distributed import *
from fastai.callback.all import *
from fastai.callback.tracker import SaveModelCallback
from fastprogress import fastprogress

from skimage import io
from sklearn.metrics import confusion_matrix

from gepcore.utils import cell_graph, convolution
from gepcore.entity import Gene, Chromosome
from gepcore.symbol import PrimitiveSet
from nas_seg.seg_model import get_net, arch_config, Network
from gepnet.utils import count_parameters
from pygraphviz import AGraph
import glob

import warnings
warnings.filterwarnings('ignore')

torch.backends.cudnn.benchmark = True
fastprogress.MAX_COLS = 120

# comp_graphs = []
# for i in range(3):
#     graph = [AGraph(g) for g in glob.glob('comp_graphs/segment/{}/*.dot'.format(i))]
#     _, comp_graph = cell_graph.generate_comp_graph(graph)#
#     comp_graphs.append(comp_graph)

graph = [AGraph(g) for g in glob.glob('comp_graphs/segment/*.dot')]
_, comp_graphs = cell_graph.generate_comp_graph(graph)

msk_labels = np.array(["roads", "buildings", "low veg.", "trees", "cars", "clutter"])
num_classes = len(msk_labels)

conf = arch_config(comp_graphs=comp_graphs, channels=64, classes=num_classes)
net = Network(conf)

def get_data(bs, codes=msk_labels):
    window_size = 256
    dataset = 'Vaihingen' #'Potsdam'
    dataset_dir = Path.home()/'Clifford/rs_imagery/ISPRS-DATASETS/{}'.format(dataset)

    if dataset == 'Potsdam':
        tiles = dataset_dir/'Ortho_IRRG/top_potsdam_{}_{}_IRRG.tif'
        masks = dataset_dir/'Labels_for_participants/top_potsdam_{}_{}_label.tif'
        e_masks = dataset_dir/'Labels_for_participants_no_Boundary/top_potsdam_{}_label_noBoundary.tif'
        trainset_dir = dataset.lower() + '_{}'.format(window_size) 
        # testset_ids = ['2_11', '2_12', '4_10', '5_11', '6_7', '7_8', '7_10']
    elif dataset == 'Vaihingen':
        tiles = dataset_dir/'top/top_mosaic_09cm_area{}.tif'
        masks = dataset_dir/'gts_for_participants/top_mosaic_09cm_area{}.tif'
        e_masks = dataset_dir/'gts_eroded_for_participants/top_mosaic_09cm_area{}_noBoundary.tif'
        trainset_dir = dataset.lower() + '_{}'.format(window_size) 
        # testset_ids = ['5', '7', '23', '30']

    data_path = dataset_dir/'{}'.format(trainset_dir)
    img_path = data_path/'images/train'
    msk_path = data_path/'masks/train'
    get_mask = lambda x: msk_path/f'{x.stem}{x.suffix}'

    dblock = DataBlock(blocks=(ImageBlock, MaskBlock(codes=codes)),
                    get_items=get_image_files,
                    get_y=get_mask,
                    splitter=RandomSplitter(seed=42),
                    batch_tfms=[*aug_transforms(flip_vert=True, size=window_size), 
                    Normalize.from_stats([0.4769, 0.3227, 0.3191], [0.1967, 0.1358, 0.1300])])

    return dblock.dataloaders(img_path, bs=bs, num_workers=num_cpus())

# def cm(preds, target, labels):
#     preds = preds.argmax(dim=1).cpu().numpy().ravel()
#     target = target.cpu().numpy().ravel()
#     return confusion_matrix(y_true=target, y_pred=preds, labels=labels) 

def overall_acc(preds, target, labels=range(num_classes)):
    """Calculate over accuracy"""
    preds = preds.argmax(dim=1).cpu().numpy().ravel()
    target = target.cpu().numpy().ravel()
    cm = confusion_matrix(y_true=target, y_pred=preds, labels=labels)
    # cm_ = cm(preds=preds, target=target, labels=labels) 
    acc = np.trace(cm) / np.sum(cm)
    return torch.tensor(acc, device='cuda')


@call_parse
def main(gpu:  Param("GPU to run on", int)=None,
        bs:    Param("Batch size", int)=8,
        arch:  Param("Architecture", str)=net,
        runs:  Param("Number of times to repeat training", int)=1):

    # gpu = setup_distrib(gpu)
    if gpu is not None: torch.cuda.set_device(gpu)
        
    data = get_data(bs)
    metrics=overall_acc

    print(f'Model size: {count_parameters(net)}M')
    weights = torch.tensor([[0.9]*5 + [1.1]]).cuda()
    loss_func = CrossEntropyLossFlat(weight=weights, axis=1) 
    model_dir = '/home/atsumilab/Clifford/geppy_nn/comp_graphs/segment'
    save = SaveModelCallback(monitor='overall_acc')

    for run in range(runs):
        print(f'Run: {run}')
        learn = Learner(data, net, wd=1e-4, loss_func=loss_func, metrics=metrics, model_dir=model_dir, cbs=[save]) #.to_fp16()

        n_gpu = torch.cuda.device_count()
        
        # The old way to use DataParallel, or DistributedDataParallel training:
        # if gpu is None and n_gpu: learn.to_parallel()
        # if num_distrib()>1: learn.to_distributed(gpu) # Requires `-m fastai2.launch`

        # the context manager way of dp/ddp, both can handle single GPU base case.
        ctx = learn.parallel_ctx if gpu is None and n_gpu else learn.distrib_ctx

        with partial(ctx, gpu)(): # distributed traing requires "-m fastai2.launch"
            print(f"Training in {ctx.__name__} context on GPU {gpu if gpu is not None else list(range(n_gpu))}")
            learn.fit_one_cycle(80, 1e-3, wd=4e-4) 
