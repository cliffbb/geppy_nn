# Utils
import itertools
import numpy as np
from sklearn.metrics import confusion_matrix


def sliding_window(image, step, window_size):
    """ Slide a window shape (window_size, window_size) across the image with a stride of step"""
    for x in range(0, image.shape[0], step):
        if x + window_size > image.shape[0]:
            x = image.shape[0] - window_size
        for y in range(0, image.shape[1], step):
            if y + window_size > image.shape[1]:
                y = image.shape[1] - window_size
            yield x, y, window_size, window_size


def count_sliding_window(image, step, window_size):
    """ Count the number of windows in an image """
    c = 0
    for x in range(0, image.shape[0], step):
        if x + window_size > image.shape[0]:
            x = image.shape[0] - window_size
        for y in range(0, image.shape[1], step):
            if y + window_size > image.shape[1]:
                y = image.shape[1] - window_size
            c += 1
    return c


def grouper(n, iterable):
    """ Iterate by chunk of n elements """
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk


def get_mask(x):
    dset = x.parent.name
    path = x.parent.parent.parent/'masks'/dset
    name = x.name
    return path/name


def overall_acc(preds, target):
    target = target.squeeze(1)
    return (preds.argmax(dim=1)==target).float().mean()


def metrics(predictions, gts, labels):
    # Compute overall accuracy
    def accuracy(cm):
        tp = 0.0
        cm = cm.astype(np.float64)
        for l in range(cm.shape[0]):
            tp += cm[l, l]
        return tp / cm.sum()

    # Compute F1 score (Dice coefficient)
    def f1score(cm, labels):
        score = []
        lb_score = {}
        for i, l in enumerate(labels):
            score.append(2. * cm[i, i] / (np.sum(cm[i, :]) + np.sum(cm[:, i])))
            lb_score[l] = score[i]
        return np.mean(score[:-1]), lb_score

    # Compute IoU score (Jaccard coefficient)
    def jaccard(cm, labels):
        score = []
        lb_score = {}
        for i, l in enumerate(labels):
            tp = cm[i, i]
            fn = np.sum(cm[i, :]) - tp
            fp = np.sum(cm[:, i]) - tp
            score.append(float(tp) / (tp + fp + fn))
            lb_score[l] = score[i]
        return np.mean(score[:-1]), lb_score

    # Compute Kappa coefficient
    def kappa(cm):
        total = np.sum(cm)
        pa = np.trace(cm) / float(total)
        pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / float(total * total)
        return (pa - pe) / (1 - pe)

    # confusion matrix
    cm = confusion_matrix(gts, predictions, range(len(labels)))

    print("Confusion matrix :")
    print(cm)
    print()

    # accuracy
    total = np.sum(cm)
    acc = accuracy(cm)
    print("{} pixels processed".format(total))
    print("Total accuracy : {}".format(acc))
    print()

    # F1 score
    mean_f1, f1 = f1score(cm, labels)
    print("F1Score :")
    for i in f1:
        print("{} -- {}".format(i, f1[i]))
    print('mean F1Score :', mean_f1)
    print()

    # IoU score
    mean_iou, iou = jaccard(cm, labels)
    print("IoU :")
    for i in iou:
        print("{} -- {}".format(i, iou[i]))
    print('mean IoU :', mean_iou)
    print()

    # Kappa coefficient
    k = kappa(cm)
    print("Kappa: ", k)
    print("-" * 40)
    print()
    return accuracy


# def cm(preds, target, labels):
#     preds = preds.argmax(dim=1).cpu().numpy().ravel()
#     target = target.cpu().numpy().ravel()
#     return confusion_matrix(y_true=target, y_pred=preds, labels=labels)

# def overall_acc(preds, target, labels=range(num_classes)):
#     """Calculate over accuracy"""
#     cm_ = cm(preds=preds, target=target, labels=labels)
#     acc = np.trace(cm_) / np.sum(cm_)
#     return torch.tensor(acc, device='cuda')


__all__ = ['count_sliding_window', 'sliding_window', 'grouper', 'get_mask', 'overall_acc', 'metrics']