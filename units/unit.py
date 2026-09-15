import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from scipy.optimize import linear_sum_assignment
 
def normalize(x, flag='default', epsilon=1e-8):
    if flag == 'default':  
        x = (x - np.min(x)) / (np.max(x) - np.min(x) + epsilon)
    elif flag == 'col_vector':   
        x = (x - np.min(x, axis=0, keepdims=True)) / \
            (np.max(x, axis=0, keepdims=True) - np.min(x, axis=0, keepdims=True) + epsilon)
    elif flag == 'row_vector':   
        x = (x - np.min(x, axis=1, keepdims=True)) / \
            (np.max(x, axis=1, keepdims=True) - np.min(x, axis=1, keepdims=True) + epsilon)
    elif flag == 'sigma':
        x = (x - np.mean(x, axis=0, keepdims=True)) / (np.std(x, axis=0, keepdims=True) + epsilon)
    return x


def clustering_metrics(y_pred, labels):
    """Return clustering ACC, ARI, and NMI."""
    labels = np.asarray(labels, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    size = int(max(y_pred.max(), labels.max()) + 1)
    contingency = np.zeros((size, size), dtype=np.int64)
    for pred, target in zip(y_pred, labels):
        contingency[pred, target] += 1

    rows, cols = linear_sum_assignment(contingency.max() - contingency)
    acc = contingency[rows, cols].sum() / labels.size
    return {
        "acc": round(float(acc), 4),
        "ari": round(float(adjusted_rand_score(labels, y_pred)), 4),
        "nmi": round(float(normalized_mutual_info_score(labels, y_pred)), 4),
    }




def set_requires_grad(module, flag: bool):
    for p in module.parameters():
        p.requires_grad = flag

def GRL_coeff(epoch, beta, n):
    coeff = 2 / (1 + np.exp(-beta * epoch / n)) - 1
    return coeff




