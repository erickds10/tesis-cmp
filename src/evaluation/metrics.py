def precision_at_k(y_true, scores, k):
    import numpy as np
    idx = np.argsort(scores)[::-1][:k]
    return (y_true[idx]==1).mean()
