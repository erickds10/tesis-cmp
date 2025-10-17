def rolling_windows(arr, T):
    import numpy as np
    return np.stack([arr[i:i+T] for i in range(len(arr)-T+1)])
