def test_p_at_k():
    from src.evaluation.metrics import precision_at_k
    import numpy as np
    y = np.array([0,1,0,1]); s = np.array([0.1,0.9,0.2,0.8])
    assert 0 <= precision_at_k(y, s, 2) <= 1
