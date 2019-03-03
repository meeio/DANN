def make_weighted_sum(vw):
    assert len(vw) == 2
    v,w = vw
    assert len(v) == len(w)
    r = sum(v[i] * w[i] for i in range(len(w))) / sum(w)
    return r