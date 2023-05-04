import numpy as np


def get_ascores_entropy(data):
    preds = []
    for _ in range(len(data)):
        if _ < 3:
            continue
        lag1, lag2, lag3 = data[_-3], data[_-2], data[_-1]
        pred = lag1 + (lag1-lag2) + 0.5*((lag1-lag2) - (lag2-lag3))
        preds.append(pred)
    ascores = []
    for _, p in enumerate(preds):
        ascore = abs(p - data[_+3])
        ascores.append(ascore)
    return ascores


def get_ascores_naive(data):
    preds = []
    for _ in range(len(data)):
        if _ == 0:
            continue
        pred = data[_-1]
        preds.append(pred)
    ascores = []
    for _, p in enumerate(preds):
        ascore = abs(p - data[_+1])
        ascores.append(ascore)
    return ascores


def get_ascore_pyod(_, data, model):
    aScore = [0]
    if _ > 0:
        aScore = model.decision_function(data[(_-1):_])  # outlier scores
    return abs(aScore[0])


def get_ascores_pyod(data, model):
    aScores = list()
    for _ in range(data.shape[0]):
        aScores.append(get_ascore_pyod(_, data, model))
    return aScores


def get_wllevels_diffs(wllevels_anomscores):
    # Get difference in median anomaly score between wllevels
    wllevels_diffs = {}
    wllevel_combos_done = []
    for wllevel, ascores in wllevels_anomscores.items():
        wllevels_diffs[wllevel] = {}
        wllevels_anomscores_other = {k: v for k, v in wllevels_anomscores.items() if k != wllevel}
        for wllevel_other, ascores_other in wllevels_anomscores_other.items():
            combo = '--'.join(sorted([wllevel, wllevel_other]))
            if combo in wllevel_combos_done:
                continue
            wllevel_combos_done.append(combo)
            ascores_total = max(np.sum(ascores), 0.025)
            ascores_total_other = max(np.sum(ascores_other), 0.025)
            pct_change = (ascores_total_other - ascores_total) / ascores_total
            wllevels_diffs[wllevel][wllevel_other] = round(pct_change*100, 2)
    return wllevels_diffs
