import os
import sys

import numpy as np
import pandas as pd
from htm_source.pipeline.htm_batch_runner import run_batch

_SOURCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
sys.path.append(_SOURCE_DIR)

from source.preprocess.preprocess import get_testtypes_alldata
from source.utils.utils import load_models, make_dir


def get_testtypes_anomscores(subj, htm_config, features_models, columns_model: list, filenames_data: dict,
                             testtypes_filenames: dict, dir_output: str):
    testtypes_anomscores = {ttype: [] for ttype in testtypes_filenames if ttype != 'training'}
    # split test data into testtpyes
    testtypes_alldata = get_testtypes_alldata(testtypes_anomscores=testtypes_anomscores,
                                              testtypes_filenames=testtypes_filenames,
                                              filenames_data=filenames_data,
                                              columns_model=columns_model,
                                              dir_output=dir_output)
    # Load model(s)
    dir_models = os.path.join(dir_output, 'models')
    features_models = load_models(dir_models)

    # get & save anomscores
    dir_out = os.path.join(dir_output, "anomaly")
    make_dir(dir_out)
    testtypes_features_anomscores = {}
    for ttype, data in testtypes_alldata.items():
        testtypes_features_anomscores[ttype] = {f: [] for f in features_models}
        feats_mods, features_outputs = run_batch(cfg=htm_config,
                                                 config_path=None,
                                                 learn=False,
                                                 data=data,
                                                 iter_print=1000,
                                                 features_models=features_models)
        for f, outs in features_outputs.items():
            testtypes_features_anomscores[ttype][f] = outs['anomaly_score']
            testtypes_anomscores[ttype] += outs['anomaly_score']

        path_out = os.path.join(dir_out, f"{ttype}.csv")
        ttype_anom = pd.DataFrame(testtypes_features_anomscores[ttype])
        ttype_anom.to_csv(path_out)

    return testtypes_anomscores


def get_ttypesdiffs(testtypes_anomscores):
    ttypesdiffs = {}
    ttype_combos_done = []
    for ttype, ascores in testtypes_anomscores.items():
        ttypesdiffs[ttype] = {}
        testtypes_anomscores_other = {k: v for k, v in testtypes_anomscores.items() if k != ttype}
        for ttype_other, ascores_other in testtypes_anomscores_other.items():
            combo = '--'.join(sorted([ttype, ttype_other]))
            if combo in ttype_combos_done:
                continue
            ttype_combos_done.append(combo)
            ttypesdiffs[ttype][ttype_other] = abs(np.median(ascores) - np.median(ascores_other))
    return ttypesdiffs
