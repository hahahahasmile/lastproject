# tuner.py
import numpy as np
import pandas as pd
import json
import time
import random
from collections import deque

# lightgbm 사용 (없으면 sklearn 기본 회귀로 대체)
try:
    import lightgbm as lgb
    _HAS_LGB = True
except Exception:
    _HAS_LGB = False
    from sklearn.ensemble import RandomForestRegressor

def sample_params(rng):
    # 기본 sample 공간 (BT-상승 전용 권장 범위)
    return {
        "k_sl_A": float(rng.uniform(0.8, 2.0)),
        "k_tp_A": float(rng.uniform(1.0, 4.0)),
        "k_sl_C": float(rng.uniform(0.8, 2.0)),
        "k_tp_C": float(rng.uniform(1.0, 4.0)),
        "sim_gate": float(rng.uniform(0.6, 0.9)),
        "delay_h": int(rng.integers(12, 36)),
    }

def params_to_vector(p):
    return [p["k_sl_A"], p["k_tp_A"], p["k_sl_C"], p["k_tp_C"], p["sim_gate"], p["delay_h"]]

def run_bayes_opt(
    evaluate_fn,
    n_trials=40,
    n_init=8,
    N_pool=3000,
    topk=2,
    random_seed=42,
    verbose=True,
    log_path=None
):
    """
    evaluate_fn: callable(params: dict) -> float (score). Must return float.
                 If no trades / invalid, should return 0.0 (neutral baseline).
    n_trials: total number of evaluated trials (including n_init)
    n_init: initial random evaluations
    N_pool: pool size for surrogate ranking
    topk: how many surrogate-top candidates to evaluate per iteration
    """
    rng = np.random.default_rng(random_seed)
    random.seed(random_seed)
    logs = []
    X = []
    y = []

    # 1) initial random
    for i in range(n_init):
        p = sample_params(rng)
        score = float(evaluate_fn(p))
        logs.append({"trial": len(logs)+1, "params": p, "score": score})
        X.append(params_to_vector(p)); y.append(score)
        if verbose:
            print(f"[init {i+1}/{n_init}] score={score:.4f} params={p}")

    # 2) iterative surrogate loop
    it = 0
    while len(logs) < n_trials:
        it += 1
        # train surrogate regressor
        X_arr = np.array(X)
        y_arr = np.array(y)
        if _HAS_LGB:
            dtrain = lgb.Dataset(X_arr, y_arr, free_raw_data=False)
            lgb_params = {"objective": "regression", "verbosity": -1, "seed": random_seed}
            booster = lgb.train(
                lgb_params,
                dtrain,
                num_boost_round=100,
                valid_sets=[dtrain],                       # 콜백이 동작하려면 valid_sets 필요
                callbacks=[lgb.log_evaluation(period=0)]   # 로그 완전 비활성 (verbose_eval 대체)
            )
            def surrogate_predict(x):
                return booster.predict(np.array(x))
        else:
            rf = RandomForestRegressor(n_estimators=100, random_state=random_seed)
            rf.fit(X_arr, y_arr)
            def surrogate_predict(x):
                return rf.predict(np.array(x))

        # pool and predict
        pool = [sample_params(rng) for _ in range(N_pool)]
        pool_X = [params_to_vector(p) for p in pool]
        preds = surrogate_predict(pool_X)
        idx_sorted = np.argsort(preds)[::-1]
        chosen = []
        # exploitation: topk by surrogate
        for iidx in range(min(topk, len(idx_sorted))):
            chosen.append(pool[int(idx_sorted[iidx])])
        # exploration: add 1 random sample
        chosen.append(sample_params(rng))

        # evaluate chosen with real evaluate_fn
        for p in chosen:
            score = float(evaluate_fn(p))
            logs.append({"trial": len(logs)+1, "params": p, "score": score})
            X.append(params_to_vector(p)); y.append(score)
            if verbose:
                print(f"[iter {it}] eval #{len(logs)} score={score:.4f} params={p}")
            if len(logs) >= n_trials:
                break

    # build logs df
    df_logs = pd.DataFrame([
        {
            "trial": l["trial"],
            "score": l["score"],
            "k_sl_A": l["params"]["k_sl_A"],
            "k_tp_A": l["params"]["k_tp_A"],
            "k_sl_C": l["params"]["k_sl_C"],
            "k_tp_C": l["params"]["k_tp_C"],
            "sim_gate": l["params"]["sim_gate"],
            "delay_h": l["params"]["delay_h"],
        } for l in logs
    ])

    best_idx = int(np.argmax([l["score"] for l in logs]))
    best = {"score": logs[best_idx]["score"], "params": logs[best_idx]["params"]}

    if log_path:
        with open(log_path, "w") as f:
            json.dump({"best": best, "logs": logs}, f, default=str)

    return best, df_logs
