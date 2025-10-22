import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import pi
from pathlib import Path

def f_exp(x): return np.exp(x)
def df_exp(x): return np.exp(x)
def f_sin(x): return np.sin(x)
def df_sin(x): return np.cos(x)

FUNC_MAP = {"exp": (f_exp, df_exp, 1.0),
            "sin": (f_sin, df_sin, pi/4)}

def run(func_names=("exp","sin"), hs=None, save_dir="outputs"):
    if hs is None:
        hs = np.array([10.0**(-k) for k in range(1, 11)], dtype=float)
    out = Path(save_dir); out.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for name in func_names:
        f, df, a = FUNC_MAP[name]
        exact = df(a)
        for h in hs:
            fwd = (f(a+h)-f(a))/h
            bwd = (f(a)-f(a-h))/h
            cen = (f(a+h)-f(a-h))/(2*h)
            all_rows.append({
                "func": name, "a": a, "h": h,
                "forward": fwd, "backward": bwd, "central": cen,
                "forward_err": abs(fwd-exact),
                "backward_err": abs(bwd-exact),
                "central_err": abs(cen-exact),
                "exact_df": exact,
            })

    df_all = pd.DataFrame(all_rows)
    df_out = df_all.copy()
    for col in ["h","forward_err","backward_err","central_err"]:
        df_out[col] = df_out[col].map(lambda x: float(f"{x:.16e}"))
    (out/"finite_diff_errors.csv").write_text(df_out.to_csv(index=False))

    # 요약
    summary_rows = []
    for name in df_all["func"].unique():
        sub = df_all[df_all["func"]==name]
        for method in ["forward_err","backward_err","central_err"]:
            idx = sub[method].idxmin()
            summary_rows.append({
                "func": name,
                "method": method.replace("_err",""),
                "best_h": sub.loc[idx,"h"],
                "min_error": sub.loc[idx,method],
            })
    pd.DataFrame(summary_rows).to_csv(out/"finite_diff_summary.csv", index=False)

    # 그래프
    for name in df_all["func"].unique():
        sub = df_all[df_all["func"]==name]
        plt.figure()
        plt.loglog(sub["h"], sub["forward_err"], marker="o", label="forward")
        plt.loglog(sub["h"], sub["backward_err"], marker="s", label="backward")
        plt.loglog(sub["h"], sub["central_err"], marker="^", label="central")
        plt.gca().invert_xaxis()
        plt.xlabel("h"); plt.ylabel("absolute error")
        plt.title(f"Error vs h (log-log) for {name}")
        plt.legend()
        plt.savefig(out/f"error_vs_h_{name}.png", dpi=200, bbox_inches="tight")
        plt.close()

if __name__ == "__main__":
    run()
