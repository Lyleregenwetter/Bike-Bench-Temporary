import warnings
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import to_rgb
import re
from typing import List, Optional
from itertools import cycle

from biked_commons.design_evaluation.design_evaluation import (
    construct_tensor_evaluator,
    get_standard_evaluations
)
from biked_commons.design_evaluation.scoring import (
    construct_scorer,
    MainScores,
    DetailedScores
)

def _ordinal(n: int) -> str:
    if 11 <= (n % 100) <= 13:
        suffix = 'th'
    else:
        suffix = {1:'st',2:'nd',3:'rd'}.get(n%10,'th')
    return f"{n}{suffix}"

def _format_num(x: float) -> str:
    """
    Format x to 3 significant figures.
    Uses scientific notation if abs(x)<1e-2 or >=1e2.
    Ensures trailing zeros to reach 3 sig‐figs.
    Collapses e+03 → e+3, e-03 → e-3.
    """
    if x == 0:
        return "0"
    # For very small or large, use 3 sig‐figs in scientific:
    if abs(x) < 1e-2 or abs(x) >= 1e3:
        s = f"{x:.2e}"  
        # this gives 2 digits after the decimal → 3 sig‐figs total
        # e.g. 1.23e+03
        return re.sub(r"e([+-])0*(\d+)", r"e\1\2", s)
    # Otherwise use general format to 3 sig‐figs
    s = f"{x:.3g}"   # e.g. "7.4" or "13"
    # If it's already scientific, leave it
    if "e" in s:
        return s
    # Count the digits (ignore the decimal point)
    digits = len(s.replace(".", ""))
    # If fewer than 3, pad with trailing zeros
    if "." in s:
        zeros_needed = 3 - digits
        if zeros_needed > 0:
            s = s + "0" * zeros_needed
    else:
        zeros_needed = 3 - digits
        s = s + "." + "0" * zeros_needed
    return s

class ScoreReportDashboard:
    def __init__(
        self,
        design_batches: List[torch.Tensor],
        model_names: Optional[List[str]],
        condition: dict,
        column_names: List[str],
        model_colors: Optional[List[str]] = None,
        device: str = "cpu"
    ):
        # default model names if needed
        n_models = len(design_batches)
        if model_names is None:
            model_names = [f"Model_{i+1}" for i in range(n_models)]
            warnings.warn(f"No model_names provided; defaulting to {model_names!r}")
        if len(model_names) != n_models:
            raise ValueError("design_batches and model_names must match in length")
        self.model_names = model_names

        # required checks
        if condition is None:
            raise ValueError("condition is required")
        if column_names is None:
            raise ValueError("column_names is required")
        D = len(column_names)
        for b in design_batches:
            if b.shape[1] != D:
                raise ValueError("All design_batches must match column_names length")
        self.design_batches = {n:b.to(device) for n,b in zip(model_names, design_batches)}
        self.condition      = condition
        self.column_names   = column_names
        self.device         = device

        # colors default
        if model_colors is None:
            base = plt.rcParams['axes.prop_cycle'].by_key()['color']
            model_colors = [c for _,c in zip(model_names, cycle(base))]
            warnings.warn("No model_colors provided; using Matplotlib cycle.")
        if len(model_colors) != n_models:
            raise ValueError("model_colors must match model_names length")
        self.model_colors = dict(zip(model_names, model_colors))

        # build eval + scorers
        eval_funcs = get_standard_evaluations(device)
        (self._tensor_evaluator,
         self.eval_names,
         self.eval_types) = construct_tensor_evaluator(
                              eval_funcs, column_names, device=device
                            )
        self._main_scorer     = construct_scorer(
                                  MainScores, eval_funcs, column_names, device=device
                               )
        self._detailed_scorer = construct_scorer(
                                  DetailedScores, eval_funcs, column_names, device=device
                               )

        # objective / constraint keys
        self.objective_names      = [n for n,t in zip(self.eval_names,self.eval_types) if t==1]
        self.constraint_names     = [n for n,t in zip(self.eval_names,self.eval_types) if t==0]
        self.mean_obj_keys        = [f"Mean Objective Score: {n}" for n in self.objective_names]
        self.const_violation_keys = [f"Constraint Violation Rate: {n}" for n in self.constraint_names]

        # precompute
        self._compute_aggregate_metrics()
        self.main_df   = self._compute_main_scores_df()
        self.detail_df = self._compute_detailed_scores_df()

    def _compute_aggregate_metrics(self):
        self.model_mean_objs   = {}
        self.model_const_rates = {}
        for name,b in self.design_batches.items():
            arr = self._detailed_scorer(b, self.condition)
            self.model_mean_objs[name]   = arr[self.mean_obj_keys].to_numpy(float)
            self.model_const_rates[name] = arr[self.const_violation_keys].to_numpy(float)
        all_means = np.stack(list(self.model_mean_objs.values()), axis=0)
        self.obj_min = all_means.min(axis=0)
        self.obj_max = all_means.max(axis=0)

    def _compute_main_scores_df(self) -> pd.DataFrame:
        rows = {n: self._main_scorer(b, self.condition)
                for n,b in self.design_batches.items()}
        return pd.DataFrame(rows).T

    def _compute_detailed_scores_df(self) -> pd.DataFrame:
        rows = {n: self._detailed_scorer(b, self.condition)
                for n,b in self.design_batches.items()}
        return pd.DataFrame(rows).T

    def _model_index(self, model_name: str) -> int:
        return self.model_names.index(model_name)

    def show_model(
        self,
        model_name: Optional[str]       = None,
        objectives_per_row: int         = 5,
        constraints_per_row: int        = 20,
        total_width: float              = 12.0,
        summary_cell_height: float      = 0.4,
        objective_cell_height: float    = 1.0
    ):
        """Render one model’s scorecard with clipped KDEs and baseline ticks."""
        # default model
        if model_name is None:
            model_name = self.model_names[0]
            warnings.warn(f"No model_name given; defaulting to {model_name!r}")
        if model_name not in self.design_batches:
            raise ValueError(f"Unknown model: {model_name!r}")
        color = self.model_colors[model_name]

        # 1) Summary stats
        md = self.main_df
        summary_defs = [
            ("Maximum Mean Discrepancy",     True),
            ("Hypervolume",                  False),
            ("Constraint Satisfaction Rate", False),
        ]
        s_vals, s_ranks, s_mins, s_maxs = [],[],[],[]
        s_all = {}
        for col, low_best in summary_defs:
            sr = md[col]
            s_all[col] = sr.values
            s_vals.append(sr.loc[model_name])
            mn, mx = sr.min(), sr.max()
            s_mins.append(mn); s_maxs.append(mx)
            s_ranks.append(int(sr.rank(method='min', ascending=low_best).loc[model_name]))

        # 2) Layout
        obj_count = len(self.objective_names)
        con_count = len(self.constraint_names)
        obj_rows  = int(np.ceil(obj_count/objectives_per_row))
        con_rows  = int(np.ceil(con_count/constraints_per_row))
        cons_cell = total_width/constraints_per_row
        fig_h     = summary_cell_height \
                  + obj_rows*objective_cell_height \
                  + con_rows*cons_cell

        fig = plt.figure(figsize=(total_width, fig_h))
        fig.subplots_adjust(
            left   = 0.02,
            right  = 0.98,
            top    = 0.95,
            bottom = 0.05,
            hspace = 0.6
        )
        outer = fig.add_gridspec(
            3,1,
            height_ratios=[
                summary_cell_height,
                obj_rows*objective_cell_height,
                con_rows*cons_cell
            ]
        )

        # 3) Summary row
        nsum   = len(summary_defs) + 1   # = 4
        gs_sum = outer[0].subgridspec(
            1,
            nsum,
            width_ratios=[0.16, 0.28, 0.28, 0.28],
            wspace=0.1
        )
        ax0    = fig.add_subplot(gs_sum[0,0])
        ax0.text(0, 0.5, f"{model_name}\nScorecard",
                 ha='left', va='center', fontsize=12, fontweight='bold')
        ax0.axis('off')

        for i,(col,low_best) in enumerate(summary_defs):
            ax  = fig.add_subplot(gs_sum[0, i+1])
            lo,hi = s_mins[i], s_maxs[i]
            pad   = 0.05*(hi-lo)
            x0,x1 = lo-pad, hi+pad
            val   = s_vals[i]
            rk    = s_ranks[i]

            # baseline + end‐ticks
            ax.hlines(0, x0, x1, color='black', linewidth=1)
            ax.plot([x0,x1],[0,0],'|k', markersize=4)

            # other models as gray ticks
            for v in s_all[col]:
                ax.plot(v,0,'|', color='gray', markersize=6)

            # focal model tick (larger, colored)
            ax.plot(val,0,'|', color=color, markersize=10)

            # extreme‐value labels down at y=0.02
            ax.text(x0, 0.02, _format_num(x0),
                    ha='center', va='bottom', fontsize=7)
            ax.text(x1, 0.02, _format_num(x1),
                    ha='center', va='bottom', fontsize=7)

            # … **new**: rank under focal tick at y = –0.2
            ax.text(val, -0.022, f"({_ordinal(rk)})",
                    ha='center', va='top', fontsize=8, color=color)

            ax.text(
                0.5, 0.45, col,
                ha='center', va='bottom',
                transform=ax.transAxes,
                fontsize=9,
                fontweight='bold'
            )
            ax.set_ylim(-0.01, 0.05)
            ax.axis('off')

        # 4) Objective KDEs
        # gather raw arrays
        all_raw = {}
        is_obj  = np.array(self.eval_types)==1
        for name,b in self.design_batches.items():
            with torch.no_grad():
                out = self._tensor_evaluator(b, self.condition)
            arr = out.detach().cpu().numpy()
            all_raw[name] = arr[:, is_obj]

        gs_obj = outer[1].subgridspec(
            obj_rows,
            objectives_per_row,
            wspace=0.05,
            hspace=0.8    # <-- try values like 0.2, 0.3, 0.4 until it feels right
        )
        for idx in range(obj_count):
            r,c = divmod(idx, objectives_per_row)
            ax  = fig.add_subplot(gs_obj[r,c])

            # compute per‐model 1%/99% then global bounds
            p1s  = [np.percentile(all_raw[m][:,idx],  1) for m in self.model_names]
            p99s = [np.percentile(all_raw[m][:,idx], 99) for m in self.model_names]
            low, high = min(p1s), max(p99s)
            mean_val  = all_raw[model_name][:,idx].mean()
            # gray KDEs clipped
            for other in self.model_names:
                sns.kdeplot(
                    data      = all_raw[other][:,idx],
                    ax        = ax,
                    # cut       = 0,
                    clip      = (low, high),
                    bw_adjust = 1.0,
                    color     = 'gray',
                    alpha     = 0.2,
                    linewidth = 1,
                    fill      = False
                )

            # colored KDE clipped
            sns.kdeplot(
                data      = all_raw[model_name][:,idx],
                ax        = ax,
                # cut       = 0,
                clip      = (low, high),
                bw_adjust = 1.0,
                color     = color,
                alpha     = 0.6,
                linewidth = 1,
                fill      = True
            )

            ax.set_xlim(low, high)

            # baseline ticks at every model's mean
            means = {m: all_raw[m][:,idx].mean() for m in self.model_names}
            for m,mv in means.items():
                ax.plot(mv, 0, '|',
                        color=('gray' if m!=model_name else color),
                        markersize=(6 if m!=model_name else 10))

            # extremes labels
            ax.set_xticks([low, high])
            ax.set_xticklabels([_format_num(low), _format_num(high)], fontsize=7)
            ax.set_yticks([]); ax.set_ylabel('')

            ax.tick_params(axis='x', pad=2)

            labels = ax.get_xticklabels()
            if len(labels) >= 2:
                labels[0].set_ha('left')   # left‐align the “low” label
                labels[1].set_ha('right')

            # **new**: rank under focal tick
            sorted_models = sorted(self.model_names, key=lambda m: means[m])
            rk = _ordinal(sorted_models.index(model_name)+1)
            y0 = 0.16 * (ax.get_ylim()[1] - ax.get_ylim()[0])
            ax.text(mean_val, y0, f"({rk})",
                    ha='center', va='bottom', fontsize=7)

            ax.set_title(self.objective_names[idx], fontsize=9, pad=2)
            for loc in ['top','right','left']:
                ax.spines[loc].set_visible(False)
            ax.spines['bottom'].set_visible(True)

        # blank extras
        for j in range(obj_count, obj_rows*objectives_per_row):
            r,c = divmod(j, objectives_per_row)
            fig.add_subplot(gs_obj[r,c]).axis('off')

        # 5) Constraints (unchanged)
        gs_con = outer[2].subgridspec(con_rows, constraints_per_row, wspace=0.02)
        white = np.array([1.0,1.0,1.0])
        for idx in range(con_count):
            r,c = divmod(idx, constraints_per_row)
            ax  = fig.add_subplot(gs_con[r,c])

            rate = self.model_const_rates[model_name][idx]
            adj  = np.sqrt(rate)
            face = white*(1-adj) + np.array(to_rgb(color))*adj
            ax.patch.set_facecolor(tuple(face))

            arr  = np.stack(list(self.model_const_rates.values()), axis=0)
            rank = int(pd.Series(arr[:,idx])
                       .rank(method='min',ascending=True)
                       .iloc[self._model_index(model_name)])
            ax.text(0.5,0.6,f"{rate:.2f}",ha='center',va='center',fontsize=7)
            ax.text(0.5,0.2,f"({_ordinal(rank)})",ha='center',va='center',fontsize=6)
            ax.set_title(f"C{idx+1}",fontsize=9,pad=2)
            ax.set_xticks([]); ax.set_yticks([])
            for loc in ax.spines:
                ax.spines[loc].set_visible(False)

        for j in range(con_count, con_rows*constraints_per_row):
            r,c = divmod(j, constraints_per_row)
            fig.add_subplot(gs_con[r,c]).axis('off')

        # for ax in fig.get_axes():
        #     # turn the frame on
        #     ax.set_frame_on(True)
        #     # ensure every spine is visible, black, and 1pt thick
        #     for spine in ax.spines.values():
        #         spine.set_visible(True)
        #         spine.set_linewidth(1)
        #         spine.set_color("black")

        plt.show()
