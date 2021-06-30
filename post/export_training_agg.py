#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 09-04-2021
           """

import os
from pathlib import Path

import numpy
import pandas
import seaborn
from apppath import ensure_existence, system_open_path
from draugr.pandas_utilities import ChainedAssignmentOptionEnum
from draugr.tqdm_utilities import progress_bar
from draugr.visualisation import (
    FigureSession,
    MonoChromeStyleSession,
    despine_all,
    exponential_moving_average,
    fix_edge_gridlines,
    latex_clean_label,
    monochrome_line_no_marker_cycler,
    save_embed_fig,
    set_y_log_scale,
)
from draugr.writers import (
    TrainingScalars,
    should_plot_y_log_scale,
    should_smooth_series,
)
from matplotlib import pyplot
from warg import ContextWrapper, GDKC

from configs import EXPORT_RESULTS_PATH


def training_agg_plot(
    agg_path=ensure_existence(EXPORT_RESULTS_PATH / "agg"),
    results_path=EXPORT_RESULTS_PATH / "csv" / "training",
    only_latest_load_time: bool = False,
    include_timestamp: bool = False,
    color_plot: bool = False,
    include_titles: bool = False,
    compute_scalar_agg: bool = True,
    compute_tensor_agg: bool = True,
) -> None:
    # use_monochrome_style()
    seaborn.set()
    seaborn.set_style("ticks", rc={"axes.grid": True, "text.usetex": True})

    if only_latest_load_time:
        max_load_time = max(
            list(results_path.iterdir()),
            key=os.path.getctime,
        )
        results_paths = [max_load_time]
    else:
        results_paths = list(results_path.iterdir())

    if compute_scalar_agg:
        for timestamp in progress_bar(results_paths, description="timestamp #"):
            if timestamp.is_dir():
                for mapping in progress_bar(
                    list(timestamp.iterdir()), description="mapping #"
                ):
                    if mapping.is_dir():
                        df_m = {}
                        for transformation in progress_bar(
                            mapping.iterdir(), description="transformation #"
                        ):

                            if transformation.is_dir():
                                dfs = {}
                                a = list(transformation.rglob("*scalars_*.csv"))
                                for i, seed_ith in enumerate(progress_bar(a)):
                                    dfs[i] = pandas.read_csv(seed_ith)
                                # df_a = pandas.concat(dfs, axis=0)

                                df_m[transformation.name] = pandas.concat(
                                    dfs, axis=0, names=["seed", "epoch"]
                                )
                        result = pandas.concat(
                            df_m, axis=0, names=["transformation", "seed", "epoch"]
                        )
                        if numpy.any(result.isna().any()) and False:
                            print(
                                result.columns[mapping.isna().any()].tolist()
                            )  # WHAT mapping.isna().any()
                            print(result.isna().any())
                            print(result.loc[:, result.isna().any()])
                            print(f"{mapping} result has nans")
                            raise Exception

                        for tag in TrainingScalars:
                            # if True or tag is not TrainingScalars.new_best_model:
                            hue_a = None
                            if color_plot:
                                hue_a = result.index
                            with ContextWrapper(
                                GDKC(
                                    MonoChromeStyleSession,
                                    prop_cycler=monochrome_line_no_marker_cycler,
                                ),
                                not color_plot,
                            ):

                                post_str = ""
                                ykey = tag.value
                                if should_smooth_series(tag):
                                    # with suppress(SettingWithCopyWarning):
                                    with pandas.option_context(
                                        "mode.chained_assignment",
                                        ChainedAssignmentOptionEnum.raises.value,
                                    ):

                                        ykey = f"smoothed_{tag.value}"
                                        ema_alpha = 0.4
                                        result[ykey] = result[tag.value]
                                        post_str = f"EMA \alpha={ema_alpha}"
                                        for i in progress_bar(
                                            result.index.unique(level="transformation"),
                                            disable=True,
                                        ):
                                            for j in progress_bar(
                                                result.index.unique(level="seed"),
                                                disable=True,
                                            ):
                                                result.loc[
                                                    (i, j), ykey
                                                ] = exponential_moving_average(
                                                    result.loc[(i, j), ykey],
                                                    decay=1.0 - ema_alpha,
                                                )

                                with FigureSession():
                                    ax = seaborn.lineplot(
                                        x="epoch",
                                        y=ykey,
                                        hue=hue_a,
                                        style=result.index.get_level_values(
                                            level="transformation"
                                        ),
                                        data=result,
                                        err_style="bars",
                                        err_kws={
                                            "errorevery": (max(result.epoch) + 1) // 10,
                                            "ecolor": ".6",
                                            "elinewidth": pyplot.rcParams[
                                                "lines.linewidth"
                                            ]
                                            * 0.6,
                                        },
                                        ci=95,
                                        seed=0,
                                        color=".3",
                                    )

                                    if should_plot_y_log_scale(tag):
                                        set_y_log_scale(ax)

                                    if include_titles:
                                        sup_title = f"{mapping.name}"
                                        if include_timestamp:
                                            sup_title += f", {timestamp.name}"
                                        pyplot.suptitle(sup_title)

                                        support_str = ""
                                        if False:
                                            if tag is TrainingScalars.training_loss:
                                                support_str = (
                                                    f"Training Support ({len(t)})"
                                                )
                                            else:
                                                support_str = (
                                                    f"Validation Support ({len(t)})"
                                                )

                                        pyplot.title(
                                            support_str
                                            + " "
                                            + r"$\bf{"
                                            + str(tag.value.replace("_", "\ "))
                                            + r"}$"
                                            + " "
                                            + "(95% confidence interval)"
                                            + " "
                                            + post_str
                                        )
                                    pyplot.ylabel(latex_clean_label(tag.value))
                                    fix_edge_gridlines()
                                    # auto_post_hatch()
                                    despine_all()
                                    pyplot.tight_layout()
                                    save_embed_fig(
                                        ensure_existence(
                                            agg_path / timestamp.name / mapping.name
                                        )
                                        / Path(tag.value).with_suffix(".pdf").name
                                    )
                                    # print(loc)

    if compute_tensor_agg:
        pass
        """
for tag in TrainingTables:
  pass
for tag in TrainingCurves:
  pass
"""


if __name__ == "__main__":

    training_agg_plot(
        only_latest_load_time=False,
        color_plot=False,
        include_titles=False,
        compute_scalar_agg=True,
        compute_tensor_agg=True,
    )

    system_open_path(EXPORT_RESULTS_PATH / "agg", verbose=True)
