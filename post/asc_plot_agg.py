# !/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 22/06/2020
           """

import os

import numpy
import pandas

# rc('text', usetex=True) # Use latex interpreter
import seaborn
from draugr.visualisation import (
    FigureSession,
    SubplotSession,
    auto_post_hatch,
    fix_edge_gridlines,
)
from matplotlib import pyplot  # ,rc
from sklearn.metrics import auc

from apppath import ensure_existence, system_open_path
from configs.path_config import EXPORT_RESULTS_PATH
from draugr.tqdm_utilities import progress_bar
from draugr.writers import TestingScalars, TrainingScalars
from draugr.misc_utilities import plot_median_labels
from draugr.visualisation import use_monochrome_style, despine_all

seaborn.set()
seaborn.set_style("ticks", rc={"axes.grid": True})
# seaborn.set_style('whitegrid')

__all__ = []


def training_plot(
    agg_path=ensure_existence(EXPORT_RESULTS_PATH / "agg"),
    results_path=EXPORT_RESULTS_PATH / "csv" / "training",
    only_latest_load_time: bool = False,
    include_timestamp: bool = False,
) -> None:
    # use_monochrome_style()

    if only_latest_load_time:
        max_load_time = max(list(results_path.iterdir()), key=os.path.getctime,)
        results_paths = [max_load_time]
    else:
        results_paths = list(results_path.iterdir())

    for timestamp in progress_bar(results_paths, description="timestamp #"):
        if timestamp.is_dir():
            for mapping in progress_bar(
                list(timestamp.iterdir()), description="mapping #"
            ):
                if mapping.is_dir():
                    df_m = {}
                    for transformation in progress_bar(
                        list(mapping.iterdir()), description="transformation #"
                    ):
                        if transformation.is_dir():
                            dfs = []
                            for seed_ith in progress_bar(
                                list(transformation.rglob("*scalars_*.csv"))
                            ):
                                dfs.append(pandas.read_csv(seed_ith))
                            # df_a = pandas.concat(dfs, axis=0)

                            df_m[transformation.name] = pandas.concat(dfs, axis=0)

                    result = pandas.concat(df_m, axis=0)

                    if numpy.any(result.isna().any()):
                        print(result.columns[mapping.isna().any()].tolist())
                        print(result.isna().any())
                        print(result.loc[:, result.isna().any()])
                        print(f"{mapping} result has nans")
                        raise Exception

                    result = result.droplevel(
                        1
                    )  # Remove redundant level ('mfcc', 0) to ('mfcc')
                    for tag in TrainingScalars:
                        if True or tag is not TrainingScalars.new_best_model:

                            with FigureSession():
                                seaborn.lineplot(
                                    x="epoch",
                                    y=tag.value,
                                    data=result,
                                    seed=0,
                                    color=".3",
                                )
                                sup_title = f"{mapping.name}"
                                if include_timestamp:
                                    sup_title += f", {timestamp.name}"
                                pyplot.suptitle(sup_title)

                                support_str = ""
                                if False:
                                    if tag is TrainingScalars.training_loss:
                                        support_str = f"Training Support ({len(t)})"
                                    else:
                                        support_str = f"Validation Support ({len(t)})"

                                pyplot.title(
                                    support_str
                                    + " "
                                    + r"$\bf{"
                                    + str(tag.value.replace("_", "\ "))
                                    + r"}$"
                                    + " "
                                    + "(95% confidence interval)"
                                )
                                fix_edge_gridlines()
                                # auto_post_hatch()
                                despine_all()
                                pyplot.savefig(
                                    ensure_existence(agg_path / mapping.name)
                                    / tag.value
                                )


def stesting_plot(
    agg_path=ensure_existence(EXPORT_RESULTS_PATH / "agg"),
    results_path=EXPORT_RESULTS_PATH / "csv" / "testing",
    compute_scalar_agg: bool = True,
    compute_tensor_agg: bool = True,
    only_latest_load_time: bool = False,
    include_timestamp: bool = False,
) -> None:
    # use_monochrome_style()
    if only_latest_load_time:
        max_load_time = max(list(results_path.iterdir()), key=os.path.getctime,)
        results_paths = [max_load_time]
    else:
        results_paths = list(results_path.iterdir())

    if compute_scalar_agg:
        showfliers = False

        for timestamp in progress_bar(results_paths, description="timestamp #"):
            if timestamp.is_dir():
                for mapping in progress_bar(
                    list(timestamp.iterdir()), description="mapping #"
                ):
                    if mapping.is_dir():
                        df_m = {}
                        for transformation in progress_bar(
                            list(mapping.iterdir()), description="transformation #"
                        ):
                            if transformation.is_dir():
                                dfs = []
                                for seed_ith in progress_bar(
                                    list(transformation.rglob("*scalars_*.csv"))
                                ):
                                    dfs.append(pandas.read_csv(seed_ith))
                                # df_a = reduce(lambda df1,df2: pandas.merge(df1,df2,on='epoch'), dfs)

                                df_m[transformation.name] = pandas.concat(dfs, axis=0)

                        result = pandas.concat(df_m, axis=0)

                        if numpy.any(result.isna().any()):
                            print(result.columns[mapping.isna().any()].tolist())
                            print(result.isna().any())
                            print(result.loc[:, result.isna().any()])
                            print(f"{mapping} result has nans")
                            raise Exception

                        result = result.droplevel(
                            1
                        )  # Remove redundant level ('mfcc', 0) to ('mfcc')
                        for tag in TestingScalars:
                            with FigureSession():
                                ax = seaborn.boxplot(
                                    x=result.index,
                                    y=tag.value,
                                    data=result,
                                    showfliers=showfliers,
                                    color=".3",
                                )
                                plot_median_labels(ax.axes, has_fliers=showfliers)
                                sup_title = f"{mapping.name}"
                                if include_timestamp:
                                    sup_title += f", {timestamp.name}"
                                pyplot.suptitle(sup_title)

                                support_str = ""
                                if False:
                                    support_str = f"Test Support ({len(t)})"
                                pyplot.title(
                                    support_str
                                    + " "
                                    + r"$\bf{"
                                    + str(tag.value.replace("_", "\ "))
                                    + r"}$"
                                )
                                fix_edge_gridlines()
                                # auto_post_hatch()
                                despine_all(ax)
                                pyplot.savefig(
                                    ensure_existence(agg_path / mapping.name)
                                    / tag.value
                                )

    if compute_tensor_agg:
        for timestamp in progress_bar(results_paths, description="timestamp #"):
            if timestamp.is_dir():
                for mapping in progress_bar(
                    list(timestamp.iterdir()), description="mapping #"
                ):
                    if mapping.is_dir():
                        df_m = {}
                        for transformation in progress_bar(
                            list(mapping.iterdir()), description="transformation #"
                        ):
                            if transformation.is_dir():
                                dfs = []
                                for seed_ith in progress_bar(
                                    list(transformation.rglob("*tensors_*.csv"))
                                ):
                                    dddd = pandas.read_csv(seed_ith)
                                    for (
                                        c
                                    ) in (
                                        dddd.columns
                                    ):  # MAKE pandas interpret string representation of list as a list
                                        dddd[c] = pandas.eval(dddd[c])
                                    dfs.append(dddd)
                                df_m[transformation.name] = pandas.concat(
                                    dfs, axis=0
                                )  # .reset_index()

                        result = pandas.concat(df_m)

                        serified_dict = {}
                        for col_ in result.columns[1:]:
                            serified = (
                                result[col_].apply(pandas.Series).stack().reset_index()
                            )
                            serified = serified.drop(columns=["level_1"])
                            serified = serified.rename(
                                columns={
                                    "level_0": "transformation",
                                    "level_2": "threshold_idx",
                                    0: col_,
                                }
                            )
                            serified = serified.set_index("threshold_idx")
                            serified_dict[col_] = serified

                        merged_df = pandas.concat(serified_dict.values(), axis=1)
                        merged_df = merged_df.T.drop_duplicates(
                            keep="first"
                        ).T  # drop duplicate columns but messes up datatype

                        for c in merged_df.columns:  # fix datatypes
                            if c is not "transformation":
                                merged_df[c] = pandas.to_numeric(merged_df[c])

                        chop_off_percentage = 0.1
                        # chop_off_size = int(len(merged_df)*chop_off_percentage)
                        # merged_df=merged_df.loc[chop_off_size:-chop_off_size]

                        # merged_df = merged_df[merged_df['test_precision_recall_recall'] > chop_off_percentage]
                        # merged_df = merged_df[merged_df['test_precision_recall_recall'] < 1.0 - chop_off_percentage]

                        with SubplotSession(figsize=(10, 10)) as a:
                            fig, (ax, *_) = a
                            for t_name, transformation_df in progress_bar(
                                merged_df.groupby("transformation")
                            ):
                                agg_group = transformation_df.groupby("threshold_idx")

                                mean_vals = agg_group.mean()
                                upper_vals = agg_group.quantile(0.975)
                                lower_vals = agg_group.quantile(0.025)
                                mean_area = auc(
                                    mean_vals["test_precision_recall_recall"],
                                    mean_vals["test_precision_recall_precision"],
                                )

                                (m_l,) = ax.plot(
                                    mean_vals["test_precision_recall_recall"],
                                    mean_vals["test_precision_recall_precision"],
                                    label=f"{t_name} (mean area: {mean_area})",
                                    color=".3",
                                )
                                pyplot.xlabel("recall")
                                pyplot.ylabel("precision")
                                ax.fill_between(
                                    mean_vals["test_precision_recall_recall"],
                                    upper_vals["test_precision_recall_precision"],
                                    lower_vals["test_precision_recall_precision"],
                                    where=upper_vals["test_precision_recall_precision"]
                                    > lower_vals["test_precision_recall_precision"],
                                    facecolor=m_l.get_color(),
                                    alpha=0.3,
                                    interpolate=True,
                                )
                                if False:
                                    for TP, coords in zip(
                                        mean_vals[
                                            "test_precision_recall_true_positive_counts"
                                        ],
                                        zip(
                                            mean_vals["test_precision_recall_recall"],
                                            mean_vals[
                                                "test_precision_recall_precision"
                                            ],
                                        ),
                                    ):
                                        ax.annotate(f"{TP}", xy=coords)

                            sup_title = f"{mapping.name}"
                            if include_timestamp:
                                sup_title += f", {timestamp.name}"
                            pyplot.suptitle(sup_title)
                            support_str = ""
                            if False:
                                support_str = f"Test Support ({len(t)})"

                            pyplot.title(
                                support_str
                                + " "
                                + r"$\bf{"
                                + "Pr Curve"
                                + r"}$"
                                + " "
                                + "(95% confidence interval)"
                            )
                            pyplot.legend()
                            fix_edge_gridlines()
                            # auto_post_hatch()
                            despine_all(ax)
                            pyplot.savefig(ensure_existence(agg_path / mapping.name))


def compute_agg_plots(only_latest_load_time: bool = False):
    training_plot(only_latest_load_time=only_latest_load_time)
    stesting_plot(only_latest_load_time=only_latest_load_time)


if __name__ == "__main__":
    compute_agg_plots(only_latest_load_time=True)
    system_open_path(EXPORT_RESULTS_PATH / "agg", verbose=True)
