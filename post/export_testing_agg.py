#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 09-04-2021
           """

import os
from math import sqrt
from pathlib import Path

import numpy
import pandas
import seaborn
from apppath import ensure_existence, system_open_path
from draugr.misc_utilities import plot_median_labels
from draugr.scipy_utilities import min_decimation_subsample
from draugr.tqdm_utilities import progress_bar
from draugr.visualisation import (
    FigureSession,
    MonoChromeStyleSession,
    SubplotSession,
    annotate_point,
    decolorise_plot,
    despine_all,
    fix_edge_gridlines,
    latex_clean_label,
    make_errorbar_legend,
    monochrome_line_no_marker_cycler,
    save_embed_fig,
)
from draugr.writers import TestingScalars
from matplotlib import pyplot
from sklearn.metrics import auc
from warg import ContextWrapper, GDKC

from configs import EXPORT_RESULTS_PATH


def stesting_agg_plot(
    agg_path=ensure_existence(EXPORT_RESULTS_PATH / "agg"),
    results_path=EXPORT_RESULTS_PATH / "csv" / "testing",
    compute_scalar_agg: bool = True,
    compute_tensor_agg: bool = True,
    only_latest_load_time: bool = False,
    include_timestamp: bool = False,
    color_plot: bool = False,
    include_titles: bool = False,
    add_tp_annotations: bool = False,
    annotate_threshold: bool = True,
    export_confusion_matrix: bool = True,
    export_mean_mean_confusion_matrix: bool = False,
    export_csvs: bool = False,
) -> None:
    seaborn.set()
    seaborn.set_style("ticks", rc={"axes.grid": True, "text.usetex": True})
    # use_monochrome_style()
    if only_latest_load_time:
        max_load_time = max(
            list(results_path.iterdir()),
            key=os.path.getctime,
        )
        results_paths = [max_load_time]
    else:
        results_paths = list(results_path.iterdir())

    if compute_scalar_agg:
        show_fliers = False

        for timestamp in progress_bar(results_paths, description="timestamp #"):
            if timestamp.is_dir():
                for mapping in progress_bar(
                    list(timestamp.iterdir()), description="mapping #"
                ):
                    if mapping.is_dir():
                        for test_set in progress_bar(
                            list(mapping.iterdir()), description="test set #"
                        ):
                            if test_set.is_dir():
                                df_transformations = {}
                                for transformation in progress_bar(
                                    list(test_set.iterdir()),
                                    description="transformation #",
                                ):
                                    if transformation.is_dir():
                                        dfs = []
                                        for seed_ith in progress_bar(
                                            list(transformation.rglob("*scalars_*.csv"))
                                        ):
                                            dfs.append(pandas.read_csv(seed_ith))
                                        # df_a = reduce(lambda df1,df2: pandas.merge(df1,df2,on='epoch'), dfs)

                                        df_transformations[
                                            transformation.name
                                        ] = pandas.concat(dfs, axis=0)

                                df_result = pandas.concat(df_transformations, axis=0)

                                if numpy.any(df_result.isna().any()):
                                    print(
                                        df_result.columns[mapping.isna().any()].tolist()
                                    )
                                    print(df_result.isna().any())
                                    print(df_result.loc[:, df_result.isna().any()])
                                    print(f"{mapping} result has nans")
                                    raise Exception

                                df_result = df_result.droplevel(
                                    1
                                )  # Remove redundant level ('mfcc', 0) to ('mfcc')

                                for tag in TestingScalars:
                                    color_a = None
                                    if not color_plot:
                                        color_a = "white"
                                    with ContextWrapper(
                                        GDKC(
                                            MonoChromeStyleSession,
                                            prop_cycler=monochrome_line_no_marker_cycler,
                                        ),
                                        not color_plot,
                                    ):
                                        with FigureSession():
                                            prc_ax = seaborn.boxplot(
                                                x=df_result.index,
                                                y=tag.value,
                                                data=df_result,
                                                showfliers=show_fliers,
                                                color=color_a,  # color=".3",
                                            )

                                            if not color_plot:
                                                decolorise_plot(prc_ax)

                                            plot_median_labels(
                                                prc_ax.axes, has_fliers=show_fliers
                                            )
                                            if include_titles:
                                                sup_title = f"{mapping.name}"
                                                if include_timestamp:
                                                    sup_title += f", {timestamp.name}"
                                                pyplot.suptitle(sup_title)

                                                support_str = ""
                                                if False:
                                                    support_str = (
                                                        f"Test Support ({len(t)})"
                                                    )
                                                pyplot.title(
                                                    support_str
                                                    + " "
                                                    + r"$\bf{"
                                                    + str(tag.value.replace("_", "\ "))
                                                    + r"}$"
                                                )

                                            pyplot.ylabel(latex_clean_label(tag.value))

                                            fix_edge_gridlines()
                                            # auto_post_hatch()
                                            despine_all(prc_ax)
                                            pyplot.tight_layout()
                                            save_embed_fig(
                                                ensure_existence(
                                                    agg_path
                                                    / timestamp.name
                                                    / mapping.name
                                                    / test_set.name
                                                )
                                                / Path(tag.value)
                                                .with_suffix(".pdf")
                                                .name
                                            )

    if compute_tensor_agg:
        for timestamp in progress_bar(results_paths, description="timestamp #"):
            if timestamp.is_dir():
                for mapping in progress_bar(
                    list(timestamp.iterdir()), description="mapping #"
                ):
                    if mapping.is_dir():
                        for test_set in progress_bar(
                            list(mapping.iterdir()), description="mapping #"
                        ):
                            if test_set.is_dir():
                                df_transformations = {}
                                for transformation in progress_bar(
                                    list(test_set.iterdir()),
                                    description="transformation #",
                                ):
                                    if transformation.is_dir():
                                        dfs = []
                                        for seed_ith in progress_bar(
                                            list(transformation.rglob("*tensors_*.csv"))
                                        ):
                                            seed_tensor = pandas.read_csv(seed_ith)
                                            for (
                                                column
                                            ) in (
                                                seed_tensor.columns
                                            ):  # MAKE pandas interpret string representation of list as a list
                                                seed_tensor[column] = pandas.eval(
                                                    seed_tensor[column]
                                                )
                                            dfs.append(seed_tensor)
                                        df_transformations[
                                            transformation.name
                                        ] = pandas.concat(
                                            dfs, axis=0
                                        )  # .reset_index()

                                df_result = pandas.concat(df_transformations)

                                serialised_dict = {}
                                for col_ in df_result.columns[1:]:
                                    serialised_df = (
                                        df_result[col_]
                                        .apply(pandas.Series)
                                        .stack()
                                        .reset_index()
                                    )
                                    serialised_df = serialised_df.drop(
                                        columns=["level_1"]
                                    )
                                    serialised_df = serialised_df.rename(
                                        columns={
                                            "level_0": "transformation",
                                            "level_2": "threshold_idx",
                                            0: col_,
                                        }
                                    )
                                    serialised_df = serialised_df.set_index(
                                        "threshold_idx"
                                    )
                                    serialised_dict[col_] = serialised_df

                                merged_df = pandas.concat(
                                    serialised_dict.values(), axis=1
                                )
                                merged_df = merged_df.T.drop_duplicates(
                                    keep="first"
                                ).T  # drop duplicate columns but messes up datatype

                                for column in merged_df.columns:  # fix datatypes
                                    if column != "transformation":
                                        merged_df[column] = pandas.to_numeric(
                                            merged_df[column]
                                        )

                                # chop_off_percentage = 0.1
                                # chop_off_size = int(len(merged_df)*chop_off_percentage)
                                # merged_df=merged_df.loc[chop_off_size:-chop_off_size]

                                # merged_df = merged_df[merged_df['test_precision_recall_recall'] > chop_off_percentage]
                                # merged_df = merged_df[merged_df['test_precision_recall_recall'] < 1.0 - chop_off_percentage]

                                with ContextWrapper(
                                    GDKC(
                                        MonoChromeStyleSession,
                                        prop_cycler=monochrome_line_no_marker_cycler,
                                    ),
                                    not color_plot,
                                ):
                                    with SubplotSession() as pr_curve_subplot:
                                        prc_fig, (prc_ax, *_) = pr_curve_subplot
                                        """
err_style = VisualisationErrorStyle.Bar
if color_plot:
err_style = VisualisationErrorStyle.Band
"""
                                        ran_once = False
                                        min_xs = (0, 1)
                                        min_ys = (0, 1)
                                        for t_name, transformation_df in progress_bar(
                                            merged_df.groupby("transformation"),
                                            description=f"groupby",
                                        ):
                                            agg_group = transformation_df.groupby(
                                                "threshold_idx"
                                            )

                                            mean_vals = agg_group.mean()
                                            upper_vals = agg_group.quantile(0.95)
                                            lower_vals = agg_group.quantile(0.05)

                                            areas = list(
                                                auc(
                                                    x["test_precision_recall_recall"],
                                                    x[
                                                        "test_precision_recall_precision"
                                                    ],
                                                )
                                                for x in (
                                                    mean_vals,
                                                    upper_vals,
                                                    lower_vals,
                                                )
                                            )

                                            areas[1] = areas[1] - areas[0]
                                            areas[2] = areas[0] - areas[2]

                                            len_x = len(
                                                mean_vals[
                                                    "test_precision_recall_recall"
                                                ]
                                            )
                                            thresholds = [
                                                i / float(len_x - 1)
                                                for i in range(len_x)
                                            ]
                                            if not ran_once:  # Only once
                                                mvs = agg_group.max()
                                                total_num_samples = int(
                                                    mvs[
                                                        "test_precision_recall_true_positive_counts"
                                                    ][0]
                                                    + mvs[
                                                        "test_precision_recall_false_positive_counts"
                                                    ][0]
                                                    + mvs[
                                                        "test_precision_recall_true_negative_counts"
                                                    ][0]
                                                    + mvs[
                                                        "test_precision_recall_false_negative_counts"
                                                    ][0]
                                                )
                                                prc_ax.text(
                                                    1.0,
                                                    -0.06,
                                                    f"num samples: {total_num_samples}",
                                                    horizontalalignment="right",
                                                    verticalalignment="top",
                                                    transform=prc_ax.transAxes,
                                                )
                                                ran_once = True  # Flag, bad design!

                                            if (
                                                export_confusion_matrix
                                                or export_mean_mean_confusion_matrix
                                            ):
                                                cf_mat = list(
                                                    (
                                                        mean_vals[x],
                                                        upper_vals[x] - mean_vals[x],
                                                        mean_vals[x] - lower_vals[x],
                                                    )
                                                    for x in (
                                                        "test_precision_recall_true_positive_counts",
                                                        "test_precision_recall_false_positive_counts",
                                                        "test_precision_recall_false_negative_counts",
                                                        "test_precision_recall_true_negative_counts",
                                                    )
                                                )
                                                cf_mat = numpy.array(cf_mat)[
                                                    :, 0
                                                ]  # discard others as they are not used
                                                epath = ensure_existence(
                                                    agg_path
                                                    / timestamp.name
                                                    / mapping.name
                                                    / test_set.name
                                                )
                                                if export_mean_mean_confusion_matrix:
                                                    mmcfm_name = (
                                                        ensure_existence(
                                                            epath
                                                            / f"mean_threshold_mean_seed_confusion_matrix"
                                                        )
                                                        / t_name
                                                    )
                                                    mean_mean_cf_mat = numpy.mean(
                                                        cf_mat, -1
                                                    )
                                                    mean_mean_cf_mat /= numpy.sum(
                                                        mean_mean_cf_mat
                                                    )
                                                    # mean_cf_mat_res = mean_cf_mat.reshape(2, 2)
                                                    # print(numpy.sum(mean_cf_mat_res)) # Kind of half as... result, does not sum to one
                                                    sqr = int(
                                                        sqrt(len(mean_mean_cf_mat))
                                                    )
                                                    mean_mean_cf_mat = (
                                                        mean_mean_cf_mat.reshape(
                                                            (sqr, sqr)
                                                        )
                                                    )
                                                    # mean_mean_cf_mat = numpy.expand_dims(mean_mean_cf_mat, 0)
                                                    my_df = pandas.DataFrame(
                                                        mean_mean_cf_mat,
                                                        columns=("t", "f"),
                                                        index=("p", "n")
                                                        # columns=('tp', 'fp', 'fn', 'tn')
                                                    )
                                                    if export_csvs:
                                                        my_df.to_csv(
                                                            str(
                                                                mmcfm_name.with_suffix(
                                                                    ".csv"
                                                                )
                                                            ),
                                                            index=True,
                                                        )
                                                    if False:
                                                        with SubplotSession() as cfm_subplot:
                                                            (
                                                                cfm_fig,
                                                                (cfm_ax, *_),
                                                            ) = cfm_subplot
                                                            seaborn.heatmap(
                                                                my_df,
                                                                annot=True,
                                                                # annot_kws={"size": 16},
                                                                cmap="binary",
                                                                # color=".3",
                                                            )
                                                            fix_edge_gridlines(cfm_ax)
                                                            # auto_post_hatch()
                                                            despine_all(cfm_ax)
                                                            # pyplot.tight_layout()
                                                            save_embed_fig(
                                                                str(
                                                                    mmcfm_name.with_suffix(
                                                                        ".pdf"
                                                                    )
                                                                ),
                                                            )
                                                            # numpy.savetxt(ensure_existence(agg_path / timestamp.name / mapping.name) / f"{test_set.name}_mean_confusion_matrix.csv", mean_cf_mat, delimiter=",", header=('tp','fp','fn','tn'))
                                                if False:
                                                    pass  # .... TODO: one cf for each line
                                                if export_confusion_matrix:
                                                    half_threshold_cf_mat = cf_mat[
                                                        :, len_x // 2
                                                    ]  # 0.5 threshold
                                                    mean_cf_mat = (
                                                        half_threshold_cf_mat
                                                        / numpy.sum(
                                                            half_threshold_cf_mat
                                                        )
                                                    )
                                                    mcfm_name = (
                                                        ensure_existence(
                                                            epath
                                                            / f"threshold_05_mean_seed_confusion_matrix"
                                                        )
                                                        / t_name
                                                    )
                                                    sqr = int(sqrt(len(mean_cf_mat)))
                                                    mean_cf_mat = mean_cf_mat.reshape(
                                                        (sqr, sqr)
                                                    )
                                                    # mean_cf_mat=numpy.expand_dims(mean_cf_mat, 0)
                                                    my_df = pandas.DataFrame(
                                                        mean_cf_mat,
                                                        columns=(
                                                            "adv-model",
                                                            "non-adv-model",
                                                        ),
                                                        index=("adv", "non-adv"),
                                                    )
                                                    if export_csvs:
                                                        my_df.to_csv(
                                                            str(
                                                                mcfm_name.with_suffix(
                                                                    ".csv"
                                                                )
                                                            ),
                                                            index=True,
                                                        )
                                                    if True:
                                                        with SubplotSession(
                                                            figsize=(3, 2)
                                                        ) as cfm_subplot:
                                                            (
                                                                cfm_fig,
                                                                (cfm_ax, *_),
                                                            ) = cfm_subplot
                                                            seaborn.heatmap(
                                                                my_df,
                                                                annot=True,
                                                                # annot_kws={"size": 16},
                                                                cmap="binary",
                                                                linewidth=1,
                                                                linecolor="w",
                                                                square=True,
                                                                # color=".3",
                                                                ax=cfm_ax,
                                                            )
                                                            fix_edge_gridlines(cfm_ax)
                                                            # auto_post_hatch()
                                                            despine_all(cfm_ax)
                                                            cfm_ax.text(
                                                                0.5,
                                                                1.06,
                                                                (
                                                                    f"num samples: {total_num_samples}"
                                                                ),
                                                                horizontalalignment="center",
                                                                verticalalignment="bottom",
                                                                transform=cfm_ax.transAxes,
                                                            )
                                                            prc_fig.tight_layout()
                                                            save_embed_fig(
                                                                str(
                                                                    mcfm_name.with_suffix(
                                                                        ".pdf"
                                                                    )
                                                                ),
                                                            )

                                            l = (
                                                len(
                                                    mean_vals[
                                                        "test_precision_recall_recall"
                                                    ]
                                                )
                                                - 1
                                                - 4
                                            )  # ith last, is the ith first, list is inverted.
                                            if (
                                                mean_vals[
                                                    "test_precision_recall_recall"
                                                ][l]
                                                < min_xs[0]
                                            ):  # chop-off uninteresting part
                                                min_xs = (
                                                    mean_vals[
                                                        "test_precision_recall_recall"
                                                    ][l],
                                                    mean_vals[
                                                        "test_precision_recall_precision"
                                                    ][l],
                                                )
                                            if (
                                                mean_vals[
                                                    "test_precision_recall_precision"
                                                ][l]
                                                < min_ys[1]
                                            ):  # chop-off uninteresting part
                                                min_ys = (
                                                    mean_vals[
                                                        "test_precision_recall_recall"
                                                    ][l],
                                                    mean_vals[
                                                        "test_precision_recall_precision"
                                                    ][l],
                                                )

                                            plot_label = f"{t_name}"
                                            if (
                                                False
                                            ):  # AUC Very sensitive to outliers, MAYBE look into pr-gain
                                                aug = ""
                                                if False:
                                                    aug = f"(+{areas[1]:.2f}/-{areas[2]:.2f})"  # Does not work properly
                                                plot_label += (
                                                    f" (mean area: {areas[0]:.2f}{aug})"
                                                )
                                            # with NoOutlineSession():
                                            if color_plot:
                                                # (m_l,) = ax.plot(mean_vals["test_precision_recall_recall"], mean_vals["test_precision_recall_precision"], label=f"{t_name} (mean area: {mean_area})",
                                                #                 marker=False,
                                                # err_style=err_style.value
                                                # color=".3",
                                                #                 )
                                                m_l = seaborn.lineplot(
                                                    x=mean_vals[
                                                        "test_precision_recall_recall"
                                                    ],
                                                    y=mean_vals[
                                                        "test_precision_recall_precision"
                                                    ],
                                                    label=plot_label,
                                                    markers=False,  # Seaborn map disable
                                                    marker="None",  # mpl disable
                                                    # color='none'
                                                    # err_style=err_style.value
                                                    # color=".3",
                                                )

                                                if (
                                                    True
                                                ):  # NOTE: Inferior, Only one dimensional error on precision
                                                    for l in (
                                                        m_l.get_lines()[-1],
                                                    ):  # Latest line plot index zero indexed
                                                        if (
                                                            color_plot
                                                        ):  # we need to do error bands ourselves
                                                            face_color = (
                                                                "none"  # l.get_color()
                                                            )
                                                            edge_color = l.get_color()
                                                            prc_ax.fill_between(
                                                                mean_vals[
                                                                    "test_precision_recall_recall"
                                                                ],
                                                                upper_vals[
                                                                    "test_precision_recall_precision"
                                                                ],
                                                                lower_vals[
                                                                    "test_precision_recall_precision"
                                                                ],
                                                                where=upper_vals[
                                                                    "test_precision_recall_precision"
                                                                ]
                                                                > lower_vals[
                                                                    "test_precision_recall_precision"
                                                                ],
                                                                alpha=0.3,
                                                                interpolate=True,
                                                                facecolor=face_color,
                                                                # hatch=next(hatcher),
                                                                edgecolor=edge_color,
                                                                # linewidth=0.0
                                                            )
                                                        else:
                                                            if False:
                                                                # seaborn.pointplot() # we need to do error bars ourselves

                                                                # xs = l.get_xdata()
                                                                # ys = l.get_ydata()
                                                                xs = mean_vals[
                                                                    "test_precision_recall_recall"
                                                                ]
                                                                ys = mean_vals[
                                                                    "test_precision_recall_precision"
                                                                ]
                                                                uys = (
                                                                    upper_vals[
                                                                        "test_precision_recall_precision"
                                                                    ]
                                                                    - mean_vals[
                                                                        "test_precision_recall_precision"
                                                                    ]
                                                                )
                                                                lys = (
                                                                    mean_vals[
                                                                        "test_precision_recall_precision"
                                                                    ]
                                                                    - lower_vals[
                                                                        "test_precision_recall_precision"
                                                                    ]
                                                                )
                                                                d_idx = min_decimation_subsample(
                                                                    xs,
                                                                    return_indices=True,
                                                                    decimation_factor=len(
                                                                        xs
                                                                    )
                                                                    // 5,
                                                                )
                                                                eb = prc_ax.errorbar(
                                                                    xs[d_idx],
                                                                    ys[d_idx],
                                                                    yerr=(
                                                                        lys[d_idx],
                                                                        uys[d_idx],
                                                                    ),
                                                                    # linestyle=l.get_linestyle() # TODO: SEEMS BROKEN
                                                                )
                                                                eb[-1][0].set_linestyle(
                                                                    l.get_linestyle()
                                                                )  # FIX to the issue above
                                                    if add_tp_annotations:
                                                        for TP, coords in zip(
                                                            mean_vals[
                                                                "test_precision_recall_true_positive_counts"
                                                            ],
                                                            zip(
                                                                mean_vals[
                                                                    "test_precision_recall_recall"
                                                                ],
                                                                mean_vals[
                                                                    "test_precision_recall_precision"
                                                                ],
                                                            ),
                                                        ):
                                                            prc_ax.annotate(
                                                                f"{TP}", xy=coords
                                                            )
                                            else:
                                                kx, ky = (
                                                    "test_precision_recall_recall",
                                                    "test_precision_recall_precision",
                                                )

                                                xs = mean_vals[kx]
                                                uxs = upper_vals[kx] - mean_vals[kx]
                                                lxs = mean_vals[kx] - lower_vals[kx]
                                                ys = mean_vals[ky]
                                                uys = upper_vals[ky] - mean_vals[ky]
                                                lys = mean_vals[ky] - lower_vals[ky]

                                                plot_error_every = len_x // 3

                                                eb = prc_ax.errorbar(
                                                    x=xs,
                                                    y=ys,
                                                    xerr=[lxs, uxs],
                                                    yerr=[lys, uys],
                                                    label=plot_label,
                                                    ecolor=".6",
                                                    elinewidth=pyplot.rcParams[
                                                        "lines.linewidth"
                                                    ]
                                                    * 0.6,
                                                    errorevery=plot_error_every,
                                                )
                                                if annotate_threshold:
                                                    for x, y, t in zip(
                                                        xs[::plot_error_every],
                                                        ys[::plot_error_every],
                                                        thresholds[::plot_error_every],
                                                    ):
                                                        annotate_point(prc_ax, x, y, t)
                                                for ei in eb[-1]:
                                                    ei.set_linestyle(
                                                        eb[0].get_linestyle()
                                                    )  # set matching linestyle

                                        min_x_lim = int(min_xs[0] * 10) / 10
                                        min_y_lim = int(min_ys[0] * 10) / 10

                                        pyplot.xlim(min_x_lim, 1)
                                        pyplot.ylim(min_y_lim, 1)
                                        pyplot.xlabel("recall")
                                        pyplot.ylabel("precision")
                                        if include_titles:
                                            sup_title = f"{mapping.name}"
                                            if include_timestamp:
                                                sup_title += f", {timestamp.name}"
                                            pyplot.suptitle(sup_title)
                                            support_str = ""
                                            if False:
                                                support_str = f"Test Support ({len(t)})"  # num samples

                                            pyplot.title(
                                                support_str
                                                + " "
                                                + r"$\bf{"
                                                + "Pr Curve"
                                                + r"}$"
                                                + " "
                                                + "(95% confidence interval)"
                                            )

                                        make_errorbar_legend(prc_ax)
                                        fix_edge_gridlines(prc_ax)
                                        # auto_post_hatch()
                                        despine_all(prc_ax)
                                        # pyplot.tight_layout()
                                        save_embed_fig(
                                            ensure_existence(
                                                agg_path
                                                / timestamp.name
                                                / mapping.name
                                                / test_set.name
                                            )
                                            / "pr_curve.pdf"
                                        )


if __name__ == "__main__":

    stesting_agg_plot(
        only_latest_load_time=True,
        color_plot=False,
        include_titles=False,
        compute_scalar_agg=True,
        compute_tensor_agg=True,
    )

    system_open_path(EXPORT_RESULTS_PATH / "agg", verbose=True)
