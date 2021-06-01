#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 22/06/2020
           """

import os
from pathlib import Path

import numpy
import pandas
from apppath import ensure_existence, system_open_path
from draugr.pandas_utilities import (
    pandas_mean_std,
    nested_dict_to_three_level_column_df,
    pandas_mean_std_to_str,
    pandas_format_bold_max_row_latex,
    color_highlight_extreme,
    pandas_to_latex_clean,
)

from draugr.tqdm_utilities import progress_bar

from configs.path_config import EXPORT_RESULTS_PATH

__all__ = ["extract_cross_latex_table"]


class DuplicateError(Exception):
    pass


def extract_cross_latex_table(
    agg_path=ensure_existence(EXPORT_RESULTS_PATH / "latex"),
    results_path=EXPORT_RESULTS_PATH / "csv" / "testing",
    only_latest_load_time: bool = False,
    merged: bool = True,
) -> None:
    if only_latest_load_time:
        max_load_time = max(
            list(results_path.iterdir()),
            key=os.path.getctime,
        )
        results_paths = [max_load_time]
    else:
        results_paths = list(results_path.iterdir())

    merged_dfs = {}
    merged_dfs_snr = {}

    for timestamp in progress_bar(results_paths, description="timestamp #"):
        if timestamp.is_dir():
            seperate_dfs = {}
            seperate_dfs_snr = {}
            for mapping in progress_bar(
                list(timestamp.iterdir()), description="mapping #"
            ):
                if mapping.is_dir():
                    seperate_dfs[mapping.name] = {}
                    if "snr" in mapping.name:
                        seperate_dfs_snr[mapping.name] = {}

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

                                    df_transformations[
                                        transformation.name
                                    ] = pandas.concat(dfs, axis=0)

                            df_result = pandas.concat(df_transformations, axis=0)

                            if numpy.any(df_result.isna().any()):
                                print(df_result.columns[mapping.isna().any()].tolist())
                                print(df_result.isna().any())
                                print(df_result.loc[:, df_result.isna().any()])
                                print(f"{mapping} result has nans")
                                raise Exception

                            df_result = df_result.droplevel(
                                1
                            )  # Remove redundant level ('mfcc', 0) to ('mfcc')

                            seperate_dfs[mapping.name][test_set.name] = pandas_mean_std(
                                df_result.reset_index().drop("epoch", axis=1), "index"
                            ).rename_axis("filter_bank", axis="index")

                            if mapping.name in seperate_dfs_snr:
                                seperate_dfs_snr[mapping.name][test_set.name] = (
                                    df_result.sort_index()
                                    .rename_axis("filter_bank", axis="index")
                                    .drop("epoch", axis=1)
                                )

                    seperate_dfs[mapping.name] = pandas.concat(
                        seperate_dfs[mapping.name]
                    )

                    if mapping.name in seperate_dfs_snr:
                        seperate_dfs_snr[mapping.name] = pandas.concat(
                            seperate_dfs_snr[mapping.name]
                        )

            if not merged:
                save_table(agg_path, seperate_dfs, timestamp.name, seperate_dfs_snr)
            else:
                for k, v in seperate_dfs.items():
                    if k not in merged_dfs:
                        merged_dfs[k] = v
                    else:
                        raise DuplicateError
                for k, v in seperate_dfs_snr.items():
                    if k not in merged_dfs_snr:
                        merged_dfs_snr[k] = v
                    else:
                        raise DuplicateError

    if merged:
        save_table(agg_path, merged_dfs, "merged", merged_dfs_snr)


def save_table(agg_path, b, name, merged_dfs_snr):
    selected_columns = pandas.concat(b).loc(axis=1)[
        "test_receiver_operator_characteristic_auc", ["mean", "std"]
    ]
    selected_columns.style.apply(color_highlight_extreme, axis=1)

    if merged_dfs_snr:
        ss = pandas.concat(
            merged_dfs_snr
        )  # ['test_receiver_operator_characteristic_auc']
        if False:  # unnecessary
            skoot = ss.index.get_level_values(0).str.contains("snr")
            skeet = ss.index.get_level_values(1).str.contains("snr")
            ss = ss.iloc[skoot & skeet]
        # ss.index = pandas.MultiIndex.from_tuples(ss.index, names=['train set', 'test set'])

        if len(ss):
            # ss = pandas_mean_std(ss,'filter_bank')
            # pivoted__sss.index = pandas.MultiIndex.from_tuples(pivoted__sss.index, names=['train set', 'test set'])
            with open(
                ensure_existence(agg_path / name)
                / Path("cross_snr").with_suffix(".tex").name,
                "w",
            ) as f:
                f.write(
                    pandas_to_latex_clean(
                        pandas_mean_std_to_str(
                            ss.pivot_table(
                                values="test_receiver_operator_characteristic_auc",
                                index=ss.index.droplevel(-1).map(
                                    "{0[0]:.20}".format
                                ),  #  'equal_mapping_###*'
                                columns="filter_bank",
                                aggfunc={
                                    "test_receiver_operator_characteristic_auc": [
                                        "mean",
                                        "std",
                                    ]
                                },  # numpy.mean, numpy.std
                            ),
                            precision=3,
                            level=0,
                        )
                    )
                )
            with open(
                ensure_existence(agg_path / name)
                / Path("cross_snr_n_noise").with_suffix(".tex").name,
                "w",
            ) as f:
                f.write(
                    pandas_to_latex_clean(
                        pandas_mean_std_to_str(
                            ss.pivot_table(
                                values="test_receiver_operator_characteristic_auc",
                                index=ss.index.droplevel(-1).map(
                                    "{0[0]:.13}".format
                                ),  # cross noise 'equal_mapping*'
                                columns="filter_bank",
                                aggfunc={
                                    "test_receiver_operator_characteristic_auc": [
                                        "mean",
                                        "std",
                                    ]
                                },  # numpy.mean, numpy.std
                            ),
                            precision=3,
                            level=0,
                        )
                    )
                )

        # selected_columns.
        # TODO EXPORT SNR MEANS
    mean_std_str = pandas_mean_std_to_str(selected_columns, precision=3)
    """
  mean_std_pivot = selecteed.pivot_table(values='test_receiver_operator_characteristic_auc',
                                         index=selecteed.index.droplevel(-1),
                                         columns='filter_bank',
                                         aggfunc='first'
                                         )
 """
    mean_std_str_pivot = mean_std_str.pivot_table(
        values="test_receiver_operator_characteristic_auc",
        index=mean_std_str.index.droplevel(-1),
        columns="filter_bank",
        aggfunc="first",
    )
    mean_std_str_pivot.index = pandas.MultiIndex.from_tuples(
        mean_std_str_pivot.index, names=["train set", "test set"]
    )
    # mean_std_str_pivot.index.
    # mean_std_str_pivot.rename_axis('experiment', axis='index')
    # print(mean_std_str_pivot.columns)
    cleaned_latex_str = pandas_to_latex_clean(mean_std_str_pivot, header_rotation=0)
    with open(
        ensure_existence(agg_path / name) / Path("table").with_suffix(".tex").name,
        "w",
    ) as f:
        f.write(cleaned_latex_str)


if __name__ == "__main__":
    extract_cross_latex_table()
    system_open_path(EXPORT_RESULTS_PATH / "latex", verbose=True)
