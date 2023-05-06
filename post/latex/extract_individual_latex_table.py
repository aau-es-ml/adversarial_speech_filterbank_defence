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
from draugr.pandas_utilities.latex_mean_std import pandas_mean_std_latex_table
from draugr.visualisation import progress_bar

from configs.path_config import EXPORT_RESULTS_PATH

__all__ = ["extract_latex_table"]


def extract_latex_table(
    agg_path=ensure_existence(EXPORT_RESULTS_PATH / "latex"),
    results_path=EXPORT_RESULTS_PATH / "csv" / "testing",
    only_latest_load_time: bool = False,
) -> None:
    if only_latest_load_time:
        max_load_time = max(
            list(results_path.iterdir()),
            key=os.path.getctime,
        )
        results_paths = [max_load_time]
    else:
        results_paths = list(results_path.iterdir())

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
                                print(df_result.columns[mapping.isna().any()].tolist())
                                print(df_result.isna().any())
                                print(df_result.loc[:, df_result.isna().any()])
                                print(f"{mapping} result has nans")
                                raise Exception

                            df_result = df_result.droplevel(
                                1
                            )  # Remove redundant level ('mfcc', 0) to ('mfcc')

                            df_result = df_result.reset_index().drop("epoch", axis=1)

                            with open(
                                ensure_existence(
                                    agg_path
                                    / timestamp.name
                                    / mapping.name
                                    / test_set.name
                                )
                                / Path("table").with_suffix(".tex").name,
                                "w",
                            ) as f:
                                tab_label = "_".join((mapping.name, test_set.name))
                                f.write(
                                    pandas_mean_std_latex_table(
                                        df_result, "index", tab_label
                                    )
                                )


if __name__ == "__main__":
    extract_latex_table(only_latest_load_time=True)
    system_open_path(EXPORT_RESULTS_PATH / "latex", verbose=True)
