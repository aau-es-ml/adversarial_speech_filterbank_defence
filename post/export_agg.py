# !/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 22/06/2020
           """

# rc('text', usetex=True) # Use latex interpreter

from apppath import system_open_path

from configs.path_config import EXPORT_RESULTS_PATH
from post.export_testing_agg import stesting_agg_plot
from post.export_training_agg import training_agg_plot

# seaborn.set()
# seaborn.set_style("ticks", rc={"axes.grid":True, "text.usetex":True})

# seaborn.set_style('whitegrid')
# pyplot.style.use('tex')

__all__ = []


def compute_agg_plots(
    *,
    only_latest_load_time: bool = False,
    color_plot: bool = False,
    include_titles: bool = False,
) -> None:
    if True:
        training_agg_plot(
            only_latest_load_time=only_latest_load_time,
            color_plot=color_plot,
            include_titles=include_titles,
        )
    if True:
        stesting_agg_plot(
            only_latest_load_time=only_latest_load_time,
            color_plot=color_plot,
            include_titles=include_titles,
        )


if __name__ == "__main__":
    if False:
        compute_agg_plots(only_latest_load_time=True, color_plot=False)
    else:
        stesting_agg_plot(
            only_latest_load_time=False,
            color_plot=False,
            include_titles=False,
            compute_scalar_agg=True,
            compute_tensor_agg=True,
        )

    system_open_path(EXPORT_RESULTS_PATH / "agg", verbose=True)
