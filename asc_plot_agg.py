from pathlib import Path

import pandas
import tqdm
from apppath import ensure_existence
from draugr.writers.standard_tags import TestingScalars, TrainingScalars
from sklearn.metrics import auc

__all__ = []

from asc_utilities.evaluation import plot_median_labels

if __name__ == '__main__':

  def main():
    base_res_path = Path.cwd() / 'exclude'

    def training_plot():
      agg_path = ensure_existence(base_res_path / 'agg')
      results_path = base_res_path / 'results' / 'training'

      from matplotlib import pyplot  # ,rc
      # rc('text', usetex=True) # Use latex interpreter
      import seaborn
      seaborn.set()

      for timestamp in tqdm.tqdm(list(results_path.iterdir())):
        if timestamp.is_dir():
          for mapping in tqdm.tqdm(list(timestamp.iterdir())):
            if mapping.is_dir():
              for transformation in tqdm.tqdm(list(mapping.iterdir())):
                if transformation.is_dir():
                  dfs = []
                  for seed_ith in tqdm.tqdm(list(transformation.rglob('*scalars_*.csv'))):
                    dfs.append(pandas.read_csv(seed_ith))
                  df_a = pandas.concat(dfs, axis=0)
                  for tag in TrainingScalars:
                    if tag is not TrainingScalars.new_best_model:
                      pyplot.cla()
                      seaborn.lineplot(x="epoch", y=tag.value, data=df_a, seed=0)
                      pyplot.title(f'{timestamp.name, mapping.name, transformation.name} ' + r'$\bf{' + str(tag.value.replace('_', '\ ')) + r'}$ (95% confidence interval)') # TODO: INCLUDE SUPPORT?
                      pyplot.savefig(ensure_existence(agg_path / transformation.relative_to(timestamp)) / tag.value)

    def stesting_plot():
      agg_path = ensure_existence(base_res_path / 'agg')
      results_path = base_res_path / 'results' / 'testing'

      from matplotlib import pyplot  # ,rc
      # rc('text', usetex=True) # Use latex interpreter
      import seaborn
      seaborn.set()

      if True:
        showfliers = False

        for timestamp in tqdm.tqdm(list(results_path.iterdir())):
          if timestamp.is_dir():
            for mapping in tqdm.tqdm(list(timestamp.iterdir())):
              if mapping.is_dir():
                df_m = {}
                for transformation in tqdm.tqdm(list(mapping.iterdir())):
                  if transformation.is_dir():
                    dfs = []
                    for seed_ith in tqdm.tqdm(list(transformation.rglob('*scalars_*.csv'))):
                      dfs.append(pandas.read_csv(seed_ith))
                    # df_a = reduce(lambda df1,df2: pandas.merge(df1,df2,on='epoch'), dfs)
                    df_m[transformation.name] = pandas.concat(dfs, axis=0)

                result = pandas.concat(df_m)

                for tag in TestingScalars:
                  pyplot.cla()
                  ax = seaborn.boxplot(x=result.index, y=tag.value, data=result, showfliers=showfliers)
                  plot_median_labels(ax.axes, has_fliers=showfliers)
                  pyplot.title(f'{timestamp.name, mapping.name} ' + r'$\bf{' + str(tag.value.replace('_', '\ ')) + r'}$')
                  pyplot.savefig(ensure_existence(agg_path / mapping.name) / tag.value)

      if True:
        for timestamp in tqdm.tqdm(list(results_path.iterdir())):
          if timestamp.is_dir():
            for mapping in tqdm.tqdm(list(timestamp.iterdir())):
              if mapping.is_dir():
                df_m = {}
                for transformation in tqdm.tqdm(list(mapping.iterdir())):
                  if transformation.is_dir():
                    dfs = []
                    for seed_ith in tqdm.tqdm(list(transformation.rglob('*tensors_*.csv'))):
                      dddd = pandas.read_csv(seed_ith)
                      for c in dddd.columns:  # MAKE pandas interpret string representation of list as a list
                        dddd[c] = pandas.eval(dddd[c])
                      dfs.append(dddd)
                    df_m[transformation.name] = pandas.concat(dfs, axis=0)  # .reset_index()

                result = pandas.concat(df_m)

                serified_dict = {}
                for col_ in result.columns[1:]:
                  serified = result[col_].apply(pandas.Series).stack().reset_index()
                  serified = serified.drop(columns=['level_1'])
                  serified = serified.rename(columns={'level_0':'transformation',
                                                      'level_2':'threshold_idx',
                                                      0:        col_
                                                      })
                  serified = serified.set_index('threshold_idx')
                  serified_dict[col_] = serified

                merged_df = pandas.concat(serified_dict.values(), axis=1)
                merged_df = merged_df.T.drop_duplicates(keep='first').T  # drop duplicate columns but messes up datatype

                for c in merged_df.columns:  # fix datatypes
                  if c is not 'transformation':
                    merged_df[c] = pandas.to_numeric(merged_df[c])

                chop_off_percentage = 0.1
                # chop_off_size = int(len(merged_df)*chop_off_percentage)
                # merged_df=merged_df.loc[chop_off_size:-chop_off_size]

                #merged_df = merged_df[merged_df['test_precision_recall_recall'] > chop_off_percentage]
                #merged_df = merged_df[merged_df['test_precision_recall_recall'] < 1.0 - chop_off_percentage]

                pyplot.cla()
                ax = pyplot.gca()
                for t_name, transformation_df in tqdm.tqdm(merged_df.groupby('transformation')):
                  agg_group = transformation_df.groupby('threshold_idx')

                  mean_vals = agg_group.mean()
                  upper_vals = agg_group.quantile(0.975)
                  lower_vals = agg_group.quantile(0.025)
                  mean_area = auc(mean_vals["test_precision_recall_recall"], mean_vals["test_precision_recall_precision"])

                  m_l, = ax.plot(mean_vals['test_precision_recall_recall'],
                                 mean_vals['test_precision_recall_precision'],
                                 label=f'{t_name} (mean area: {mean_area})')
                  pyplot.xlabel('recall')
                  pyplot.ylabel('precision')
                  ax.fill_between(mean_vals['test_precision_recall_recall'],
                                  upper_vals['test_precision_recall_precision'],
                                  lower_vals['test_precision_recall_precision'],
                                  where=upper_vals['test_precision_recall_precision'] > lower_vals['test_precision_recall_precision'],
                                  facecolor=m_l.get_color(),
                                  alpha=0.3,
                                  interpolate=True)
                  if False:
                    for TP, coords in zip(mean_vals['test_precision_recall_true_positive_counts'],
                                          zip(mean_vals['test_precision_recall_recall'],
                                              mean_vals['test_precision_recall_precision'])):
                      ax.annotate(f'{TP}', xy=coords)

                pyplot.title(f'{timestamp.name, mapping.name} ' + r'$\bf{' + "Pr Curve" + r'}$ (95% confidence interval)')
                pyplot.legend()
                pyplot.savefig(ensure_existence(agg_path / mapping.name))

    training_plot()
    stesting_plot()


  main()
