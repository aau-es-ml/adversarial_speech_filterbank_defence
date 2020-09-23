import glob
import os
from pathlib import Path
from typing import Any

import numpy
from matplotlib import patheffects, pyplot
from pandas.core.dtypes.missing import array_equivalent
from scipy.stats import sem, t

__all__ = ['confidence_interval',
           'receiver_operator_characteristic',
           'misclassified_names']


def confidence_interval(x: Any, p: float = 0.95, axis: int = 0) -> Any:
  assert len(x) > 1
  return sem(x, axis=axis) * t.ppf((1 + p) / 2, x.shape[axis] - 1)


def confidence_interval_py38(data, confidence=0.95):
  from statistics import NormalDist
  dist = NormalDist.from_samples(data)
  z = NormalDist().inv_cdf((1 + confidence) / 2.)
  h = dist.stdev * z / ((len(data) - 1) ** .5)
  return dist.mean - h, dist.mean + h


def receiver_operator_characteristic(confusion):
  fp_rate = confusion[:, 0, 1] / (confusion[:, 0, 1] + confusion[:, 1, 1])
  fp_rate = numpy.append(fp_rate, fp_rate[-1])
  fn_rate = confusion[:, 1, 0] / (confusion[:, 1, 0] + confusion[:, 0, 0])
  fn_rate = numpy.append(fn_rate, 1)

  accuracy = numpy.max((confusion[:, 0, 0] + confusion[:, 1, 1]) / confusion[0, ...].sum())  # optimal p
  return fp_rate, fn_rate, accuracy


def misclassified_names(y, d, names, p=0.5):
  return names[(1 * (y > p) != d).reshape(-1)]  # Array shape (x,) instead of (x,1)


def auc_ci (upper_precision, precision, lower_precision, upper_recall, recall, lower_recall)->float:
  '''
    # assuming data is a list of (upper_precision, precision, lower precision, upper_recall, recall, lower_recall)

  https://stackoverflow.com/a/57483123
  :return:
  '''

  data =  (upper_precision, precision, lower_precision, upper_recall, recall, lower_recall)
  auc = 0
  numpy.sort(data, key=lambda x: x[1])  # sort by precision
  last_point = (0, 0)  # last values of x,y
  for up, p, lp, ur, r, lr in data:
    # whatever was used to sort should come first
    new_point = (p, ur)  # or (r, up) for upper bound; (p, lr), (r, lp) for lower bound
    dx = new_point[0] - last_point[0]
    y = last_point[1]
    auc += dx * y + dx * (new_point[1] - y) * 0.5
  return auc


def agg_validation_acc(outfolder, val_interval, T, dataname, trainset):
  """

  :param config:
  :type config:
  :param args:
  :type args:
  """

  val_idxs = numpy.arange(0, T, val_interval) + val_interval

  feature_folders = glob.glob(outfolder.replace(dataname, '*[0-9]'))
  feature_folders = set([folder[:-1] for folder in feature_folders])
  pyplot.figure('Validation_acc', dpi=150)
  for feat_folder in sorted(list(feature_folders)):
    feature = os.path.normpath(feat_folder).split(os.sep)[-1]
    foldernames = glob.glob(feat_folder + '*')
    accuracy = numpy.zeros((len(foldernames), len(val_idxs)))
    for i, folder in enumerate(foldernames):
      for j, val in enumerate(val_idxs):
        data = numpy.load(os.path.join(folder,
                                       f'Validation_{feature}_{val}.npz'))
        confusion = data['confusion']
        _, _, accuracy[i, j] = receiver_operator_characteristic(confusion)

    meanAcc = numpy.mean(accuracy, axis=0)
    acc95 = confidence_interval(accuracy, p=0.95, axis=0)

    pyplot.plot(val_idxs, meanAcc, label=feature.upper())
    pyplot.fill_between(val_idxs, meanAcc - acc95, meanAcc + acc95, alpha=0.5)
  pyplot.xlim([val_idxs[0], val_idxs[-1]])
  pyplot.ylim([0.85, 1])
  pyplot.xlabel('Epoch')
  pyplot.ylabel('Accuracy')
  pyplot.legend(loc='lower right')
  pyplot.tight_layout()

  pyplot.savefig(outfolder / f'val_acc_{trainset}.pdf')


def agg_train_loss(outfolder: Path, dataname):
  filenames = list(Path(outfolder).rglob('loss_final.npy'))
  # Compute losses, the mean loss and the 95% confidence interval
  losses = numpy.zeros((len(filenames), len(numpy.load(filenames[0]))))

  for i, file in enumerate(filenames):
    losses[i, :] = numpy.load(file)

  mean_loss = numpy.mean(losses, axis=0)
  loss95 = evaluation.confidence_interval(losses, p=0.95, axis=0)

  T = len(losses)
  # Plot
  epochs = numpy.arange(0, 1, 1 / len(mean_loss)) * T
  pyplot.figure(f'Train_loss {dataname}', dpi=150)
  pyplot.plot(epochs[:len(mean_loss)], mean_loss, color='k')
  xticks = (numpy.arange(0, 6) / 5 * T).astype(numpy.int32)
  pyplot.xticks(xticks)
  pyplot.xlim([0, T])
  pyplot.ylim([0, numpy.max(mean_loss + loss95)])
  pyplot.xlabel('Epoch')
  pyplot.ylabel('Loss')
  pyplot.tight_layout()

  pyplot.savefig(outfolder / 'train_loss.pdf')


def wrong_names(outfolder, dataname, testset):
  with open(outfolder / 'wrong_name.txt')as f:
    foldernames = set([folder[:-1] for folder in glob.glob(outfolder.replace(dataname, '') + 'stft[0-9]')])  # Remove end number and duplicate entries
    for folder in sorted(list(foldernames)):
      feature = os.path.normpath(folder).split(os.sep)[-1]
      print(f'\nNames of wrong files for {feature.upper()}\n')
      filenames = glob.glob(folder + f'*[0-9]/Test*_{testset}.npz')
      w = []
      for file in filenames:
        data = numpy.load(file)
        wrong = data['wrongnames']
        w.extend(wrong)
      w = [s.split('/')[-1] for s in w]
      w = sorted(list(set(w)))
      print(*w, sep='\n')
      f.writelines(w)
      # f.write('\n'.join(w))


def ftest_roc(outfolder, dataname, trainset, testset, per_file: bool):
  print(f'Test set {testset}')
  foldernames = glob.glob(outfolder.replace(dataname, '') + '*[0-9]')
  foldernames = set([folder[:-1] for folder in foldernames])  # Remove end number and duplicate entries
  zoommax = 0
  for folder in sorted(list(foldernames)):
    feature = os.path.normpath(folder).split(os.sep)[-1]
    filenames = glob.glob(folder + f'*/Test*_{testset}.npz')
    fig1 = pyplot.figure(f'Test_ROC {testset} normal', dpi=150)
    ax1 = pyplot.gca()
    fig2 = pyplot.figure(f'Test_ROC {testset} zoom', dpi=150)
    ax2 = pyplot.gca()
    data = numpy.load(filenames[0])
    tmp, _, _ = receiver_operator_characteristic(data[f'confusion{per_file}'])
    FPrate = numpy.zeros((len(filenames), tmp.shape[0]))
    FNrate = numpy.zeros((len(filenames), tmp.shape[0]))
    accuracy = numpy.zeros((len(filenames)))
    for i, file in enumerate(filenames):
      data = numpy.load(file)
      confusion = data[f'confusion{"pf" if per_file else "pb"}']
      accuracy[i] = data[f'accuracy{"pf" if per_file else "pb"}']
      FPrate[i, :], FNrate[i, :], _ = receiver_operator_characteristic(confusion)

    mean_fp = FPrate.mean(axis=0)
    mean_fn = FNrate.mean(axis=0)
    meanAcc = accuracy.mean()
    acc95 = confidence_interval(accuracy, p=0.95)

    # For determining zoom amount
    max_fp = mean_fp[mean_fn > 1e-3][0]
    max_fn = mean_fn[mean_fp > 1e-3][-1]
    if (max_fp > zoommax) or (max_fn > zoommax):
      zoommax = max(max_fp, max_fn)

    print(f'{feature.upper()} Accuracy: {round(meanAcc * 100, 2):.2f} $\\pm$ {round(acc95 * 100, 2):.2f}')
    ax1.plot(mean_fp, mean_fn, label=feature.upper())
    ax2.plot(mean_fp, mean_fn, label=feature.upper())
  if zoommax == 0:
    zoommax = 1e-6
  ax1.set_ylim([0, 1])
  ax1.set_xlim([0, 1])
  ax1.set_xlabel('False positive rate')
  ax1.set_ylabel('False negative rate')
  ax1.legend(loc='upper right')
  fig1.tight_layout()
  ax2.set_ylim([0, zoommax])
  ax2.set_xlim([0, zoommax])
  ax2.set_xlabel('False positive rate')
  ax2.set_ylabel('False negative rate')
  ax2.legend(loc='upper right')
  fig2.tight_layout()

  fig1.savefig(outfolder / f'Test_{trainset}_ROC_{testset}{per_file}.pdf')
  fig2.savefig(outfolder / f'Test_{trainset}_ROC_zoom_{testset}{per_file}.pdf')


def plot_median_labels(ax, has_fliers: bool = False, text_size: int = 10, text_weight: str = 'normal', stroke_width: int = 2, precision: int = 3):
  lines = ax.get_lines()
  # depending on fliers, toggle between 5 and 6 lines per box
  lines_per_box = 5 + int(has_fliers)
  # iterate directly over all median lines, with an interval of lines_per_box
  # this enables labeling of grouped data without relying on tick positions
  for median_line in lines[4:len(lines):lines_per_box]:
    # get center of median line
    mean_x = sum(median_line._x) / len(median_line._x)
    mean_y = sum(median_line._y) / len(median_line._y)

    text = ax.text(mean_x,
                   mean_y,
                   f'{round(mean_y, precision)}',
                   ha='center',
                   va='center',
                   fontweight=text_weight,
                   size=text_size,
                   color='white')  # print text to center coordinates

    # create small black border around white text
    # for better readability on multi-colored boxes
    text.set_path_effects([
        patheffects.Stroke(linewidth=stroke_width, foreground='black'),
        patheffects.Normal(),
        ])


def show_values_on_bars(axs, h_v="v", space=0.4):
  def _show_on_single_plot(ax):
    if h_v == "v":
      for p in ax.patches:
        _x = p.get_x() + p.get_width() / 2
        _y = p.get_y() + p.get_height()
        value = int(p.get_height())
        ax.text(_x, _y, value, ha="center")
    elif h_v == "h":
      for p in ax.patches:
        _x = p.get_x() + p.get_width() + float(space)
        _y = p.get_y() + p.get_height()
        value = int(p.get_width())
        ax.text(_x, _y, value, ha="left")

  if isinstance(axs, numpy.ndarray):
    for idx, ax in numpy.ndenumerate(axs):
      _show_on_single_plot(ax)
  else:
    _show_on_single_plot(axs)


def duplicate_columns(frame):
  groups = frame.columns.to_series().groupby(frame.dtypes).groups
  dups = []

  for t, v in groups.items():

    cs = frame[v].columns
    vs = frame[v]
    lcs = len(cs)

    for i in range(lcs):
      ia = vs.iloc[:, i].values
      for j in range(i + 1, lcs):
        ja = vs.iloc[:, j].values
        if array_equivalent(ia, ja):
          dups.append(cs[i])
          break

  return dups