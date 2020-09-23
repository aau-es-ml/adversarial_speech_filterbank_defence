#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Christian Heider Nielsen'
__doc__ = r'''

           Created on 22/06/2020
           '''

import math
import time

import numpy
import torch
import tqdm
from apppath import ensure_existence
from draugr.torch_utilities import (TensorBoardPytorchWriter,
                                    TorchEvalSession,
                                    TorchTrainSession,
                                    auto_select_available_cuda_device,
                                    global_torch_device,
                                    to_device_iterator,
                                    to_scalar,
                                    to_tensor,
                                    torch_clean_up,
                                    torch_seed,
                                    )
from draugr.writers.standard_tags import TrainingCurves, TrainingScalars
from torch.types import Device
from torch.utils.data import DataLoader, TensorDataset

from architectures.adversarial_signal_classifier import AdversarialSignalClassifier
from asc_transformations import TransformationEnum
from asc_utilities.block_data_utilities import get_processed_block_splits
from experiment_config import DATA_ROOT_PATH, PROJECT_APP_PATH, RESULTS_PATH


def train_asc(writer: TensorBoardPytorchWriter,
              *,
              train_loader,
              val_loader,
              cfg_name,
              out_path,
              input_size,
              validation_interval,
              learning_rate,
              adam_betas,
              num_epochs,
              device: Device = global_torch_device()) -> None:
  """

  :rtype: object
  """

  model = AdversarialSignalClassifier(*input_size).to(device)
  criterion = torch.nn.BCEWithLogitsLoss()
  optimiser = torch.optim.Adam(model.parameters(),
                               lr=learning_rate,
                               betas=adam_betas)
  best_val = math.inf
  for ith_epoch in tqdm.tqdm(range(num_epochs), desc=f'{cfg_name}'):
    accum_loss = 0.0
    with TorchTrainSession(model):
      for ith_batch, (predictors, category) in enumerate(to_device_iterator(train_loader,
                                                                            device=device), 1):
        optimiser.zero_grad()  # Initialize the gradients to zero
        loss = criterion(model(predictors), category)  # Forward
        loss.backward()  # Back propagation
        optimiser.step()  # Parameter update

        accum_loss += to_scalar(loss)

    writer.scalar(TrainingScalars.training_loss.value, accum_loss / ith_batch, ith_epoch)

    if ith_epoch % validation_interval == 0:
      loss_accum_val = 0.0
      accuracy_accum_val = 0.0
      with TorchEvalSession(model):
        with torch.no_grad():
          for ith_batch, (predictors, category) in enumerate(to_device_iterator(val_loader,
                                                                                device=device), 1):
            pred = model(predictors)
            loss_accum_val += to_scalar(criterion(pred, category))
            # accuracy_accum_val += accuracy_score(category.cpu().numpy(),(pred > 0.5).cpu().numpy().astype(numpy.int))
            accuracy_accum_val += to_scalar(((torch.sigmoid(pred) > 0.5) == category).float().sum() / float(pred.shape[0]))
            writer.precision_recall_curve(TrainingCurves.validation_precision_recall.value, pred, category, ith_epoch)

      val_loss = loss_accum_val / ith_batch
      writer.scalar(TrainingScalars.validation_loss.value, val_loss, ith_epoch)
      writer.scalar(TrainingScalars.validation_accuracy.value, accuracy_accum_val / ith_batch, ith_epoch)

      new_best = 0
      if val_loss < best_val:
        best_val = val_loss
        # print(f'new best {best_val}')
        torch.save(model.state_dict(), str(out_path / f'best_val_model_params.pt'))
        new_best = 1
      writer.scalar(TrainingScalars.new_best_model.value, new_best, ith_epoch)

  torch.save(model.state_dict(), str(out_path / f'final_model_params.pt'))

  torch_clean_up()


if __name__ == '__main__':

  def main():
    print(f'torch.cuda.is_available{torch.cuda.is_available()}')

    if torch.cuda.is_available():
      device = auto_select_available_cuda_device(2048)
    else:
      device = torch.device('cpu')

    global_torch_device(override=device)
    print(f'using device {global_torch_device()}')

    load_time = str(int(time.time()))

    from experiment_config import  CONFIG

    for transformation_name in tqdm.tqdm(TransformationEnum):
      for exp_name, exp_v in tqdm.tqdm(CONFIG.Experiments):
        for ith_run in tqdm.tqdm(range(CONFIG.Num_Runs), desc=f'{exp_name}, seed #'):
          results_folder = ensure_existence(RESULTS_PATH/
                                            load_time /
                                            exp_name /
                                            f'{transformation_name.value}' /  # 'mfcc/' or 'stft/' or 'igfcc/'
                                            f'seed_{ith_run}')  # 'run0/' or 'run1/' or 'run2/'

          torch_seed(ith_run)

          with TensorBoardPytorchWriter(PROJECT_APP_PATH.user_log /
                                        load_time /
                                        exp_name /
                                        f'{transformation_name.value}' /
                                        f'seed_{ith_run}',
                                        verbose=False) as writer:
            predictors_train = []
            categories_train = []
            train_names = []
            for k,t in exp_v.Train_Sets.items():
              ((pt, ct, nt), *_) = get_processed_block_splits(
                  t.path / f'{transformation_name.value}_{k}{CONFIG.processed_file_ending}',
                  train_percentage=t.train_percentage,
                  test_percentage=t.test_percentage,
                  validation_percentage=t.validation_percentage,
                  random_seed=ith_run)
              predictors_train.append(pt)
              categories_train.append(ct)
              train_names.append(nt)

            predictors_train = numpy.concatenate(predictors_train)
            categories_train = numpy.concatenate(categories_train)
            train_names = numpy.concatenate(train_names)

            predictors_val = []
            categories_val = []
            val_names = []
            for k,t in exp_v.Validation_Sets.items():
              (*_, (pv, cv, nv)) = get_processed_block_splits(
                  t.path / f'{transformation_name.value}_{k}{CONFIG.processed_file_ending}',
                  train_percentage=t.train_percentage,
                  test_percentage=t.test_percentage,
                  validation_percentage=t.validation_percentage,
                  random_seed=ith_run)
              predictors_val.append(pv)
              categories_val.append(cv)
              val_names.append(nv)

            predictors_val = numpy.concatenate(predictors_val)
            categories_val = numpy.concatenate(categories_val)
            val_names = numpy.concatenate(val_names)

            # assert categories_val.shape == categories_train.shape, f'Categories of train and val are not alike: {categories_val, categories_train}'
            assert min(categories_train) == min(categories_val), f'min not the same {min(categories_train), min(categories_val)}'
            assert max(categories_train) == max(categories_val), f'max not the same {max(categories_train), max(categories_val)}'

            predictors_train = to_tensor(predictors_train[:, numpy.newaxis, ...], dtype=torch.float)  # Add channel for convolutions
            categories_train = to_tensor(categories_train[:, numpy.newaxis], dtype=torch.float)  # Add channel for loss functions

            predictors_val = to_tensor(predictors_val[:, numpy.newaxis, ...], dtype=torch.float)  # Add channel for convolutions
            categories_val = to_tensor(categories_val[:, numpy.newaxis], dtype=torch.float)  # Add channel for loss functions

            batch_size = CONFIG['Training_Methodology']['batch_size']
            train_loader = DataLoader(TensorDataset(predictors_train, categories_train),
                                      batch_size=batch_size,
                                      shuffle=True,
                                      num_workers=1,  # os.cpu_count(),
                                      pin_memory=True)
            val_loader = DataLoader(TensorDataset(predictors_val, categories_val),
                                    batch_size=batch_size,
                                    shuffle=False,
                                    num_workers=1,  # os.cpu_count(),
                                    pin_memory=True)

            print(f'predictors_train.shape:{predictors_train.shape}')
            assert len(predictors_train.shape) == 4
            *_, x_height, x_width = predictors_train.shape
            print(f'x_height, x_width: {x_height, x_width}')

            train_asc(writer,
                      train_loader=train_loader,
                      val_loader=val_loader,
                      out_path=results_folder,
                      cfg_name=transformation_name.value,
                      input_size=(1, x_height, x_width),
                      validation_interval=CONFIG['Validation']['val_interval'],
                      learning_rate=CONFIG['Training_Methodology']['learning_rate'],
                      adam_betas=(CONFIG['Training_Methodology']['beta1'],
                                  CONFIG['Training_Methodology']['beta2']),
                      num_epochs=CONFIG['Training_Methodology']['epochs'])


  main()
