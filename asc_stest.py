import os

import numpy
import torch
import tqdm
from draugr.torch_utilities import TensorBoardPytorchWriter, TorchEvalSession, auto_select_available_cuda_device, global_torch_device, to_device_iterator, to_tensor
from draugr.writers import TestingCurves, TestingScalars
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

import asc_utilities.block_data_utilities
from architectures.adversarial_signal_classifier import AdversarialSignalClassifier
from asc_transformations import TransformationEnum
from asc_utilities import misclassified_names
from experiment_config import DATA_ROOT_PATH, PROJECT_APP_PATH,CONFIG

if __name__ == '__main__':
  def main(verbose: bool = False):
    if torch.cuda.is_available():
      device = auto_select_available_cuda_device(2048)
    else:
      device = torch.device('cpu')

    global_torch_device(override=device)


    for transformation_name in tqdm.tqdm(TransformationEnum, desc='configs #'):
      for exp_name, exp_v in tqdm.tqdm(CONFIG.Experiments, desc=f'{transformation_name}'):
        map_path = (PROJECT_APP_PATH.user_data /
                    'results')

        latest_model = max(list(map_path.rglob('best_val_model_params.pt')), key=os.path.getctime)
        load_time = latest_model.relative_to(map_path).parts[0]
        print(f'Load time: {load_time} ')
        model_runs = list((map_path /
                           load_time /
                           exp_name /  # 'A/' or 'B/' or 'AB/'
                           f'{transformation_name.value}'  # 'mfcc/' or 'stft/' or 'igfcc/'
                           ).rglob('best_val_model_params.pt'))
        if not len(model_runs):
          print('no model runs found')
          exit(-1)
        for model_path in tqdm.tqdm(model_runs, desc=f'{exp_name} seed #'):
          if verbose:
            print(f'Loading model in {model_path}')
          with TensorBoardPytorchWriter(PROJECT_APP_PATH.user_log /
                                        load_time /
                                        exp_name /  # 'A/' or 'AB/'
                                        f'{transformation_name.value}' /
                                        model_path.parent.name,
                                        verbose=True) as writer:
            predictors = []
            categories = []
            test_names = []
            for k, t in exp_v.Test_Sets.items():
              (_, (pt, ct, nt), _) = asc_utilities.get_processed_block_splits( t.path / f'{transformation_name.value}_{k}{CONFIG.processed_file_ending}',
                                                                              train_percentage=t.train_percentage,
                                                                              test_percentage=t.test_percentage,
                                                                              validation_percentage=t.validation_percentage,
                                                                              random_seed=int(model_path.parent.name[-1]))

              predictors.append(pt)
              categories.append(ct)
              test_names.append(nt)

            predictors = numpy.concatenate(predictors)
            categories = numpy.concatenate(categories)
            test_names = numpy.concatenate(test_names)

            predictors = to_tensor(predictors[:, numpy.newaxis, ...])
            categories = to_tensor(categories[:, numpy.newaxis])
            # test_names = to_tensor(test_names[:, numpy.newaxis])

            test_loader = DataLoader(TensorDataset(predictors, categories),
                                     batch_size=CONFIG['Training_Methodology']['batch_size'],
                                     shuffle=False,
                                     num_workers=1,  # os.cpu_count(),
                                     pin_memory=True)

            model = AdversarialSignalClassifier(*predictors.shape[1:])
            model.load_state_dict(torch.load(str(model_path), map_location=device))
            model.to(device)

            predictions = []
            truth = []
            with TorchEvalSession(model):
              with torch.no_grad():
                for (ith_batch,
                     (predictors,
                      category)) in enumerate(to_device_iterator(test_loader, device=device)):
                  predictions += torch.sigmoid(model(predictors))
                  truth += category

            predictions = torch.stack(predictions)
            truth = torch.stack(truth)

            predictions_int = (predictions > 0.5).cpu().numpy().astype(numpy.int)
            truth = truth.cpu()
            truth_np = truth.numpy()
            predictions_np = predictions.cpu().numpy()
            accuracy_bw = accuracy_score(truth, predictions_int)

            writer.scalar(TestingScalars.test_accuracy.value, accuracy_bw)
            writer.scalar(TestingScalars.test_precision.value, precision_score(truth_np, predictions_int))
            writer.scalar(TestingScalars.test_recall.value, recall_score(truth_np, predictions_int))
            writer.scalar(TestingScalars.test_receiver_operator_characteristic_auc.value, roc_auc_score(truth_np, predictions_int))
            writer.precision_recall_curve(TestingCurves.test_precision_recall.value, predictions, truth)

            predictions_per_file, truth_per_file, names_per_file = asc_utilities.block_data_utilities.combine_blocks_to_file(predictions_int, truth_np, test_names)

            predictions_per_file_int = (predictions_per_file > 0.5).astype(numpy.int)

            confusion_bw = confusion_matrix(truth_np, predictions_int)
            wrongnames_bw = misclassified_names(predictions_int, truth_np, test_names)

            numpy.savez(model_path.parent / f'test_results',
                        y_bw=predictions_np,
                        d_bw=truth_np,
                        accuracy_bw=accuracy_bw,
                        confusion_bw=confusion_bw,
                        wrongnames_bw=wrongnames_bw,

                        y_pf=predictions_per_file,
                        d_pf=truth_per_file,
                        accuracy_pf=accuracy_score(truth_per_file, predictions_per_file_int),
                        confusion_pf=confusion_matrix(truth_per_file, predictions_per_file_int),
                        wrongnames_pf=misclassified_names(predictions_per_file_int, truth_per_file, names_per_file))


  main()
