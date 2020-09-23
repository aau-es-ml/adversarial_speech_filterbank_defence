import os
from pathlib import Path

import tqdm
from apppath import AppPath, ensure_existence
from draugr.tensorboard_utilities import TensorboardEventExporter
from draugr.writers.standard_tags import TestingCurves, TestingScalars, TrainingCurves, TrainingScalars

if __name__ == '__main__':
  def extract_scalars_as_csv(train_path: Path = Path('exclude') /
                                                'results' /
                                                "training",
                             test_path: Path = Path('exclude') /
                                               'results' /
                                               "testing", verbose=False):

    max_load_time = max(list(AppPath('Adversarial Speech', 'Christian Heider Nielsen').user_log.iterdir()), key=os.path.getctime)
    event_files = list(max_load_time.rglob('events.out.tfevents.*'))
    # _path_to_events_file = max(event_files, key=os.path.getctime)

    for e in tqdm.tqdm(set([ef.parent for ef in event_files])):
      relative_path = e.relative_to(max_load_time)
      with TensorboardEventExporter(e, save_to_disk=True) as tee:
        if True:
          out_tags = []
          for tag in tqdm.tqdm(TestingScalars):
            if tag.value in tee.available_scalars:
              out_tags.append(tag.value)

          if len(out_tags):
            tee.scalar_export_csv(*out_tags,
                                  out_dir=ensure_existence(test_path /
                                                                      max_load_time.name /
                                                                      relative_path,
                                                                      force_overwrite=True,
                                                                      verbose=verbose))
          else:
            print('no tags found')

        if True:
          out_tags = []
          for tag in tqdm.tqdm(TrainingScalars):
            if tag.value in tee.available_scalars:
              out_tags.append(tag.value)

          if len(out_tags):
            tee.scalar_export_csv(*out_tags,
                                  out_dir=ensure_existence(train_path /
                                                                      max_load_time.name /
                                                                      relative_path,
                                                                      force_overwrite=True,
                                                                      verbose=verbose))
          else:
            print('no tags found')


  def extract_tensors_as_csv(train_path=Path('exclude') /
                                        'results' /
                                        "training",
                             test_path=Path('exclude') /
                                       'results' /
                                       "testing", verbose=False):

    load_times = list(AppPath('Adversarial Speech', 'Christian Heider Nielsen').user_log.iterdir())
    max_load_time = max(load_times, key=os.path.getctime)
    event_files = list(max_load_time.rglob('events.out.tfevents.*'))
    # _path_to_events_file = max(event_files, key=os.path.getctime)

    unique_event_files_parents = set([ef.parent for ef in event_files])

    for e in tqdm.tqdm(unique_event_files_parents):
      relative_path = e.relative_to(max_load_time)
      with TensorboardEventExporter(e, save_to_disk=True) as tee:
        if True:
          out_tags = []
          for tag in tqdm.tqdm(TestingCurves):
            if tag.value in tee.available_tensors:
              out_tags.append(tag.value)

          if len(out_tags):
            tee.pr_curve_export_csv(*out_tags,
                                    out_dir=ensure_existence(test_path /
                                                             max_load_time.name /
                                                             relative_path,
                                                             force_overwrite=True,
                                                             verbose=verbose))
          else:
            print('no tags found')

        if False:  # TODO: OUTPUT for all epoch steps, no support yet
          out_tags = []
          for tag in tqdm.tqdm(TrainingCurves):
            if tag.value in tee.available_tensors:
              out_tags.append(tag.value)

          if len(out_tags):
            tee.pr_curve_export_csv(*out_tags,
                                    out_dir=ensure_existence(train_path /
                                                             max_load_time.name /
                                                             relative_path,
                                                             force_overwrite=True,
                                                             verbose=verbose))
          else:
            print('no tags found')


  extract_scalars_as_csv()
  extract_tensors_as_csv()
