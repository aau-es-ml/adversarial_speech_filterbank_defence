from random import seed, shuffle
from typing import Any, Sequence, Tuple

import numpy
from sklearn.model_selection import train_test_split

__all__ = ['get_processed_block_splits', 'combine_blocks_to_file']

import tqdm


def get_processed_block_splits(data_path,
                               train_percentage: float = 0.7,
                               test_percentage: float = 0.2,
                               validation_percentage: float = 0.1,
                               shuffle_data: bool = False,
                               random_seed: int = 42) -> Tuple[Tuple, Tuple, Tuple]:
  '''

  :param data_path:
  :param train_percentage:
  :param test_percentage:
  :param validation_percentage:
  :param shuffle_data:
  :param random_seed:
  :return:
  '''
  assert train_percentage >= 0 and test_percentage >= 0 and validation_percentage >= 0

  def get_block_indexes(file_indices: Sequence, selection: Sequence) -> Sequence:
    # Get the indices of all blocks that belong to each file in the set.
    block_indices = []
    for s in selection:
      file_start_idx = numpy.where(s == file_indices)[0][0]  # returns first occurrence idx
      block_indices.extend(range(file_indices[file_start_idx], file_indices[file_start_idx + 1]))
    return block_indices

  def fetch_data(path) -> Tuple[Any, Any, Any]:
    data = numpy.load(path)
    x = data['features']
    category = data['category']
    names = data['id']

    if shuffle_data:
      ind_list = [i for i in range(len(x))]
      seed(random_seed)
      shuffle(ind_list)
      return x[ind_list], category[ind_list], names[ind_list]
    else:
      return x, category, names

  predictors, response, file_names = fetch_data(data_path)

  if not numpy.isclose(train_percentage + test_percentage + validation_percentage, 1.0):
    raise Exception(f"The split percentages must add up to 1.0 but was {train_percentage + test_percentage + validation_percentage:f}")

  unique_file_indices = numpy.where('block0' == numpy.array([name.split('.')[1] for name in file_names]))[0]  # Return idx of occurence, Get indices of the first block of each file. # has same length as number of files.
  print(f'x: {len(predictors)}, cat: {len(response)}, names:{len(file_names)}, unique:{len(unique_file_indices)}')

  response_file_indices = response[unique_file_indices]  # Split unique array into train, test and val.

  if train_percentage == 1.0:
    train_file_indices = unique_file_indices
    test_file_indices = []
    validation_file_indices = []
  else:
    if 1.0 - train_percentage == 1.0:
      train_file_indices = []
      validation_file_indices, test_file_indices, _, _ = train_test_split(unique_file_indices,
                                                                          response_file_indices,
                                                                          test_size=test_percentage / (test_percentage + validation_percentage),
                                                                          random_state=random_seed,
                                                                          stratify=response_file_indices)  # Each file is still in their own set
    else:
      train_file_indices, test_file_indices, _, residual_response_unique = train_test_split(unique_file_indices,
                                                                                            response_file_indices,
                                                                                            test_size=1.0 - train_percentage,
                                                                                            random_state=random_seed,
                                                                                            stratify=response_file_indices)  # Each file is in their own set
      validation_file_indices, test_file_indices, _, _ = train_test_split(test_file_indices,
                                                                          residual_response_unique,
                                                                          test_size=test_percentage / (test_percentage + validation_percentage),
                                                                          random_state=random_seed,
                                                                          stratify=residual_response_unique)  # Each file is still in their own set

  exclusion_filter = [i for i, s in enumerate(file_names) if 'remove' in s]  # Remove the 'remove' labelled files from each subset.   # Specifically for speech-nonspeech test.
  file_indices = numpy.append(unique_file_indices, len(file_names))
  train_block_indices = get_block_indexes(file_indices, [i for i in train_file_indices if i not in exclusion_filter])
  test_block_indices = get_block_indexes(file_indices, [i for i in test_file_indices if i not in exclusion_filter])
  validation_block_indices = get_block_indexes(file_indices, [i for i in validation_file_indices if i not in exclusion_filter])

  return (
      (predictors[train_block_indices], response[train_block_indices], file_names[train_block_indices]),
      (predictors[test_block_indices], response[test_block_indices], file_names[test_block_indices]),
      (predictors[validation_block_indices], response[validation_block_indices], file_names[validation_block_indices])
      )


def combine_blocks_to_file(predictors: Sequence, response: Sequence, names: Sequence, integration: str = 'mean') -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
  sort_idxs = numpy.argsort(names)
  predictors = predictors[sort_idxs]
  response = response[sort_idxs]
  names = names[sort_idxs].tolist()
  names_uniq, unique = numpy.unique([name.split('.')[0] for name in names], return_index=True)

  unique = numpy.append(unique, len(names))

  predictors_per_file = []  # output 'y' per file
  response_per_file = []
  for i in tqdm.tqdm(range(len(unique) - 1)):
    if integration == 'mean':
      predictors_per_file.append(numpy.mean(predictors[unique[i]:unique[i + 1]]))
      response_per_file.append(response[unique[i]][0])
    elif integration == 'max':
      predictors_per_file.append(numpy.max(predictors[unique[i]:unique[i + 1]]))
      response_per_file.append(response[unique[i]][0])
    else:
      raise Exception("method must be set to 'mean' or 'max'")

  return numpy.array(predictors_per_file), numpy.array(response_per_file, dtype=numpy.int), numpy.array(names_uniq)
