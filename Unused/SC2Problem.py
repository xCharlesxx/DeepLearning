from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import DeepNetwork as dn
import tensorflow.contrib.layers as layers
import tensor2tensor as t2t
from tensor2tensor import problems
from tensor2tensor.data_generators.gym_env import T2TGymEnv
from tensor2tensor.data_generators.translate import TranslateDistillProblem
from tensor2tensor.models import transformer 
from tensor2tensor.data_generators import problem
from tensor2tensor.utils import hparams_lib
from tensor2tensor.utils.hparams_lib import create_hparams
from tensor2tensor.utils.trainer_lib import create_run_config, create_experiment
from tensor2tensor.utils import registry
import keras as ks
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.callbacks import TensorBoard


import numpy as np 
import os 
import csv
import ast
import re
import random
from decimal import Decimal
from Constants import const

class SC2Problem(TranslateDistillProblem):


  @property
  def approx_vocab_size(self):
    return 2**13  # ~8k

  @property
  def is_generate_per_split(self):
    # generate_data will shard the data into TRAIN and EVAL for us.
    return False

  @property
  def dataset_splits(self):
    """Splits of data to produce and number of output shards for each."""
    # 10% evaluation data
    return [{
        "split": problem.DatasetSplit.TRAIN,
        "shards": 9,
    }, {
        "split": problem.DatasetSplit.EVAL,
        "shards": 1,
    }]

  def generate_samples(self, data_dir, tmp_dir, dataset_split):
    del data_dir
    del tmp_dir
    del dataset_split

    data = dn.get_training_data("training_data")

    books = [
        # bookid, skip N lines
        (19221, 223),
        (15553, 522),
    ]

    for (book_id, toskip) in books:
      text = cleanup.strip_headers(acquire.load_etext(book_id)).strip()
      lines = text.split("\n")[toskip:]
      prev_line = None
      ex_count = 0
      for line in lines:
        # Any line that is all upper case is a title or author name
        if not line or line.upper() == line:
          prev_line = None
          continue

        line = re.sub("[^a-z]+", " ", line.strip().lower())
        if prev_line and line:
          yield {
              "inputs": prev_line,
              "targets": line,
          }
          ex_count += 1
        prev_line = line