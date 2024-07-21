import os
import pandas as pd
import numpy as np
import os
import random
import gc
import glob
import json
import hashlib
import time
# from google.colab import drive

# from datasets import Dataset, load_dataset, Features, Sequence, Value
from datasets import Dataset
# методы для конвертации даты в pd.Period на лету
from functools import lru_cache, partial
# модель трансформера для временных рядов и её конфигуратор
from transformers import TimeSeriesTransformerConfig, TimeSeriesTransformerForPrediction

# для трансформации и обогащения данных
from gluonts.time_feature import (
    # TimeFeature,
    # функции фреймворка GluonTS для формирования дополнительных признаков временного ряда
    time_features_from_frequency_str,
    # get_lags_for_frequency,
)
from gluonts.dataset.field_names import FieldName
from gluonts.transform import (
    AddAgeFeature,
    AddObservedValuesIndicator,
    AddTimeFeatures,
    AsNumpyArray,
    Chain,
    ExpectedNumInstanceSampler,
    InstanceSplitter,
    RemoveFields,
    SelectFields,
    SetField,
    TestSplitSampler,
    Transformation,
    ValidationSplitSampler,
    VstackFeatures,
    RenameFields,
)

# базовый класс для определения трансформации
from transformers import PretrainedConfig

# для нарезки временного ряда на окна
from gluonts.transform.sampler import InstanceSampler
from typing import Optional

# для загрузчиков данных
from typing import Iterable
from gluonts.itertools import Cached, Cyclic
from gluonts.dataset.loader import as_stacked_batches

# для расчёта метрик MASE/sMAPE
from evaluate import load
from gluonts.time_feature import get_seasonality

# для обучения модели
from accelerate import Accelerator
from torch.optim import AdamW
from torch.optim.lr_scheduler import LRScheduler
import torch

# вывод графиков
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from common.logger import Logger
from show import Show
from hyperparameters import Hyperparameters
from datamanager import DataManager, CHANNEL_NAMES
from genetic import Researcher

logger = Logger(__name__)


if __name__ == '__main__':
    os.chdir('..')
    logger.setup(level=logger.INFO, layout='debug')

    # создаём инстанс гиперпараметров
    hp = Hyperparameters()

    # (!) plotly странно работает при первом вызове в цикле - выведем графические
    # индикаторы и удалим инстанс
    Show.indicators(
        hp,
        params=['n_epochs', 'warmup_epochs', 'prediction_len'],
        titles=['Эпох обучения', 'Эпох прогрева', 'Глубина предикта']
    )

    # создаём менеджер данных и готовим исходный DataFrame для формирования выборок
    dm = DataManager()
    dm.prepare(
        freq=hp.get('freq'),
        prediction_len=hp.get('prediction_len')
    )
    # формируем выборки
    train_ds = dm.from_generator(splits=2, split='train')
    test_ds = dm.from_generator(splits=2, split='test')

    # создаём Researcher и передаём ему датасеты и инстанс гиперпараметров
    researcher = Researcher(train_ds, test_ds, hp,
                            CHANNEL_NAMES,
                            mode='genetic',
                            show_graphs=True,
                            train=True, save_bots=True, verbose=1)
    researcher.run()
