import os
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

from common.logger import Logger

logger = Logger(__name__)



class ModelBuilder:
    def __init__(self, dataset, hp, verbose=0):
        # создаём дополнительные признаки - временные лаги - на сколько дней мы "смотрим назад"
        lags_sequence = [
            1, 2, 3, 4, 5, 6, 7, 8,     # недельные признаки
            13, 14, 15,                 # две недели назад
            20, 21, 22,                 # три недели назад
            27, 28, 29, 30, 31,         # четыре недели и месяц назад
            58, 59, 60, 61,             # два месяца назад
            89, 90, 91, 92,             # квартал назад
            363, 364, 365, 366          # год назад
            ]

        # формируем дополнительные временные признаки: день недели, день месяца, день года
        # при определении трансформации к ним будет добавлен ещё один - "возраст" последовательности
        time_features = time_features_from_frequency_str(hp.get('freq'))

        if verbose:
            print(f'Временные лаги: ({type(lags_sequence)}): {lags_sequence}')
            print(f'Временные признаки: ({type(time_features)}): {time_features}')

        prediction_len = hp.get('prediction_len')

        self.config = TimeSeriesTransformerConfig(
            # длина предсказываемой последовательности
            prediction_length=prediction_len,
            # длина контекста:
            context_length=int(prediction_len * hp.get('context_ratio')),
            # временные лаги
            lags_sequence=lags_sequence,
            # добавляем три временных признака + при трансформации будет добавлен
            # ещё "возраст" временного шага
            num_time_features=len(time_features) + 1,
            # единственный статический категориальный признак - ID серии:
            num_static_categorical_features=1,
            # 4 возможных значения:
            cardinality=[len(dataset)],
            # модель будет обучать эмбеддинги размерностью 2 для каждого из 4-х возможных значений
            embedding_dimension=[hp.get('embedding_dim')],

            # transformer params:
            encoder_layers=hp.get('encoder_layers'),
            decoder_layers=hp.get('decoder_layers'),
            d_model=hp.get('d_model'),
        )

    def get(self):
        return TimeSeriesTransformerForPrediction(self.config), self.config


if __name__ == '__main__':
    os.chdir('..')
    logger.setup(level=logger.DEBUG, layout='debug')
