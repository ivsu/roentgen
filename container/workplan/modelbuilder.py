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

from common.logger import Logger

logger = Logger(__name__)


class ModelBuilder:
    def __init__(self, dataset, hp, verbose=0):
        # создаём дополнительные признаки - временные лаги - на сколько дней мы "смотрим назад"
        lags_sequence = [
            1, 2, 3, 4, 5,      # предшествующие недели
            51, 52, 53, 54,  # год назад
            103, 104, 105, 106, 107  # 2 года назад
            ]

        # формируем дополнительные временные признаки: день недели, день месяца, день года
        # при определении трансформации к ним будет добавлен ещё один - "возраст" последовательности
        time_features = time_features_from_frequency_str(hp.get('freq'))
        # print(f'time_features:\n{time_features}')

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
            # количество временных признака + 1 (возраст временного шага) будет добавлен при трансформации
            num_time_features=len(time_features) + 1,
            # единственный статический категориальный признак - ID серии:
            num_static_categorical_features=1,
            # количество каналов
            cardinality=[len(dataset)],
            # размерность эмбеддингов
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

    from workplan.hyperparameters import Hyperparameters
    from workplan.genetic import Bot

    hp = Hyperparameters()
    bot = Bot(hp)
    values, bot_hash = hp.generate(mode='default', hashes=[])
    bot.activate(values, bot_hash)

    print(f'prediction_len: {bot.get("prediction_len")}')
    print(f'freq: {bot.get("freq")}')
    print(f'context_ratio: {bot.get("context_ratio")}')

    dataset = [tmp for tmp in range(10)]
    model = ModelBuilder(dataset, bot)

    print(f'model config:\n{model.config}')
