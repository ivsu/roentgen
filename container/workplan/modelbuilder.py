import os
# модель трансформера для временных рядов и её конфигуратор
from transformers import TimeSeriesTransformerConfig, TimeSeriesTransformerForPrediction
# для трансформации и обогащения данных
from gluonts.time_feature import (
    # TimeFeature,
    # функции фреймворка GluonTS для формирования дополнительных признаков временного ряда
    time_features_from_frequency_str,
    # get_lags_for_frequency,
    # week_of_year
)

from common.logger import Logger

logger = Logger(__name__)


class ModelBuilder:
    def __init__(self, bot, n_channels, n_time_features):

        # дополнительные признаки - временные лаги - на сколько дней мы "смотрим назад"
        lags_sequence = bot.get_lags_sequence()

        logger.debug(f'Временные лаги: ({type(lags_sequence)}): {lags_sequence}')

        prediction_len = bot.get('prediction_len')

        self.config = TimeSeriesTransformerConfig(
            # длина предсказываемой последовательности
            prediction_length=prediction_len,
            # длина контекста:
            context_length=int(prediction_len * bot.get('context_ratio')),
            # временные лаги
            lags_sequence=lags_sequence,
            # количество временных признака + 1 (возраст временного шага) будет добавлен при трансформации
            num_time_features=n_time_features + 1,
            # единственный статический категориальный признак - ID серии:
            num_static_categorical_features=1,
            # количество каналов
            cardinality=[n_channels],
            # размерность эмбеддингов
            embedding_dimension=[bot.get('embedding_dim')],
            # параметры трансформера
            encoder_layers=bot.get('encoder_layers'),
            decoder_layers=bot.get('decoder_layers'),
            d_model=bot.get('d_model'),
        )

    def get(self):
        return TimeSeriesTransformerForPrediction(self.config), self.config

    def get_time_features(self):
        return self.time_features


if __name__ == '__main__':
    os.chdir('..')
    logger.setup(level=logger.DEBUG, layout='debug')

    from workplan.hyperparameters import Hyperparameters
    from workplan.genetic import Bot

    hp = Hyperparameters()
    bot = Bot(hp)
    values, bot_hash = hp.generate(mode='default', hashes=[])
    bot.activate(values, bot_hash)

    logger.info(f'prediction_len: {bot.get("prediction_len")}')
    logger.info(f'freq: {bot.get("freq")}')
    logger.info(f'context_ratio: {bot.get("context_ratio")}')

    ds = [tmp for tmp in range(10)]
    mb = ModelBuilder(bot, n_channels=len(ds), n_time_features=1)

    logger.debug(f'model config:\n{mb.config}')

