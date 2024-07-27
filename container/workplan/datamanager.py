import os
import gc
import pandas as pd
from datetime import datetime, timedelta
from datasets import Dataset
from functools import lru_cache
from common.logger import Logger
from hyperparameters import Hyperparameters
from settings import DB_VERSION
if DB_VERSION == 'PG':
    from common.db import DB, get_all
else:
    from common.dblite import DB, get_all

logger = Logger(__name__)

# текстовые метки каналов данных
CHANNEL_NAMES = ['kt', 'kt_ce1', 'kt_ce2', 'mrt', 'mrt_ce1', 'mrt_ce2', 'rg', 'flg', 'mmg', 'dens']
# словарь с именами и индексами каналов данных
CHANNEL_INDEX = {name: ch for ch, name in enumerate(CHANNEL_NAMES)}


def generator(df, splits, split, prediction_len):
    """
    Генератор выборок.

    :param df: датафрейм для формирования выборки
    :param splits: количество выборок, на которое делится датасет (2, 3)
    :param split: имя сплита (train, validation, test)
    :param prediction_len: глубина предсказания
    """
    assert splits in [2, 3]
    assert split in ['train', 'validation', 'test']

    # определим конечный индекс данных, в зависимости от выборки
    # сплиты будут разной длины на prediction_len, но начало у них всех одинаковое
    if splits == 3 and split == 'train':
        end_index = len(df) - prediction_len * 2
    elif (splits == 3 and split == 'validation' or
          splits == 2 and split == 'train'):
        end_index = len(df) - prediction_len
    elif split == 'test':
        end_index = len(df)
    else:
        raise Exception(f'Неверное сочетание имени выборки [{split}] и количества выборок [{splits}].')

    # дата начала временной последовательности - одинаковая для всех данных
    # переводим в timestamp, т.к. Arrow не понимает Period, -
    # будем конвертировать в Period на этапе трансформации
    start_date = df.index.min().to_timestamp()

    for channel, index in CHANNEL_INDEX.items():
        yield {
            'start': start_date,
            'target': df[channel].iloc[:end_index].to_list(),
            # статический признак последовательности
            'feat_static_cat': [index]
        }


class DataManager:

    def __init__(self):
        """
        Класс для чтения данных с диска и формирования выборок.
        Также позволяет сгруппировать данные до заданной частоты.
        Хранит полученный датафрейм для генерации датасетов.
        """
        # датафрейм - источник данных для выборок датасетов
        self.df = None
        # глубина предсказания - длина предсказываемой последовательности
        self.prediction_len = None
        # частота данных
        self.freq = None
        # префикс схемы БД
        self.db_schema_prefix = 'roentgen.' if DB_VERSION == 'PG' else ''

    def read(self):
        db = DB()
        query = f"""
            select
                year, week, modality, contrast_enhancement as ce, amount
            from {self.db_schema_prefix}work_summary
            order by year, week
            ;
        """
        with db.get_cursor() as cursor:
            cursor.execute(query)
            df: pd.DataFrame = get_all(cursor)
        db.close()

        def get_first_week_sunday(year, week):
            first_date = datetime(year, 1, 1)
            return first_date + timedelta(weeks=week - 1, days=6 - first_date.weekday())

        def week_to_date_time(row):
            return get_first_week_sunday(row['year'], row['week'])

        def compose_channel(row):
            return row['modality'] \
                if row['ce'] == 'none' \
                else row['modality'] + '_' + row['ce']

        df['datetime'] = df.apply(week_to_date_time, axis=1)
        df['channel'] = df.apply(compose_channel, axis=1)
        # print(df[['datetime', 'channel', 'amount']].iloc[-20:])
        # print(df.columns)

        df = df.pivot(index=['datetime'], columns=['channel'], values=['amount'])
        # print(df.columns)
        df.columns = df.columns.levels[1]
        self.df = df

        logger.info('Данные загружены.')
        # logger.info(f'self.df:\n{self.df.head()}')

    def prepare(self, freq, prediction_len):
        """ Подготовка источника данных - DataFrame """
        self.freq = freq
        self.prediction_len = prediction_len
        # загружаем данные в датафрейм
        self.read()
        # переводим индекс в pd.Period
        self.df.index = self.df.index.to_period(freq)
        # print(f'self.df:\n{self.df}')

    def from_generator(self, splits, split):
        """ Возвращает датасет в зависимости от аргумента split"""
        # ds = Dataset.from_generator(self._gen, gen_kwargs={
        ds = Dataset.from_generator(generator, gen_kwargs=dict(
            df=self.df, splits=splits, split=split, prediction_len=self.prediction_len
        ))
        # используем функциональность датасета set_transform для конвертации
        # признака start в pd.Period на лету
        ds.set_transform(self.transform_start_field)

        return ds

    @lru_cache(10_000)
    def convert_to_pandas_period(self, date):
        """
        Конвертирует дату в pd.Period соответствующей частоты.
        Данные преобразования кэшируются.
        """
        return pd.Period(date, self.freq)

    def transform_start_field(self, batch):
        """
        Конвертирует признак start в pd.Period соответствующей частоты по батчу.
        Используется для конвертации на лету.
        """
        # данные преобразования кэшируются
        batch["start"] = [self.convert_to_pandas_period(date) for date in batch["start"]]
        return batch

    # def _gen(self, splits, split):
    #     """
    #     Генератор выборок.
    #
    #     :param splits: количество выборок, на которое делится датасет (2, 3)
    #     :param split: имя сплита (train, validation, test)
    #     """
    #     assert splits in [2, 3]
    #     assert split in ['train', 'validation', 'test']
    #
    #     # определим конечный индекс данных, в зависимости от выборки
    #     # сплиты будут разной длины на prediction_len, но начало у них всех одинаковое
    #     if splits == 3 and split == 'train':
    #         end_index = len(self.df) - self.prediction_len * 2
    #     elif (splits == 3 and split == 'validation' or
    #           splits == 2 and split == 'train'):
    #         end_index = len(self.df) - self.prediction_len
    #     elif split == 'test':
    #         end_index = len(self.df)
    #     else:
    #         raise Exception(f'Неверное сочетание имени выборки [{split}] и количества выборок [{splits}].')
    #
    #     # дата начала временной последовательности - одинаковая для всех данных
    #     # переводим в timestamp, т.к. Arrow не понимает Period, -
    #     # будем конвертировать в Period на этапе трансформации
    #     start_date = self.df.index.min().to_timestamp()
    #
    #     for channel, index in CHANNEL_INDEX.items():
    #         yield {
    #             'start': start_date,
    #             'target': self.df[channel].iloc[:end_index].to_list(),
    #             # статический признак последовательности
    #             'feat_static_cat': [index]
    #         }


if __name__ == '__main__':
    os.chdir('..')
    logger.setup(level=logger.INFO, layout='debug')

    check_hp = Hyperparameters()
    # сгенерируем набор дефолтных гиперпараметров и посмотрим на их значения
    values, bot_hash = check_hp.generate(mode='default', hashes=[])
    # print(check_hp.repr(values))

    # создаём менеджер датасета и готовим исходный DataFrame для формирования выборок
    dm = DataManager()
    dm.prepare(
        freq=check_hp.get('freq'),
        prediction_len=check_hp.get('prediction_len')
    )

    # формируем выборки
    train_dataset = Dataset.from_generator(generator, gen_kwargs={
        'df': dm.df, 'splits': 2, 'split': 'train', 'prediction_len': check_hp.get('prediction_len')})
    test_dataset = Dataset.from_generator(generator, gen_kwargs={
        'df': dm.df, 'splits': 2, 'split': 'test', 'prediction_len': check_hp.get('prediction_len')})

    for sample in train_dataset:
        print(sample)
