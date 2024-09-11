import os
import gc
import pandas as pd
from datetime import datetime, timedelta
from datasets import Dataset
from functools import lru_cache

from common.logger import Logger
from workplan.hyperparameters import Hyperparameters
from settings import DB_VERSION
if DB_VERSION == 'PG':
    from common.db import DB, get_all
else:
    from common.dblite import DB, get_all

logger = Logger(__name__)

# текстовые метки каналов данных
CHANNEL_NAMES = ['kt', 'kt_ce1', 'kt_ce2', 'mrt', 'mrt_ce1', 'mrt_ce2', 'rg', 'flg', 'mmg', 'dens']
COLLAPSED_CHANNEL_NAMES = ['kt', 'mrt', 'rg', 'flg', 'mmg', 'dens']
# словарь с именами и индексами каналов данных
CHANNEL_INDEX = {name: ch for ch, name in enumerate(CHANNEL_NAMES)}

FREQ = None


def generator(df, splits, split, prediction_len, end_shift):
    """
    Генератор выборок.

    :param df: датафрейм для формирования выборки
    :param splits: количество выборок, на которое делится датасет (2, 3)
    :param split: имя сплита (train, validation, test)
    :param prediction_len: глубина предсказания
    :param end_shift: сдвиг на количество шагов (с конца) при обучении бота на временном ряде разной длины

    """
    assert splits in [2, 3]
    assert split in ['train', 'validation', 'test']

    ts_len = len(df) + end_shift

    # определим конечный индекс данных, в зависимости от выборки
    # сплиты будут разной длины на prediction_len, но начало у них всех одинаковое
    if splits == 3 and split == 'train':
        end_index = ts_len - prediction_len * 2
    elif (splits == 3 and split == 'validation' or
          splits == 2 and split == 'train'):
        end_index = ts_len - prediction_len
    elif split == 'test':
        end_index = ts_len
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


@lru_cache(10_000)
def _convert_to_pandas_period(date):
    """
    Конвертирует дату в pd.Period соответствующей частоты.
    Данные преобразования кэшируются.
    """
    return pd.Period(date, FREQ)


def _transform_start_field(batch):
    """
    Конвертирует признак start в pd.Period соответствующей частоты по батчу.
    Используется для конвертации на лету.
    """
    # данные преобразования кэшируются
    batch["start"] = [_convert_to_pandas_period(date) for date in batch["start"]]
    return batch


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
        # self.freq = None
        # длина временного ряда
        self.ts_len = None
        # количество каналов данных
        self.n_channels = None
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

        def week_to_date_time(row):
            first_date = datetime(row['year'], 1, 1)
            return first_date + timedelta(weeks=row['week'] - 1, days=7-first_date.weekday())

        def compose_channel(row):
            return row['modality'] \
                if row['ce'] == 'none' \
                else row['modality'] + '_' + row['ce']

        # by_mod = df['modality'] == 'rg'
        # by_year = df['year'] == 2024
        # print(df[by_mod & by_year])
        df['datetime'] = df.apply(week_to_date_time, axis=1)
        df['channel'] = df.apply(compose_channel, axis=1)

        df = df.pivot(index=['datetime'], columns=['channel'], values=['amount'])
        # print(df.columns)
        df.columns = df.columns.levels[1]
        self.df = df
        self.ts_len = len(df)
        self.n_channels = len(df.columns)

        logger.info('Данные загружены.')
        # logger.info(f'self.df:\n{self.df.head()}')

    def read_and_prepare(self, freq, prediction_len):
        """ Подготовка источника данных - DataFrame """
        # self.freq = freq
        global FREQ
        FREQ = freq
        self.prediction_len = prediction_len
        # загружаем данные в датафрейм
        self.read()
        # переводим индекс в pd.Period
        self.df.index = self.df.index.to_period(freq)
        # print(f'self.df:\n{self.df}')

    def from_generator(self, splits, split, end_shift):
        """ Возвращает датасет в зависимости от аргумента split"""
        ds = Dataset.from_generator(generator, gen_kwargs=dict(
            df=self.df, splits=splits, split=split, prediction_len=self.prediction_len, end_shift=end_shift
        ))
        # используем функциональность датасета set_transform для конвертации
        # признака start в pd.Period на лету
        ds.set_transform(_transform_start_field)

        return ds

    def get_ts_len(self):
        return self.ts_len

    def get_channels_num(self):
        return self.n_channels


if __name__ == '__main__':
    os.chdir('..')
    logger.setup(level=logger.INFO, layout='debug')

    hp = Hyperparameters()

    # создаём менеджер датасета и готовим исходный DataFrame для формирования выборок
    dm = DataManager()
    dm.read_and_prepare(
        freq=hp.get('freq'),
        prediction_len=hp.get('prediction_len')
    )

    context = {
        # текущая общая длина временного ряда (для случая, когда используется разное количество данных)
        'ts_len': dm.get_ts_len(),
        'n_channels': len(CHANNEL_NAMES)
    }

    # сгенерируем набор дефолтных гиперпараметров и посмотрим на их значения
    values, bot_hash = hp.generate(mode='default', hashes=[], context=context)
    # print(test_hp.repr(values))

    # формируем выборки
    train_dataset = dm.from_generator(splits=2, split='train', end_shift=0)
    test_dataset = dm.from_generator(splits=2, split='test', end_shift=0)

    for sample in train_dataset:
        print(sample)

