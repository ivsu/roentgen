import os
import gc
import pandas as pd
from datetime import datetime, timedelta
from datasets import Dataset
from functools import lru_cache
from datasets import disable_caching
from datasets import config as datasets_config

from common.logger import Logger
from workplan.hyperparameters import Hyperparameters
from workplan.test_dataset import generate_debug_df
from settings import DB_VERSION
if DB_VERSION == 'PG':
    from common.db import DB, get_all
else:
    from common.dblite import DB, get_all

logger = Logger(__name__)

# текстовые метки каналов данных для полного и просуммированного набора каналов
ALL_CHANNELS = ['kt', 'kt_ce1', 'kt_ce2', 'mrt', 'mrt_ce1', 'mrt_ce2', 'rg', 'flg', 'mmg', 'dens']
COLLAPSED_CHANNELS = ['kt', 'mrt', 'rg', 'flg', 'mmg', 'dens']
CHANNEL_LEGEND = {
    'kt': 'КТ', 'kt_ce1': 'КТ с контрастом, вариант 1', 'kt_ce2': 'КТ с контрастом, вариант 2',
    'mrt': 'МРТ', 'mrt_ce1': 'МРТ с контрастом, вариант 1', 'mrt_ce2': 'МРТ с контрастом, вариант 2',
    'rg': 'Рентгенография', 'flg': 'Флюорография',
    'mmg': 'Маммография', 'dens': 'Денситометрия'
}
# частота данных (используется в методе конвертации в Period)
FREQ = None


def get_channels_settings():
    try:
        n_channels = int(os.environ['ROENTGEN.N_CHANNELS'])
    except KeyError:
        raise RuntimeError('Не задана переменная среды: ROENTGEN.N_CHANNELS')

    assert n_channels in [6, 10]
    channels = COLLAPSED_CHANNELS if n_channels == 6 else ALL_CHANNELS
    return channels, {name: ch for ch, name in enumerate(channels)}, n_channels


def generator(df, splits, split, prediction_len, end_shift):
    """
    Генератор выборок.

    :param df: датафрейм для формирования выборки
    :param splits: количество выборок, на которое делится датасет (2, 3)
    :param split: имя сплита (train, validation, test)
    :param prediction_len: глубина предсказания
    :param end_shift: сдвиг на количество шагов (с конца) при обучении бота на временном ряде разной длины

    """
    assert split in ['train', 'validation', 'test']

    ts_len = len(df) + end_shift  # end_shift <= 0
    _, channel_index, _ = get_channels_settings()

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

    for channel, index in channel_index.items():
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

    def __init__(self, date_cut=True):
        """
        Класс для чтения данных с диска и формирования выборок.
        Также позволяет сгруппировать данные до заданной частоты.
        Хранит полученный датафрейм для генерации датасетов.
        """
        self.date_cut = date_cut
        # датафрейм - источник данных для выборок датасетов
        self.df = None
        # глубина предсказания - длина предсказываемой последовательности
        self.prediction_len = None
        # длина временного ряда
        self.ts_len = None
        # количество каналов данных
        self.n_channels = None
        # префикс схемы БД
        self.db_schema_prefix = 'roentgen.' if DB_VERSION == 'PG' else ''
        # версия данных, считываемая из БД (для work_summary)
        self.data_version = None

    def read(self):
        _, _, n_channels = get_channels_settings()

        db = DB()
        query = f"""
            select
                year, week, modality, contrast_enhancement as ce, amount
            from {self.db_schema_prefix}work_summary
            where version = '{self.data_version}'
            order by year, week
            ;
        """
        with db.get_cursor() as cursor:
            cursor.execute(query)
            df: pd.DataFrame = get_all(cursor)
        db.close()

        if len(df) == 0:
            raise RuntimeError("Из БД получен пустой датафрейм.")

        def week_to_date_time(row):
            first_date = datetime(row['year'], 1, 1)
            # если год начинается с понедельника, занулим смещение по дню недели
            days_shift = 7 - first_date.weekday() if first_date.weekday() > 0 else 0
            return first_date + timedelta(weeks=row['week'] - 1, days=days_shift)

        # for year in range(2021, 2025):
        #     week = 1
        #     data = dict(year=year, week=week)
        #     year_start = datetime(year, 1, 1)
        #     print(f'week: {week:2d}, date: {week_to_date_time(data)}, year_start weekday: {year_start.weekday()}')
        #     week = 52
        #     data = dict(year=year, week=week)
        #     print(f'week: {week:2d}, date: {week_to_date_time(data)}')

        def compose_channel(row):
            return row['modality'] \
                if row['ce'] == 'none' \
                else row['modality'] + '_' + row['ce']

        df['datetime'] = df.apply(week_to_date_time, axis=1)
        df['channel'] = df.apply(compose_channel, axis=1)

        df = df.pivot(index=['datetime'], columns=['channel'], values=['amount'])
        # оставляем индекс только по модальностям (убираем amount)
        df.columns = df.columns.levels[1]

        # объединим данные по контрастному усилению
        if n_channels == 6:
            df['kt'] = df[['kt', 'kt_ce1', 'kt_ce2']].sum(axis=1)
            df['mrt'] = df[['mrt', 'mrt_ce1', 'mrt_ce2']].sum(axis=1)
            df.drop(['kt_ce1', 'kt_ce2', 'mrt_ce1', 'mrt_ce2'], axis=1, inplace=True)

        # если задана дата начала прогноза, урежем датафрейм с этой даты
        if self.date_cut:
            assert 'ROENTGEN.FORECAST_START_DATE' in os.environ, 'Дата начала прогноза не найдена в переменных среды.'
            forecast_start_date = os.environ['ROENTGEN.FORECAST_START_DATE']
            df = df[df.index < datetime.fromisoformat(forecast_start_date)]
        print(f'Конечная дата в данных: {df.index.max()}')

        # print(f'columns: {df.columns}')
        self.df = df
        self.ts_len = len(df)
        self.n_channels = n_channels

        logger.info('Данные загружены.')
        # logger.info(f'self.df:\n{self.df.head()}')

    def read_and_prepare(self, freq, prediction_len, data_version):
        """ Подготовка источника данных - DataFrame """
        assert data_version in ['source', 'train', 'debug'], "Неизвестная версия данных: " + data_version
        global FREQ
        FREQ = freq
        self.prediction_len = prediction_len
        self.data_version = data_version
        # загружаем данные в датафрейм
        if data_version == 'debug':
            self.generate_debug_data()
        else:
            self.read()
            # переводим индекс в pd.Period
            self.df.index = self.df.index.to_period(freq)
        # print(f'self.df:\n{self.df}')

    def generate_debug_data(self):
        channels, _, n_channels = get_channels_settings()
        dots_per_period = 13
        self.df = generate_debug_df(channels, dots_per_period)
        self.ts_len = len(self.df)
        self.n_channels = n_channels

        logger.info('Загружен отладочный датасет.')

    def from_generator(self, splits, split, end_shift):
        """ Возвращает датасет в зависимости от аргумента split"""
        # TODO: не работает
        datasets_config.IN_MEMORY_MAX_SIZE = 10000
        disable_caching()

        ds = Dataset.from_generator(generator, keep_in_memory=True, gen_kwargs=dict(
            df=self.df, splits=splits, split=split, prediction_len=self.prediction_len, end_shift=end_shift
        ))
        # используем функциональность датасета set_transform для конвертации
        # признака start в pd.Period на лету
        ds.set_transform(_transform_start_field)

        return ds


if __name__ == '__main__':
    os.chdir('..')
    logger.setup(level=logger.INFO, layout='debug')

    # установим количество каналов данных
    os.environ['ROENTGEN.N_CHANNELS'] = '6'

    os.environ['ROENTGEN.FORECAST_START_DATE'] = '2024-04-29'

    hp = Hyperparameters()

    # создаём менеджер датасета и готовим исходный DataFrame для формирования выборок
    dm = DataManager()
    dm.read_and_prepare(
        freq=hp.get('freq'),
        prediction_len=hp.get('prediction_len'),
        data_version='debug'  # source, train, debug
    )

    # сгенерируем набор дефолтных гиперпараметров и посмотрим на их значения
    values, bot_hash = hp.generate(mode='default', hashes=[])
    # print(test_hp.repr(values))

    # формируем выборки
    # dataset = dm.from_generator(splits=2, split='train', end_shift=0)
    dataset = dm.from_generator(splits=2, split='test', end_shift=0)

    for sample in dataset:
        print(sample)
