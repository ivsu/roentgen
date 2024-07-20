import os
import gc
import pandas as pd
from datetime import datetime, timedelta
from datasets import Dataset
from functools import lru_cache, partial
from common.logger import Logger
from lib.db import DB, get_all
from hyperparameters import Hyperparameters

logger = Logger(__name__)


class DataManager:
    # текстовые метки каналов данных
    CHANNEL_NAMES = ['kt', 'kt_ce1', 'kt_ce2', 'mrt', 'mrt_ce1', 'mrt_ce2', 'rg', 'flg', 'mmg', 'dens']
    # словарь с именами и индексами каналов данных
    CHANNEL_INDEX = {name: ch for ch, name in enumerate(CHANNEL_NAMES)}

    def __init__(self):
        """
        Класс для чтения данных с диска и формирования выборок.
        Также позволяет сгруппировать данные до заданной частоты.
        Хранит полученный датафрейм для генерации датасетов.
        """
        # глубина предсказания - длина предсказываемой последовательности
        self.prediction_len = None
        # датафрейм - источник данных для выборок датасетов
        self.df = None
        # частота данных
        self.freq = '1W'

    def read(self):
        db = DB()
        query = """
            select
                year, week, modality, contrast_enhancement as ce, amount
            from "roentgen".work_summary
            where year >= 2022
            order by year, week
            ;
        """
        with db.get_cursor() as cursor:
            cursor.execute(query)
            df: pd.DataFrame = get_all(cursor)
        db.close()

        # просуммируем данные по неполным неделям в начале и конце года
        if False:
            check_sum = df['amount'].sum()
            for year in range(df['year'].min() + 1, df['year'].max() + 1):
                # print(f'year: {year}')
                first_date = datetime(year, 1, 1)
                if first_date.weekday() == 0:
                    continue
                last_week_cond = (df['year'] == (year - 1)) & (df['week'] == df['week'].max())
                first_week_cond = (df['year'] == year) & (df['week'] == 1)
                summarized = df[last_week_cond | first_week_cond]
                # print(f'summarized:\n{summarized}')
                summarized = summarized.groupby(['modality', 'ce'], as_index=False).sum()
                summarized['year'] = year
                summarized['week'] = 1
                # print(f'summarized:\n{summarized}')
                df.drop(df[last_week_cond | first_week_cond].index, inplace=True)
                # print(f'dropped rows: {df[last_week_cond | first_week_cond]}')
                df = pd.concat([df, summarized], axis=0, ignore_index=True)
                df.sort_values(['year', 'week', 'modality', 'ce'])
                first_week_cond = (df['year'] == year) & (df['week'] == 1)
                # print(f'inserted rows:\n{df[first_week_cond]}')
            assert check_sum == df['amount'].sum()

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
        print(df.columns)

        df = df.pivot(index=['datetime'], columns=['channel'], values=['amount'])
        print(df.columns.levels)
        df.columns = df.columns.levels[1]
        print(df.columns)
        self.df = df

        logger.info('Данные загружены.')
        # logger.info(f'self.df:\n{self.df.head()}')

    def prepare(self, freq, prediction_len):
        """ Подготовка источника данных - DataFrame """
        self.freq = freq
        self.prediction_len = prediction_len
        # загружаем данные в датафрейм
        self.read()
        # self._rebuild(freq)
        # переводим индекс в pd.Period
        self.df.index = self.df.index.to_period(freq)
        print(f'self.df:\n{self.df}')

    def _rebuild(self, freq):
        """
        Группирует исходный датафрейм до заданной частоты,
        добавляет индекс data (PeriodIndex), который равномерно следует
        с заданным временным шагом, - при этом также образуются значения NaN,
        для которых при определении трансформации в дальнейшем формируется маска,
        по которой декодер не формирует значения функции потерь для пустых значений.
        """
        # создаём колонку с заданным периодом (pd.Period)
        self.df['date'] = self.df['datetime'].dt.to_period(freq)
        logger.info(f'Создана колонка периода "date", тип: {self.df["date"].dtype}')
        # print(self.df.head())

        # группируем признаки по заданной частоте
        self.df = self.df.groupby('date').agg(
            open=pd.NamedAgg('open', 'first'),
            high=pd.NamedAgg('high', 'max'),
            low=pd.NamedAgg('low', 'min'),
            close=pd.NamedAgg('close', 'last')
        )
        logger.info(f'Данные сгруппированы по периоду: {freq}, строк: {len(self.df)}\nHEAD:\n{self.df.head()}')

        # формируем полный RangeIndex дат
        full_range = pd.period_range(
            self.df.index.min().to_timestamp(),
            self.df.index.max().to_timestamp(),
            freq=freq,
            name='date'
        )
        logger.info(f'\nСформирован полный диапазон дат, всего шагов: {len(full_range)}')

        # готовим пустой датафрейм с полным Period-индексом
        range_df = pd.DataFrame(index=full_range)
        # присоединяем данные
        self.df = range_df.join(self.df)
        logger.info(f'Диапазон значений: {len(self.df)}')
        logger.info(f'HEAD:\n{self.df.head(10)}')

        del range_df, full_range
        gc.collect()

    def from_generator(self, splits, split):
        """ Возвращает датасет в зависимости от аргумента split"""
        ds = Dataset.from_generator(self._gen, gen_kwargs={
            'splits': splits, 'split': split
        })
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

    def _gen(self, splits, split):
        """
        Генератор выборок.

        :param splits: количество выборок, на которое делится датасет (2, 3)
        :param split: имя сплита (train, validation, test)
        """
        assert splits in [2, 3]
        assert split in ['train', 'validation', 'test']

        # определим конечный индекс данных, в зависимости от выборки
        # сплиты будут разной длины на prediction_len, но начало у них всех одинаковое
        if splits == 3 and split == 'train':
            end_index = len(self.df) - self.prediction_len * 2
        elif (splits == 3 and split == 'validation' or
              splits == 2 and split == 'train'):
            end_index = len(self.df) - self.prediction_len
        elif split == 'test':
            end_index = len(self.df)
        else:
            raise Exception(f'Неверное сочетание имени выборки [{split}] и количества выборок [{splits}].')

        # дата начала временной последовательности - одинаковая для всех данных
        # переводим в timestamp, т.к. Arrow не понимает Period, -
        # будем конвертировать в Period на этапе трансформации
        start_date = self.df.index.min().to_timestamp()

        for channel, index in self.CHANNEL_INDEX.items():
            yield {
                'start': start_date,
                'target': self.df[channel].iloc[:end_index].to_list(),
                # статический признак последовательности
                'feat_static_cat': [index]
            }

    def _newname(self, sample):
        """ Формирует имя файла рабочего датасета """
        parts = self.path.split('/')
        parts[-1] = '/data/' + sample + '.csv'
        return '/'.join(parts)


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
    # print(dm.df.columns)
    print(dm.df['rg'])

    # формируем выборки
    train_dataset = dm.from_generator(splits=2, split='train')
    test_dataset = dm.from_generator(splits=2, split='test')
