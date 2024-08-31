import numpy as np
import pandas as pd
import os

from schedule.dataloader import DataLoader
from workplan.datamanager import DataManager, CHANNEL_NAMES
from workplan.hyperparameters import Hyperparameters
from workplan.show import time_series_by_year
import settings



def convert_dataset(ds, skip_start_weeks=0, verbose=0) -> list[dict]:
    """
    Конвертирует датасет в список объектов по годам в формате
        data = [{
            'index': индекс канала,
            'label': метка канала,
            'start_date': дата начала данных,
            'years': {
                2021: np.array([...]),
                2022: np.array([...]),
                ...
            }, ...]
    :param ds: датасет (Dataset)
    :param skip_start_weeks: количество недель пропускаемых в начале данных
    :param verbose: признак вывода отладочной информации
    """

    data = []
    for channel_index, item in enumerate(ds):
        # конвертируем колонку периода в дату начала недели
        start_date: pd.Timestamp = item['start'].to_timestamp(how='start').floor('d')
        # print(f'item start: {item["start"]}')
        start_date = start_date + pd.Timedelta(weeks=skip_start_weeks)
        start_year = start_date.year
        year_end = start_date.replace(month=12, day=31)
        # print(f'year_end_weekday: {year_end.weekday()}')
        # считаем разницу между воскресеньем последней недели года и начальной датой
        diff = (year_end + pd.Timedelta(days=6 - year_end.weekday())) - start_date
        # считаем количество недель первого года, в которых есть данные
        year_weeks = diff.days // 7 + 1
        # print(f'start_date: {start_date}, year_end: {year_end}')
        # print(f'diff: {diff}, year_weeks: {year_weeks}')

        target = item['target'][skip_start_weeks:]
        years = {}
        total_years = (52 - year_weeks + len(target)) // 52 + 1
        # print(f'total_years: {total_years}')
        start_row = 0
        end_row = year_weeks
        for i, year in enumerate(range(start_year, start_year + total_years)):
            years[year] = np.array(target[start_row:end_row])
            start_row = end_row
            end_row += 52

        data.append(dict(
            index=channel_index,
            label=CHANNEL_NAMES[channel_index],
            start_date=start_date,
            years=years
        ))
    if verbose:
        print('Данные по РГ:')
        for key, value in data[6].items():
            if key == 'years':
                for year, target  in value.items():
                    print(f'{year} ({len(target)}): {target}')
            else:
                print(f'{key}: {value}')

    return data


def show_time_series_by_year():

    hp = Hyperparameters()

    dm = DataManager()
    dm.read_and_prepare(
        freq=hp.get('freq'),
        prediction_len=hp.get('prediction_len')
    )
    # формируем полную выборку
    ds = dm.from_generator(splits=2, split='test', end_shift=0)

    data = convert_dataset(ds, skip_start_weeks=0, verbose=0)
    time_series_by_year(data)


def show_doctors():
    """Отображает датафрейм с врачами"""
    dataloader = DataLoader()
    doctor_df = dataloader.get_doctors()
    doctor_df.drop(['uid'], axis=1, inplace=True)
    if os.environ['RUN_ENV'] == 'COLAB':
        doctor_df
    else:
        expand_pandas_output()
        print(doctor_df.head(10))


def expand_pandas_output():
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)


if __name__ == '__main__':
    # show_time_series_by_year()
    show_doctors()