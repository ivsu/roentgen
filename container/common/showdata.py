import numpy as np
import pandas as pd
import os
from transformers import TimeSeriesTransformerConfig

from schedule.dataloader import DataLoader
from workplan.datamanager import DataManager, CHANNEL_NAMES
from workplan.hyperparameters import Hyperparameters
from workplan.dataloaders import create_train_dataloader
from workplan.show import time_series_by_year
# import settings  # загружается, чтобы сформировать переменную среды RUN_ENV


CHANNEL_LEGEND = {
    'kt': 'КТ', 'kt_ce1': 'КТ с контрастом, вариант 1', 'kt_ce2': 'КТ с контрастом, вариант 2',
    'mrt': 'МРТ', 'mrt_ce1': 'МРТ с контрастом, вариант 1', 'mrt_ce2': 'МРТ с контрастом, вариант 2',
    'rg': 'Рентгенография', 'flg': 'Флюорография',
    'mmg': 'Маммография', 'dens': 'Денситометрия'
}


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
    expand_pandas_output()
    print(doctor_df.head(10))


def show_legend():
    print('Расшифровка модальностей врачей:')
    print("\n".join(f'{k:>7}: {v}' for k, v in CHANNEL_LEGEND.items()))


def show_sample_example(batch_size):
    hp = Hyperparameters()

    # создаём менеджер датасета и готовим исходный DataFrame для формирования выборок
    dm = DataManager()
    dm.read_and_prepare(
        freq=hp.get('freq'),
        prediction_len=hp.get('prediction_len')
    )

    # сгенерируем набор дефолтных гиперпараметров и посмотрим на их значения
    values, bot_hash = hp.generate(mode='default', hashes=[])
    # print(test_hp.repr(values))

    # формируем выборки
    train_dataset = dm.from_generator(splits=2, split='train', end_shift=0)
    test_dataset = dm.from_generator(splits=2, split='test', end_shift=0)

    # for sample in train_dataset:
    #     print(sample)

    time_features = hp.time_features_set[-1].copy()
    lags_sequence = hp.lags_sequence_set[-1].copy()

    prediction_len = hp.get('prediction_len')
    freq = hp.get('freq')

    # тестовая конфигурация
    config = TimeSeriesTransformerConfig(
        # длина предсказываемой последовательности
        prediction_length=prediction_len,
        # длина контекста:
        context_length=prediction_len,
        # временные лаги
        lags_sequence=lags_sequence,
        # количество временных признака + 1 (возраст временного шага) будет добавлен при трансформации
        num_time_features=len(time_features) + 1,
        # единственный статический категориальный признак - ID серии:
        num_static_categorical_features=1,
        # количество каналов
        cardinality=[len(CHANNEL_NAMES)],
        # размерность эмбеддингов
        embedding_dimension=[2],
        # параметры трансформера
        encoder_layers=2,
        decoder_layers=2,
        d_model=32,
    )
    # формируем загрузчик данных
    train_dataloader = create_train_dataloader(
        config=config,
        freq=freq,
        data=train_dataset,
        batch_size=batch_size,
        num_batches_per_epoch=1,
        time_features=time_features,
    )

    print(f'Частотность данных: {freq}')
    print(f'Глубина предикта: {prediction_len} недель')
    print(f'Динамические признаки: {time_features}')
    print(f'Временные лаги (на сколько шагов "смотрим назад"): {lags_sequence}')

    for sample in train_dataloader:
        print('\nСостав данных одного сэмпла:')
        data = sample['static_categorical_features']
        print(f'\nСтатические временные признаки / static_categorical_features, shape {tuple(data.shape)}:')
        print(data[:, 0])
        data = sample['past_values']
        print(f'\nПрошлый временной ряд / past_values, shape {tuple(data.shape)}. Пример:')
        print(data[0, :])
        data = sample['past_observed_mask']
        print(f'\nМаска прошлого временного ряда / past_observed_mask, shape {tuple(data.shape)}. Пример:')
        print(data[0, :])
        data = sample['past_time_features']
        print(f'\nПрошлые временные признаки / past_time_features, shape {tuple(data.shape)}. Пример:')
        print(data[0, -10:, :])
        data = sample['future_values']
        print(f'\nБудущий временной ряд / future_values, shape {tuple(data.shape)}. Пример:')
        print(data[0, :])
        data = sample['future_observed_mask']
        print(f'\nМаска будущего временного ряда / future_observed_mask, shape {tuple(data.shape)}. Пример:')
        print(data[0, :])
        data = sample['future_time_features']
        print(f'\nБудущие временные признаки / future_time_features, shape {tuple(data.shape)}. Пример:')
        print(data[0, -10:, :])


def expand_pandas_output():
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)


if __name__ == '__main__':
    # show_time_series_by_year()
    # show_doctors()
    show_legend()