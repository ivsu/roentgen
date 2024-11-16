import numpy as np
import pandas as pd
import os
from transformers import TimeSeriesTransformerConfig
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from schedule.dataloader import DataLoader
from workplan.datamanager import DataManager, get_channels_settings, CHANNEL_LEGEND, COLLAPSED_CHANNELS, ALL_CHANNELS
from workplan.hyperparameters import Hyperparameters
from workplan.dataloaders import create_train_dataloader
from workplan.show import time_series_by_year, activate_plotly
# import settings  # загружается, чтобы сформировать переменную среды RUN_ENV


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

    channels, _, _ = get_channels_settings()

    data = []
    for ch_index, item in enumerate(ds):
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
            index=ch_index,
            label=channels[ch_index],
            start_date=start_date,
            years=years
        ))
    if verbose:
        print('Данные по РГ:')
        for key, value in data[6].items():
            if key == 'years':
                for year, target in value.items():
                    print(f'{year} ({len(target)}): {target}')
            else:
                print(f'{key}: {value}')

    return data


def show_time_series_by_year(data_version):

    activate_plotly()

    hp = Hyperparameters()

    if data_version == 'debug':
        hp.set('prediction_len', 13)

    dm = DataManager(date_cut=False)
    dm.read_and_prepare(
        freq=hp.get('freq'),
        prediction_len=hp.get('prediction_len'),
        data_version=data_version,
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
    assert 'ROENTGEN.N_CHANNELS' in os.environ, 'Не задано количество каналов в переменных среды.'
    n_channels = int(os.environ['ROENTGEN.N_CHANNELS'])
    assert n_channels in [6, 10]
    channels = COLLAPSED_CHANNELS if n_channels == 6 else ALL_CHANNELS

    fig = go.Figure(
        data=[go.Table(
            header=dict(
                values=channels,
                height=30,
            ),
            cells=dict(
                values=[CHANNEL_LEGEND[key] for key in channels],
                height=30,
            )
        )])
    fig.update_layout(
        title_text='Расшифровка модальностей врачей',
        width=900,
        # autosize=False,
        height=240,  # это примерно (для точности нужно учесть высоту заголовка)
        # grid={'rows': 1, 'columns': len(params), 'pattern': 'independent'},
        paper_bgcolor='rgba(222, 222, 222, 1)',
        # plot_bgcolor='rgba(128, 128, 192, 1)',  # для индикаторов, похоже, не работает
        # margin={'autoexpand': False, 'pad': 30, 'l': 20}
        # margin=dict(l=30, r=30, t=30, b=20, pad=40),
    )
    fig.show()


def show_sample_example(batch_size):
    _, _, n_channels = get_channels_settings()
    hp = Hyperparameters()

    # создаём менеджер датасета и готовим исходный DataFrame для формирования выборок
    dm = DataManager()
    dm.read_and_prepare(
        freq=hp.get('freq'),
        prediction_len=hp.get('prediction_len'),
        data_version='train'
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
        cardinality=[n_channels],
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

    # установим количество каналов данных
    os.environ['ROENTGEN.N_CHANNELS'] = '10'

    show_time_series_by_year(data_version='train')  # source, train, debug
    # show_doctors()
    # show_legend()