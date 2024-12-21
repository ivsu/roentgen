import os

import uuid

import json
import glob
import gc
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import copy

# для обучения модели
from accelerate import Accelerator
from torch.optim import AdamW
import torch
# для расчёта метрик MASE/sMAPE
from evaluate import load
# from gluonts.time_feature import get_seasonality

from workplan.modelbuilder import ModelBuilder
from workplan.show import dashboard
from workplan.dataloaders import create_train_dataloader, create_test_dataloader
from workplan.schedulers import WarmupAndDecayScheduler
from workplan.datamanager import get_channels_settings
from common.logger import Logger
from common.showdata import expand_pandas_output, activate_plotly
from settings import DB_VERSION
if DB_VERSION == 'PG':
    from common.db import DB, get_all
else:
    from common.dblite import DB, get_all

logger = Logger(__name__)

# периодичность временного ряда для расчёта метрики MASE
MASE_METRIC_PERIODICITY = 52
# признак управления LR после каждого батча (иначе в рамках эпохи применяется один и тот же LR ко всем батчам)
CHANGE_LR_ON_EVERY_BATCH = True
# количество временных рядов
N_CHANNELS = None


def train_model(model, config, dataloader, bot, stage_index, n_stages):
    epochs = bot.get('n_epochs')
    warmup_epochs = bot.get('warmup_epochs')
    decay_epochs = bot.get('decay_epochs')
    steps = 0
    # на первой стадии будем обучать трансформер с прогревом
    final_lr = bot.get('final_lr')
    initial_lr = bot.get('initial_lr') if stage_index == 0 else final_lr
    target_lr = bot.get('target_lr') if stage_index == 0 else final_lr

    # определим количество батчей
    num_batches = len(list(dataloader))

    # используем Accelerator от HuggingFace
    accelerator = Accelerator()
    device = accelerator.device
    model.to(device)

    # создаём оптимизатор Adam с дополнительным методом затухания весов
    # и передаём ему параметры модели
    optimizer = AdamW(model.parameters(), betas=(0.9, 0.95), weight_decay=1e-1)

    scheduler = WarmupAndDecayScheduler(
        optimizer,
        warmup_steps=num_batches * warmup_epochs,
        decay_steps=num_batches * decay_epochs,
        decay_rate=0.3,
        initial_lr=initial_lr,
        target_lr=target_lr,
        final_lr=final_lr,
        steps_per_epoch=num_batches,
        change_on_every_batch=CHANGE_LR_ON_EVERY_BATCH,
        device=device
    )

    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )

    model.train()
    losses = []
    learning_rates = []
    start_time = datetime.now()
    for epoch in range(epochs):
        epoch_losses = []
        for idx, batch in enumerate(dataloader):
            steps += 1
            optimizer.zero_grad()
            # выполняем шаг обучения на одном батче и получаем ошибку модели
            outputs = model(
                static_categorical_features=batch["static_categorical_features"].to(device)
                if config.num_static_categorical_features > 0
                else None,
                static_real_features=batch["static_real_features"].to(device)
                if config.num_static_real_features > 0
                else None,
                past_time_features=batch["past_time_features"].to(device),
                past_values=batch["past_values"].to(device),
                past_observed_mask=batch["past_observed_mask"].to(device),
                future_time_features=batch["future_time_features"].to(device),
                future_values=batch["future_values"].to(device),
                future_observed_mask=batch["future_observed_mask"].to(device),
            )
            loss = outputs.loss
            epoch_losses.append(loss.item())

            # обратное распространение
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()

        losses.append(epoch_losses)
        # запоминаем текущий LR
        lr = optimizer.param_groups[0]['lr'].item()
        learning_rates.append(lr)

        mean_loss = sum(epoch_losses) / len(epoch_losses)
        # scheduler.step(mean_loss)
        t = datetime.now() - start_time

        end = chr(10) if epoch == epochs - 1 else ''
        stage_prefix = f'stage: {stage_index + 1}/{n_stages} | '
        print(f'\rstage: {stage_index + 1}/{n_stages}'
              f' | epoch: {epoch + 1:3d}'
              f' | mean loss: {mean_loss:.4f}'
              f' | LR: {learning_rates[-1]:.6f}'
              f' | steps: {steps:4d}'
              f' | total time: {t.seconds:3.0f}s',
              end=end)

    return model, losses, device, t.seconds, learning_rates


def inference(model, dataloader, config, device):
    """
    Реализует инференс модели с помощью метода generate.
    Возвращает вероятностый прогнозный вектор формы:
    (количество_временных_рядов, config.num_parallel_samples [100], глубина_предикта),
    Например: (6, 100, 5)
    """
    model.eval()

    forecasts = []

    for batch in dataloader:
        outputs = model.generate(
            static_categorical_features=batch["static_categorical_features"].to(device)
            if config.num_static_categorical_features > 0
            else None,
            static_real_features=batch["static_real_features"].to(device)
            if config.num_static_real_features > 0
            else None,
            past_time_features=batch["past_time_features"].to(device),
            past_values=batch["past_values"].to(device),
            future_time_features=batch["future_time_features"].to(device),
            past_observed_mask=batch["past_observed_mask"].to(device),
        )
        forecasts.append(outputs.sequences.cpu().numpy())

    forecasts = np.vstack(forecasts)
    logger.debug(f'Обработано батчей тестовой выборки: {len(forecasts)}')
    logger.debug(f'Форма вероятностного вектора прогноза: {forecasts.shape}')

    return forecasts


def convert_data_for_metrics(dataset, forecasts):
    """:deprecated:"""
    # сложим значения по КТ и МРТ с разным контрастным усилением, т.к. эти исследования выполняются
    # врачами с той же самой квалификацией
    forecasts_converted = np.concatenate([
        forecasts[0:3].sum(axis=0, keepdims=True),
        forecasts[3:6].sum(axis=0, keepdims=True),
        forecasts[6:],
    ])
    target_converted = np.array([item['target'] for item in list(dataset)])
    target_converted = np.concatenate([
        target_converted[0:3].sum(axis=0, keepdims=True),
        target_converted[3:6].sum(axis=0, keepdims=True),
        target_converted[6:],
    ])
    return forecasts_converted, target_converted


import numpy as np


def calculate_mase(actual, forecast, prediction_len, seasonal_period):
    """
    Рассчитывает MASE для сезонного временного ряда.
    Сделана для проверки расчёта метрики.

    :param actual: numpy массив с фактическими значениями
    :param forecast: numpy массив с предсказанными значениями
    :param seasonal_period: длина сезонного компонента

    :return: значение MASE
    """
    ground_truth = actual[-prediction_len:]
    # исключим NaN на входе (дни, когда не было данных) из расчёта метрик
    future_mask = np.isnan(ground_truth)

    # Вычисление абсолютной ошибки
    absolute_error = np.abs(ground_truth[~future_mask] - forecast[~future_mask])

    # Вычисляем MAE для базовой модели
    # Сначала вычисляем MAE по базовой модели (например, случайные блуждания)
    # Используем значение сезонного среднего для шкалирования
    seasonal_data = actual[-seasonal_period - prediction_len:-seasonal_period]
    mae_naive = np.nanmean(np.abs(ground_truth[~future_mask] - seasonal_data[~future_mask]))

    # Общая MAE
    mae_forecast = np.mean(absolute_error)

    # Убедимся, что mae_naive не равно нулю для избегания деления на ноль
    if mae_naive == 0:
        raise ValueError("MAE для базовой модели не должен быть равен нулю.")

    # Считаем MASE
    mase = mae_forecast / mae_naive

    return mase


def calc_metrics(dataset, forecasts, bot):
    """
    Возвращет метрики MASE/sMAPE.

    :param dataset: тестовый датасет, содержащий все последовательности целиком
    :param forecasts: прогноз, полученный методом inference
    :param bot: тестовый датасет, содержащий все последовательности целиком
    """
    prediction_len = bot.get('prediction_len')
    mase_metric = load("evaluate-metric/mase")
    smape_metric = load("evaluate-metric/smape")

    target = np.array([item['target'] for item in list(dataset)])

    # возьмём медианное по батчам значение прогноза: (channels, n_batches, prediction_len) -> (channels, prediction_len)
    forecast_median = np.median(forecasts, axis=1)

    mase_metrics = []
    smape_metrics = []
    # по каждому временному ряду датасета
    for item_id in range(target.shape[0]):
        ts = target[item_id]
        training_data = ts[:-prediction_len]
        ground_truth = ts[-prediction_len:]
        # исключим NaN на входе (дни, когда не было данных) из расчёта метрик
        past_mask = np.isnan(training_data)
        future_mask = np.isnan(ground_truth)

        # if item_id == 0:
        #     print(f'forecast_median:\n{forecast_median[item_id]}')
        #     print(f'ground_truth:\n{ground_truth}')
        #     print(f'training_data (-52):\n{training_data[-52:-52+prediction_len]}')
        #     print(f'training_data (-104):\n{training_data[-104:-104+prediction_len]}')

        mase = mase_metric.compute(
            predictions=forecast_median[item_id][~future_mask],
            references=np.array(ground_truth)[~future_mask],
            training=np.array(training_data)[~past_mask],
            # TODO:
            #  - учесть тренд в расчёте метрики; странно, что скалирование MASE по всей поляне вычисляется.
            periodicity=MASE_METRIC_PERIODICITY)
        mase_metrics.append(mase["mase"])
        # print(f'mase: {mase["mase"]}')

        # mase = calculate_mase(ts, forecast_median[item_id], prediction_len, MASE_METRIC_PERIODICITY)
        # mase_metrics.append(mase)
        # print(f'mase: {mase}')

        smape = smape_metric.compute(
            predictions=forecast_median[item_id][~future_mask],
            references=np.array(ground_truth)[~future_mask],
        )
        smape_metrics.append(smape["smape"])

    logger.debug(f"MASE (средняя): {np.mean(mase_metrics)}")
    logger.debug(f"sMAPE (средняя): {np.mean(smape_metrics)}")

    return mase_metrics, smape_metrics


def save_forecast(forecast, forecast_start_date, update_forecast):
    """
    Сохраняет данные прогноза в БД.

    :param forecast: данные прогноза (n_channels, n_batches, periods)
    """
    assert forecast.shape[0] == 6, "Метод реализован для сохранения только сгруппированных по модальностям данных."
    channels, _, _ = get_channels_settings()
    # усредняем по батчам
    forecast = forecast.mean(axis=1)
    # формируем массив с колонкой индекса модальности и колонкой прогнозных данных
    ch_index = np.arange(forecast.shape[0], dtype=int).repeat(forecast.shape[1])
    forecast = np.vstack([ch_index, forecast.reshape((-1,)).round(0).astype('int')]).T
    n_weeks = int(np.unique(forecast[:, 0], return_counts=True)[1][0])

    df = pd.DataFrame(forecast, columns=['ch_index', 'amount'], dtype=int)
    df.reset_index(inplace=True, drop=True)
    df['contrast_enhancement'] = "'none'"
    df['version'] = "'forecast'"
    df['modality'] = df['ch_index'].apply(lambda i: f"'{channels[i]}'")

    df['year'] = [(forecast_start_date + timedelta(weeks=i % n_weeks)).year for i in range(len(df))]
    df['week'] = [(forecast_start_date + timedelta(weeks=i % n_weeks)).isocalendar()[1] for i in range(len(df))]

    df['uid'] = df.apply(lambda _: f"'{uuid.uuid4()}'", axis=1)
    df.drop(['ch_index'], inplace=True, axis=1)

    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    df['created_at'] = f"'{now}'"
    df['updated_at'] = f"'{now}'"

    print('Данные прогноза объёмов исследований на 5 недель по 6-ти модальностям:')
    show_df = df.pivot(index=['year', 'week'], columns=['modality'], values=['amount'])
    print(show_df)

    # expand_pandas_output()
    # print(df)
    if update_forecast:
        cond = df[['year', 'week']].drop_duplicates()
        where = "version = 'forecast' and (\n\t" + " or\n\t".join(
            [f'(year = {row["year"]} and week = {row["week"]})' for _, row in cond.iterrows()]
        ) + "\n\t)"
        unique = ['version', 'year', 'week', 'modality', 'contrast_enhancement']
        db = DB()
        # удаляем имеющиеся данные и вставляем новые
        db.delete('work_plan_summary', where)
        db.upsert(df, 'work_plan_summary', unique)
        db.close()
        print('Данные прогноза сохранены.')


class Bot:
    def __init__(self, hp, state=None, bot_id=None, shift=None, index=None):
        """
        Класс единичного бота.
        Бот может быть создан либо новый - для этого необходимо передать id, shift, index,
        либо восстанавливается из состояния - для этого нужно передать state.

        :param hp: инстанс гиперпараметров
        :param state: словарь состояния бота
        :param bot_id: ID бота, с которым он будет создан
        :param shift: индекс популяции, к которой относится бот
            идентификация бота
            определение первой обучающей смены
            колучество мутаций на определённой смене
        :param index: индекс бота в популяции:
            сохраняется на диск, выводится в обозначениии бота 00.00
            +по нему определяется первый обучаемый бот и уже обученные боты (< first_index)
            в self.mode == 'best' лучшие боты переиндексируются (в setup),
            индексация ботов осуществляется в populate
            в generate индексы задаются начиная со start_index
        """
        # папка, где хранится бот
        self.bots_folder = hp.get('bots_folder')
        # словарь значений гиперпараметров бота
        self.values = None
        # метрики бота
        self.metrics = None
        # оценка бота после обучения соответствующей модели
        self.score = None
        # хэш значений гиперпараметров для проверки уникальности бота
        self.hash = None
        # время, которое обучался бот
        self.train_time = 0

        # если передано состояние, восстанавливаем бота из него
        if state:
            self.from_state(state)
            self.namespace = self.get('namespace')
        # иначе - новый бот
        else:
            self.id = bot_id
            self.shift = shift
            self.index = index
            # пространство имён, к которому относится бот
            self.namespace = hp.get('namespace')

        # список ключей изменяемых параметров
        self.changeable = [k for k in hp.space.keys()]
        self.printable = self.changeable + hp.calculated
        # возможные временные лаги и фичи, заданные в гиперпараметрах
        self.lags_sequence_set = copy.deepcopy(hp.lags_sequence_set)
        self.time_features_set = copy.deepcopy(hp.time_features_set)
        # сокращённые названия параметров для печати
        self.cuts = ['bs', 'cr', 'ls', 'tf', 'emb', 'enc', 'dec', 'dm', 'nb']
        assert len(self.cuts) == len(self.printable)
        # имя метрики, по которой оценивается бот
        self.metric = 'sMAPE'
        # форматы вывода параметров для методов __str__, __repr__
        self.repr_formats = ['3d', '.2f', '1d', '1d', '1d', '1d', '1d', '3d', '3d']
        self.str_formats = ['', '.2f', '', '', '', '', '', '', '']

    def activate(self, values, bot_hash):
        """Инициализирует значения гиперпараметров бота и сохраняет хэш их значений"""
        self.values = copy.deepcopy(values)
        self.hash = bot_hash

    def calc_params(self, ts_len):
        """Рассчитывает количество батчей на эпоху так, чтобы все данные поместились в последовательности"""
        min_sequence_len = self.values['prediction_len'] * (1 + self.values['context_ratio'])
        n_sequencies_total = N_CHANNELS * (ts_len - min_sequence_len)
        self.values['num_batches_per_epoch'] = int(
            self.values['k_expand'] * n_sequencies_total / self.values['train_batch_size']
        )

    def get(self, key):
        """Возвращает значение гиперпараметра бота по его ключу"""
        if key in self.values:
            return self.values[key]
        else:
            raise KeyError(f'Не задан гиперпараметр с ключом: {key}.')

    def set(self, key, value):
        """Устанавливает значение гиперпараметра бота по его ключу"""
        if key in self.values:
            self.values[key] = value
        else:
            raise KeyError(f'У бота отсутствует гиперпараметр с ключом: {key}.')

    def set_result(self, metrics: dict, score, train_time):
        self.metrics = metrics
        self.score = score
        self.train_time = train_time

    def get_lags_sequence(self):
        index = self.get('lags_sequence_index')
        # return [i + 1 for i in range(self.get('prediction_len'))] + self.lags_sequence_set[index]
        return self.lags_sequence_set[index]

    def get_time_features(self):
        index = self.get('time_features_index')
        return self.time_features_set[index]

    def save(self):
        """Сохраняет состояние бота на диск"""
        fname = self.bots_folder + self.filename() + '.json'
        state = self.get_state()

        def show_vals(dct):
            for key, value in dct.items():
                print(key, value, type(value))
                if isinstance(value, dict):
                    show_vals(value)

        def convert(src_dict):
            for key, value in src_dict.items():
                if isinstance(value, np.int64):
                    src_dict[key] = int(value)
                if isinstance(value, np.float64):
                    src_dict[key] = float(value)
                if isinstance(value, dict):
                    convert(value)

        convert(state)
        # show_vals(state)

        with open(fname, "w") as fp:
            json.dump(state, fp)
        return fname

    def get_state(self):
        """Возвращает состояние бота для сохранения на диск"""
        excluded_values_keys = ['bots_folder', 'n_search', 'n_bots', 'n_survived', 'n_random']

        values = {key: self.values[key] for key in self.values if key not in excluded_values_keys}
        return {
            'id': self.id,
            'shift': self.shift,
            'index': self.index,
            'values': values,
            'metrics': self.metrics,
            'train_time': self.train_time,
            'score': self.score,
            'hash': self.hash,
        }

    def from_state(self, state):
        """Восстанавливает бота из словаря состояния"""
        for key, value in state.items():
            setattr(self, key, value)

    def filename(self):
        """Возвращает имя файла для бота"""
        return f'bot_{self.namespace}_{self.id:05d}'

    def __str__(self, formats=None):

        if formats is None:
            formats = self.str_formats
        params = ''
        delimiter = ''
        for i in range(len(self.cuts)):
            if formats[i] and self.values[self.printable[i]]:
                value = f'{self.values[self.printable[i]]:{formats[i]}}'
            else:
                value = f'{self.values[self.printable[i]]}'
            params += delimiter + f'{self.cuts[i]}: {value}'
            delimiter = ', '

        score = f'{self.score:.4f}' if self.score is not None else None
        shift = f'{self.shift:02d}' if self.shift is not None else None
        index = f'{self.index:02d}' if self.index is not None else None
        return (f'ID {self.id:3d} [{self.namespace}.{shift}.{index}], '
                f'⚙️[{params}], {self.metric}: {score}, '
                f't: {self.train_time:3.0f}s [{self.get("n_epochs")}]')

    def __repr__(self):
        return self.__str__(self.repr_formats)


def _rate_bots(bots: dict[Bot], n_bots: int = None):
    """
    Сортирует и возвращает словарь ботов в порядке убывания оценки.

    :param bots: словарь ботов по их ID
    :param n_bots: количество возвращаемых лучших ботов; если не задано, то все
    """
    if n_bots is None:
        n_bots = len(bots)
    # получаем пары (ID, оценка); боты без оценки окажутся в конце
    rating = [(bot_id, bot.score if bot.score else 1e9) for bot_id, bot in bots.items()]
    # сортируем по оценке и отбираем заданное количество лучших
    rating = sorted(rating, key=lambda t: t[1])[:n_bots]
    # возвращаем словарь лучших ботов
    return {rate[0]: bots[rate[0]] for rate in rating}


class Researcher:
    """
    Класс исследователя, который создаёт и испытывает популяции ботов
    с различными гиперпараметрами, включая параметры нарезки входных данных
    на последовательности.

    :param datamanager: менеджер данных, формирующий генератор датасета
    :param hp: гиперпараметры (инстанс HyperParameters);
    :param show_graphs: флаг вывода графиков в процессе обучения ботов;
    :param mode: режим работы:
        genetic - генерация и обучение ботов генетическим алгоритмом;
        test - режим тестирования, в котором создаётся
            только одна популяция с единственным ботом и гиперпараметрами
            по-умолчанию; выполняется всего  2 эпохи обучения;
        best - обучение лучших ботов, считанных с диска;
            остальные боты при этом удаляются из памяти.
    :param train: флаг выполнения обучения, иначе результаты обучения будут
        сгенерированы случайным образом (для быстрой проверки работы
        генетического алгоритма);
    :param save_bots: флаг сохранения ботов на диск;
    """

    def __init__(self, datamanager, hp,
                 mode='genetic',
                 show_graphs=False,
                 train=True, save_bots=True
                 ):
        self.datamanager = datamanager
        self.hp = hp
        self.show_graphs = show_graphs
        self.mode = mode
        self.train = train
        self.save = save_bots
        # папка проекта, в которую будут сохраняться данные
        self.bots_folder = hp.get('bots_folder')
        # подключаем/создаём папку
        # self._connect()

        # словарь для хранения всех ботов по их ID
        self.bots = {}
        # словарь для хранения лучших ботов по их ID
        self.best_bots = {}
        # словарь для хранения текущей популяции ботов по их ID
        self.population = {}
        # список для хранения хэшей ботов
        self.hashes = []
        # старший ID бота
        self.max_id = 0
        # индекс текущей популяции
        self.shift = None
        # инициализуем генератор случайных чисел
        self.gen = np.random.default_rng()

        # количество временных рядов
        global N_CHANNELS
        _, _, N_CHANNELS = get_channels_settings()

    def _setup(self):
        """
        Пробует считать ботов с диска и определяет режим старта генетического алгоритма.

        :return: * from_shift - индекс начальной смены популяции ботов;
                 * evolve - признак создания популяции ботов с помощью генетики.
        """
        # индекс начальной смены популяции ботов
        from_shift = 0
        # признак создания популяции ботов с помощью генетики
        evolve = False
        # количество смен популяций ботов
        n_search = 1

        def bots_refactor(bots):
            """Промежуточный метод для переформатирования старых ботов к изменениям"""
            for bot in bots.values():
                if 'decay_epochs' not in bot.values.keys():
                    bot.values['decay_epochs'] = bot.get('n_epochs') - bot.get('warmup_epochs')
            pass

        def set_from_hp(bots: dict[int: Bot], keys: list[str], unlearned: bool = False):
            """Устанавливает параметры ботов из значений гиперпараметров"""
            for bot in bots.values():
                for key in keys:
                    assert key in bot.values.keys(), f'Бот не содержит параметра с ключом: {key}'
                    assert key in self.hp.fixed, f'Гиперпараметры не содержат ключа: {key}'
                    if not unlearned or not bot.score:
                        bot.set(key, self.hp.get(key))

        if self.mode == 'genetic':
            # считываем ботов с диска
            self.bots, self.max_id = self.load_bots()
            # обновляем старых ботов, если это необходимо
            bots_refactor(self.bots)
            # устанавливаем ботам параметры из hp
            set_from_hp(self.bots, ['n_epochs', 'warmup_epochs', 'decay_epochs', 'end_shifts'], unlearned=True)
            # восстанавливаем хэши
            self.hashes = [bot.hash for bot in self.bots.values()]
            n_search = self.hp.get('n_search')
            # получаем индекс текущей популяции и индекс последнего бота в ней
            shifts = [bot.shift for bot in self.bots.values()]
            if shifts:
                # получаем индекс последней популяции ботов
                from_shift = max(shifts)

                # восстанавливаем последнюю популяцию (боты отсортированы по ID)
                self.population = {
                    bot.id: bot
                    for bot in self.bots.values()
                    if bot.shift == from_shift
                }
                # если считанная популяция неполная
                # n_extra_bots = self.hp.get('n_bots') - len(self.population)
                # if n_extra_bots > 0:
                #     self.population = self.population | self.generate(n_extra_bots)
                assert len(self.population) == self.hp.get('n_bots')
                # получаем индексы обученных ботов
                learned_indices = [
                    bot.index
                    for bot in self.population.values()
                    if bot.score
                ]
                # если последняя популяция обучена целиком
                if len(learned_indices) == self.hp.get('n_bots'):
                    from_shift += 1
                    evolve = True

            # если ботов не считано с диска, создаём новую популяцию
            else:
                self.shift = 0
                self.population = self.generate(self.hp.get('n_bots'))
                self.bots = self.bots | self.population
                # предсохраняем популяцию на диск
                self.save_bots(self.population)

        elif self.mode == 'test':
            # создаём единственного бота с параметрами по умолчанию
            self.shift = 0
            self.population = self.generate(1, mode='default')
            for bot in self.population.values():
                bot.set('n_epochs', 2)
            self.bots = self.population

        elif self.mode == 'single':
            # создаём единственного бота с параметрами по умолчанию
            self.shift = 0
            self.population = self.generate(1, mode='default')
            self.bots = self.population

        # если мы дообучаем лучших ботов
        elif self.mode == 'best':
            # считываем ботов с диска
            self.bots, _ = self.load_bots()
            # обновляем старых ботов, если это необходимо
            bots_refactor(self.bots)
            # создаём популяцию из лучших ботов
            self.population = self.get_best_bots()
            # установим новые параметры считанному боту
            set_from_hp(self.population, ['n_epochs', 'warmup_epochs', 'decay_epochs', 'end_shifts'])

            # меняем индексы ботов в популяции для корректного вывода
            for i, bot_id in enumerate(self.population):
                bot = self.population[bot_id]
                bot.index = i
                bot.score = None

            # остальных ботов удаляем, чтобы корректно рассчитывался рейтинг
            self.bots = self.population

        elif self.mode == 'update':
            # считываем всех ботов с диска и формируем из них популяцию
            self.bots, _ = self.load_bots()

            # TODO: доработать, если понадобится

            self.population = self.bots

        else:
            raise ValueError(f'Неизвестный режим обучения: {self.mode}')

        return from_shift, evolve, n_search

    def run(self, do_forecast=False, update_forecast=True):

        from_shift, evolve, n_search = self._setup()
        train_ds, test_ds = None, None
        forecast, config, time_features, freq = None, None, None, None
        model, device = None, None
        learning_rates = None
        activate_plotly()

        # цикл смены популяций ботов
        for shift in range(from_shift, n_search):
            self.shift = shift

            # если установлен флаг эволюционирования или смена не первая
            if evolve or shift > from_shift:
                # отбираем лучших и генерируем из них популяцию
                self.best_bots = self.get_best_bots()
                self.population = self.populate()
                # предсохраняем популяцию на диск
                # (после обучения каждый бот будет перезаписан вместе с метриками,
                #  полученными в процессе обучения)
                self.save_bots(self.population)
                # сбросим индекс первого бота для обучения, актуальный для первой итерации
                # first_index = 0

            if self.population is None:
                print('Не удалось создать популяцию. Обучение прекращено.')
                return

            # цикл по популяции ботов
            for bot_id, bot in self.population.items():
                # пропустим обученных ботов (актуально для первой смены популяций)
                if bot.score:
                    print(f'Бот уже обучен, пропущен: {bot}')
                    continue
                print(f'Популяция #{shift:02d}, bot ID: {bot.id} ({bot.index + 1}/{len(self.population.items())})')

                # получаем модель
                time_features = bot.get_time_features()
                freq = bot.get('freq')
                mb = ModelBuilder(bot, n_channels=N_CHANNELS,
                                  n_time_features=len(time_features))
                model, config = mb.get()
                # print(f'model distribution_output: {config.distribution_output}')

                forecasts = np.empty((N_CHANNELS, config.num_parallel_samples, 0))
                losses, mase_metrics, smape_metrics = [], [], []
                train_sec = 0

                # цикл по этапам обучения
                for stage_index, end_shift in enumerate(bot.get('end_shifts')):

                    # формируем выборки
                    train_ds = self.datamanager.from_generator(splits=2, split='train', end_shift=end_shift)
                    test_ds = self.datamanager.from_generator(splits=2, split='test', end_shift=end_shift)
                    ts_len = len(train_ds[0]['target'])
                    bot.calc_params(ts_len)
                    # print(f'ts_len: {ts_len}')
                    # print(f'num_batches_per_epoch: {bot.get("num_batches_per_epoch")}')

                    if self.train:
                        # формируем загрузчик данных
                        train_dataloader = create_train_dataloader(
                            config=config,
                            freq=freq,
                            data=train_ds,
                            batch_size=bot.get('train_batch_size'),
                            num_batches_per_epoch=bot.get('num_batches_per_epoch'),
                            time_features=time_features,
                        )
                        # обучаем модель
                        model, stage_losses, device, stage_train_sec, lrs \
                            = train_model(model, config, train_dataloader, bot, stage_index, len(bot.get("end_shifts")))
                        if stage_index == 0:
                            learning_rates = lrs

                        # формируем тестовую выборку и делаем инференс
                        test_dataloader = create_test_dataloader(
                            config=config,
                            freq=freq,
                            # подаём на инференс тренировочный датасет, чтобы получить
                            # последующую предсказанную последовательность
                            data=train_ds,
                            # data=test_ds,
                            batch_size=64,
                            time_features=time_features,
                        )
                        # forecast: (N_CHANNELS, num_parallel_samples=100, prediction_len),
                        forecast = inference(model, test_dataloader, config, device)
                        forecasts = np.concatenate([forecasts, forecast], axis=2)

                        # считаем и оцениваем метрики MASE/sMAPE
                        metrics = calc_metrics(test_ds, forecast, bot)
                        losses.append(stage_losses)
                        mase_metrics.append(metrics[0])
                        smape_metrics.append(metrics[1])
                        train_sec += stage_train_sec
                    else:  # TODO: доработать, если эта фича понадобится
                        # тестовые значения
                        stage_losses = self.gen.random(size=(bot.get('n_epochs'),
                                                             bot.get('num_batches_per_epoch'))).tolist()
                        # train_ds, test_ds, forecasts = ...
                        losses.append(stage_losses)
                        mase_metrics.append(self.gen.random(size=len(train_ds)).tolist())
                        smape_metrics.append(self.gen.random(size=len(train_ds)).tolist())

                # print(f'forecasts: {forecasts.shape}')
                # запоминаем результаты бота
                bot.set_result(
                    metrics=dict(losses=losses, mase=mase_metrics, smape=smape_metrics),
                    # оценка бота - среднее значение ошибки sMAPE по всем временным рядам
                    score=np.array(smape_metrics).mean(),
                    train_time=train_sec
                )

                # сохраняем бота на диск
                if self.save and self.mode != 'test':
                    bot.save()

                # выводим статистику бота
                if self.show_graphs:
                    # learning_rates - данные последней стадии обучения
                    dashboard(bot.metrics, test_ds, forecasts, learning_rates, bot, name=str(bot))
                # освобождаем память
                del train_dataloader, test_dataloader
                torch.cuda.empty_cache()
                gc.collect()

            # выводим рейтинг 10-ти лучших ботов на текущий цикл
            self.print_bots_rating(num=10)

        # сохраним последний полученный прогноз - применимо при обучении единственного бота
        if forecast is not None and do_forecast:
            # используем тестовый датасет для прогноза - он всегда содержит полные данные до даты прогноза
            test_dataloader = create_test_dataloader(
                config=config,
                freq=freq,
                data=test_ds,
                batch_size=64,
                time_features=time_features,
            )
            # forecast: (N_CHANNELS, num_parallel_samples=100, prediction_len),
            forecast = inference(model, test_dataloader, config, device)
            assert 'ROENTGEN.FORECAST_START_DATE' in os.environ, 'Дата начала прогноза не найдена в переменных среды.'
            forecast_start_date = os.environ['ROENTGEN.FORECAST_START_DATE']
            save_forecast(forecast, datetime.fromisoformat(forecast_start_date), update_forecast)

        # очистим кэш сохраннённых выборок, т.к. при следующем запуске они могут быть другими
        if train_ds:
            train_files = train_ds.cleanup_cache_files()
            test_files = test_ds.cleanup_cache_files()
            # кэш пустой
            # print(f'Кэш выборок очищен. Удалено файлов: {train_files} / {test_files}.')

    def generate(self, number, mode='random', start_index=0):
        """
        Генерирует заданное количество случайных ботов.

        :param number: количество генерируемых ботов
        :param mode: режим активации бота: default, random, data_nearest
        :param start_index: индекс первого бота в популяции (используется для пополнения популяции)
        :returns:
            Популяция ботов
        """
        population = {}
        print(f'Новые боты (режим {mode}):')
        for i in range(number):
            # создаём бота
            bot = self.create_bot(mode, start_index + i, keep_hash=True)
            if bot is None:
                return None
            # добавляем бота в популяцию
            population[self.max_id] = bot

        for bot in population.values():
            print(repr(bot))

        return population

    def create_bot(self, mode, index, values=None, keep_hash=True):
        """
        Создаёт нового бота.

        :param mode: режим активации бота: default, random, data_nearest
        :param index: индекс бота в популяции
        :param values: параметры бота
        :param keep_hash: Признак добавления хэша бота в общий список хэшей.
                False используется, когда необходимо инициализировать бота случайными
                значениями, а потом их переопределить.
        """
        # задаём ID для нового бота
        self.max_id += 1
        # создаём бота
        bot = Bot(self.hp, bot_id=self.max_id, shift=self.shift, index=index)
        # добавляем бота в общий словарь ботов
        self.bots[self.max_id] = bot

        # создаём дефолтного бота, если задано
        if mode == 'default':
            values, bot_hash = self.hp.generate(mode, self.hashes)
        else:
            # инициализуем значения гиперпараметров случайным образом
            values, bot_hash = self.hp.generate(mode, self.hashes, current_values=values)
            if values is None:
                return None
            if keep_hash:
                self.hashes.append(bot_hash)

        # задаём значения и их хэш боту
        bot.activate(values, bot_hash)

        return bot

    def get_best_bots(self) -> dict[Bot]:
        """Определяет и возвращает лучших ботов"""
        # количество отбираемых ботов равно количеству выживших
        best_bots = _rate_bots(self.bots, self.hp.get('n_survived'))

        print('Лучшие боты:')
        for bot in best_bots.values():
            print(repr(bot))
        return best_bots

    def populate(self):
        """Восполняет популяцию ботов из набора лучших.
        """
        population = {}
        # получаем список ID лучших ботов
        best_ids = list(self.best_bots.keys())
        # запоминаем количество выживших
        n_survived = self.hp.get('n_survived')

        # лучшие боты уже оценены, – произведём из них партию новых,
        # в которых изменим только параметры данных к ближайшим значениям
        print('Новые потомки с изменёнными данными:')
        # индекс бота в популяции
        index = 0
        for bot_id, best_bot in self.best_bots.items():
            bot = self.create_bot('data_nearest', index, values=best_bot.values)
            # если создать бота с новыми параметрами данных не удалось,
            # будет создан дополнительный бот путём скрещивания родителей
            if bot is None:
                print('Не удалось создать бота с новыми параметрами данных. Будет создан бот путём скрещивания.')
                n_survived -= 1
                continue
            # добавим бота в популяцию
            population[bot.id] = bot
            index += 1
            print(repr(bot))

        # произведём новых ботов путём скрещивания генов
        n_descendants = self.hp.get('n_bots') - n_survived - self.hp.get('n_random')
        print('Новые потомки путём скрещивания и мутации:')
        for _ in range(n_descendants):
            # создадим бота и инициализируем его случайным образом
            bot = self.create_bot('random', index, keep_hash=False)
            # получаем ID двух уникальных лучших родителей
            parent_ids = self.gen.choice(best_ids, size=2)
            # получаем имена параметров для мутации
            mut_names = self._mutation()
            # print(f'Параметры для мутации: {mut_names}')
            values = copy.deepcopy(bot.values)
            for key in bot.changeable:
                # для мутирующих параметров оставим случайные значения,
                # остальные перезапишем от родителей
                if key not in mut_names:
                    # берём случайное значение у одного из родителей
                    values[key] = self.bots[
                        self.gen.choice(parent_ids)
                    ].values[key]

            # установим хэш для новых значений
            bot_hash = self.hp.get_hash(values)
            self.hashes.append(bot_hash)
            # установим боту новые значения и хэш
            bot.activate(values, bot_hash)
            index += 1
            print(repr(bot))

            # добавим бота в популяцию
            population[bot.id] = bot

        # добавим в популяцию заданное количество случайных ботов
        population = population | self.generate(self.hp.get('n_random'),
                                                start_index=index)
        return population

    def _mutation(self):
        """Возвращает список имён параметров для случайной мутации"""
        # начнём с 3-х и будем уменьшать на 1 каждые две смены популяции до 1-го
        n_mutations = max(3 - self.shift // 2, 1)
        return list(self.gen.choice(list(self.hp.space.keys()), size=n_mutations))

    def save_bots(self, bots):
        """Сохраняет всех ботов на диск"""
        if self.save:
            for bot in bots.values():
                bot.save()

    def load_bots(self):
        namespace = self.hp.get('namespace')
        bots = {}
        max_id = 0
        if self.mode == 'test':
            return bots, max_id
        filelist = glob.glob(self.filepath())
        for fname in filelist:
            with open(fname, "r") as fp:
                state = json.load(fp)
                bot = Bot(self.hp, state=state)
                bots[bot.id] = bot
                if bot.id > max_id:
                    max_id = bot.id
        if len(filelist) > 0:
            print(f'Считано ботов с диска в пространстве имён "{namespace}": {len(filelist)}')
            bots = _rate_bots(bots)
            if self.mode != 'best':
                for bot in bots.values():
                    print(repr(bot))

        else:
            print(f'Ботов на диске в пространстве имён "{namespace}" не найдено.')

        # возвращаем словарь ботов, отсортированный по ID, и ID старшего бота
        if len(bots) > 0:
            bots = dict(sorted(bots.items()))
            assert list(bots.keys())[-1] == max_id
        return bots, max_id

    def filepath(self):
        """Возвращает шаблон пути для считывания ботов с диска"""
        return f'{self.bots_folder}bot_{self.hp.get("namespace")}_*.j*'

    def print_bots_rating(self, num=10):
        print('Рейтинг лучших ботов:')
        best_bots = _rate_bots(self.bots, num)
        for bot in best_bots.values():
            print(repr(bot))
