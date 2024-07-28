import json
import os
import glob
import gc
import random
import time
import numpy as np

# для обучения модели
from accelerate import Accelerator
from torch.optim import AdamW
import torch
# для расчёта метрик MASE/sMAPE
from evaluate import load
from gluonts.time_feature import get_seasonality

from modelbuilder import ModelBuilder
from show import Show
from dataloaders import create_train_dataloader, create_test_dataloader
from schedulers import WarmAndDecayScheduler
from common.logger import Logger

logger = Logger(__name__)


def train(model, config, dataloader, hp, mode='genetic'):
    epochs = 2 if mode == 'test' else hp.get('n_epochs')
    warmup_epochs = hp.get('warmup_epochs')
    steps = 0

    # определим количество батчей
    num_batches = 0
    for _ in iter(dataloader):
        num_batches += 1

    # используем Accelerator от HuggingFace
    accelerator = Accelerator()
    device = accelerator.device
    model.to(device)

    # создаём оптимизатор Adam с дополнительным методом затухания весов
    # и передаём ему параметры модели
    optimizer = AdamW(model.parameters(), betas=(0.9, 0.95), weight_decay=1e-1)

    # будем обучать трансформер с прогревом
    scheduler = WarmAndDecayScheduler(
        optimizer,
        warmup_steps=num_batches * warmup_epochs,
        decay_steps=num_batches * (epochs - warmup_epochs),
        decay_rate=0.01,
        target_learning_rate=1e-3,
        initial_learning_rate=1e-5,
        device=device
    )

    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )

    model.train()
    losses = []
    start_time = time.time()
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
            # запоминаем текущий LR
            lr = optimizer.param_groups[0]['lr']

            # обратное распространение
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()

        losses.append(epoch_losses)
        mean_loss = sum(epoch_losses) / len(epoch_losses)
        # scheduler.step(mean_loss)
        t = time.time() - start_time

        end = chr(10) if epoch == epochs - 1 else ''
        print('\r'
              f'epoch: {epoch:3d}'
              f' | mean loss: {mean_loss:.4f}'
              f' | LR: {lr:.6f}'
              f' | steps: {steps:4d}'
              f' | total time: {t:3.0f}s',
              end=end)

    return model, losses, device


def inference(model, dataloader, config, device):
    """
    Реализует инференс модели с помощью метода generate.
    Возвращает вероятностый прогнозный вектор формы:
    (количество_временных_рядов, батчей_в_эпохе, глубина_предикта),
    Например: (4, 100, 30)
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


def metrics(dataset, forecasts, bot):
    """
    Возвращет метрики MASE/sMAPE.

    :param dataset: тестовый датасет, содержащий все последовательности целиком
    :param forecasts: прогноз, полученный методом inference
    :param bot: тестовый датасет, содержащий все последовательности целиком
    """
    prediction_len = bot.get('prediction_len')
    freq = bot.get('freq')
    mase_metric = load("evaluate-metric/mase")
    smape_metric = load("evaluate-metric/smape")

    # возмём медианное по батчам значение прогноза: (channels, n_batches, prediction_len) -> (channels, prediction_len)
    forecast_median = np.median(forecasts, axis=1)

    mase_metrics = []
    smape_metrics = []
    # по каждому временному ряду датасета
    for item_id, ts in enumerate(dataset):
        training_data = ts["target"][:-prediction_len]
        ground_truth = ts["target"][-prediction_len:]
        # исключим NaN на входе (дни, когда торгов не было) из расчёта метрик
        past_mask = np.isnan(training_data)
        future_mask = np.isnan(ground_truth)

        mase = mase_metric.compute(
            predictions=forecast_median[item_id][~future_mask],
            references=np.array(ground_truth)[~future_mask],
            training=np.array(training_data)[~past_mask],
            periodicity=get_seasonality(freq))
        mase_metrics.append(mase["mase"])

        smape = smape_metric.compute(
            predictions=forecast_median[item_id][~future_mask],
            references=np.array(ground_truth)[~future_mask],
        )
        smape_metrics.append(smape["smape"])

    logger.debug(f"MASE (средняя): {np.mean(mase_metrics)}")
    logger.debug(f"sMAPE (средняя): {np.mean(smape_metrics)}")

    # посмотрим на графики метрик MASE/sMAPE
    # Show.metrics(mase_metrics, smape_metrics, pds.CHANNEL_NAMES)

    return mase_metrics, smape_metrics


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
        :param index: индекс бота в популяции
        """
        # папка, где хранится бот
        self.bots_folder = hp.get('bots_folder')
        # если передано состояние, восстанавливаем бота из него
        if state:
            self.from_state(state)
        # иначе - новый бот
        else:
            self.id = bot_id
            self.shift = shift
            self.index = index
            # пространство имён, к которому относится бот
            self.namespace = hp.get('namespace')
            # словарь значений гиперпараметров бота
            self.values = None
            # метрики бота
            self.metrics = None
            # оценка бота после обучения соответствующей модели
            self.score = None
            # хэш значений гиперпараметров для проверки уникальности бота
            self.hash = None

        # список ключей изменяемых параметров
        self.changeable = [k for k in hp.space.keys()]
        # возможные временные лаги, заданные в гиперпараметрах
        self.lags_sequencies = hp.lags_sequencies.copy()
        # сокращённые названия параметров для печати
        self.cuts = ['bs', 'nb', 'cr', 'ls', 'tff', 'emb', 'enc', 'dec', 'dm']
        assert len(self.cuts) == len(self.changeable)
        # имя метрики, по которой оценивается бот
        self.metric = 'sMAPE'
        # форматы вывода параметров для методов __str__, __repr__
        self.repr_formats = ['3d', '3d', '.1f', '1d', '1d', '1d', '1d', '1d', '3d']
        self.str_formats = ['', '', '.1f', '', '', '', '', '', '']

    def activate(self, values, bot_hash):
        """Инициализирует значения гиперпараметров бота и сохраняет хэш их значений"""
        self.values = values.copy()
        self.hash = bot_hash

    def get(self, key):
        """Возвращает значение гиперпараметра бота по его ключу"""
        if key in self.values:
            return self.values[key]
        else:
            raise KeyError(f'Не задан гиперпараметр с ключом: {key}.')

    def get_lags_sequence(self):
        index = self.get('lags_sequence')
        return self.lags_sequencies[index]

    def save(self):
        """Сохраняет состояние бота на диск"""
        fname = self.bots_folder + self.filename() + '.jd'
        state = self.get_state()

        def show_vals(dct):
            for key, value in dct.items():
                # if isinstance(value, int):
                #     print(key, value)
                print(key, value, type(value))
                if isinstance(value, dict):
                    show_vals(value)

        def convert(src_dict):
            # TODO: разобраться и сделать, чтобы np.int64 и np.float64 не попадали в данные бота,
            #       иначе не сериализуется
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
        return {
            'id': self.id,
            'shift': self.shift,
            'index': self.index,
            'namespace': self.namespace,
            'values': self.values,
            'metrics': self.metrics,
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
            if formats[i]:
                value = f'{self.values[self.changeable[i]]:{formats[i]}}'
            else:
                value = f'{self.values[self.changeable[i]]}'
            params += delimiter + f'{self.cuts[i]}: {value}'
            delimiter = ', '

        score = f'{self.score:.4f} ({self.metric})' if self.score is not None else None
        shift = f'{self.shift:02d}' if self.shift is not None else None
        index = f'{self.index:02d}' if self.index is not None else None
        return (f'ID {self.id:3d} [{self.namespace}.{shift}.{index}]'
                f', values: [{params}], score: {score}')

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
    rating = [(bot_id, bot.score if bot.score else -1.) for bot_id, bot in bots.items()]
    # сортируем по оценке и отбираем заданное количество лучших
    rating = sorted(rating, key=lambda t: t[1])[:n_bots]
    # возвращаем словарь лучших ботов
    return {rate[0]: bots[rate[0]] for rate in rating}


class Researcher:
    """
    Класс исследователя, который создаёт и испытывает популяции ботов
    с различными гиперпараметрами, включая параметры нарезки входных данных
    на последовательности.

    :param train_ds: обучающий датасет (HuggingFace Dataset);
    :param test_ds: тестовый датасет (HuggingFace Dataset);
    :param hp: гиперпараметры (инстанс HyperParameters);
    :param channel_names: список имён каналов данных;
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

    def __init__(self, train_ds, test_ds, hp,
                 channel_names,
                 mode='genetic',
                 show_graphs=False,
                 train=True, save_bots=True
                 ):
        self.train_ds = train_ds
        self.test_ds = test_ds
        self.hp = hp
        self.channel_names = channel_names
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
        # индекс первого обучаемого бота в смене (актуально для возобновления обучения)
        self.first_index = None
        # инициализуем генератор случайных чисел системным временем
        random.seed()

    def run(self):

        self.first_index = 0
        # индекс начальной смены популяции ботов
        from_shift = 0
        # признак создания популяции ботов с помощью генетики
        evolve = False

        if self.mode == 'genetic':
            # считываем ботов с диска
            self.bots, self.max_id = self.load_bots()
            # восстанавливаем хэши
            self.hashes = [bot.hash for bot in self.bots.values()]
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
                # пробуем получить индекс следующего бота для обучения
                if learned_indices:
                    self.first_index = max(learned_indices) + 1
                # если последняя популяция обучена целиком
                if self.first_index == self.hp.get('n_bots'):
                    from_shift += 1
                    self.first_index = 0
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
            self.bots = self.population

        elif self.mode == 'best':
            # считываем ботов с диска
            self.bots, _ = self.load_bots()
            # создаём популяцию из лучших ботов
            self.population = self.get_best_bots()
            # меняем индексы ботов в популяции для корректного вывода
            for i, bot_id in enumerate(self.population):
                self.population[bot_id].index = i
            # остальных ботов удаляем, чтобы корректно рассчитывался рейтинг
            self.bots = self.population

        else:
            raise ValueError(f'Неизвестный режим обучения: {self.mode}')

        n_search = self.hp.get('n_search') if self.mode == 'genetic' else 1

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
                self.first_index = 0

            if self.population is None:
                print('Не удалось создать популяцию. Обучение прекращено.')
                return

            # цикл по популяции ботов
            for bot_id, bot in self.population.items():
                # пропустим обученных ботов (актуально для первой смены популяций)
                if bot.index < self.first_index:
                    print(f'Бот уже обучен, пропущен: {bot}')
                    continue
                print(f'Популяция #{shift:02d}, bot ID: {bot.id} ({bot.index + 1}/{len(self.population.items())})')

                # если мы дообучаем лучших ботов, установим им другие параметры
                if self.mode == 'best':
                    bot.values['n_epochs'] = self.hp.get('n_epochs')

                # получаем модель
                mb = ModelBuilder(self.train_ds, bot)
                model, config = mb.get()
                time_features = mb.get_time_features()
                # print(f'model distribution_output: {config.distribution_output}')

                if self.train:
                    # формируем загрузчик данных
                    train_dataloader = create_train_dataloader(
                        config=config,
                        freq=bot.get('freq'),
                        data=self.train_ds,
                        batch_size=bot.get('train_batch_size'),
                        num_batches_per_epoch=bot.get('num_batches_per_epoch'),
                        time_features=time_features,
                    )
                    # обучаем модель
                    model, losses, device = train(model, config, train_dataloader, bot, self.mode)

                    # формируем тестовую выборку и делаем инференс
                    # print('Перед формированием тестового даталоадера')
                    test_dataloader = create_test_dataloader(
                        config=config,
                        freq=bot.get('freq'),
                        data=self.test_ds,
                        batch_size=64,
                        time_features=time_features,
                    )
                    # print('После формирования тестового даталоадера')
                    forecasts = inference(model, test_dataloader, config, device)
                    # print('После инференса')

                    # считаем и оцениваем метрики MASE/sMAPE
                    mase_metrics, smape_metrics = metrics(self.test_ds, forecasts, bot)
                    # print('После расчёта метрик')
                else:
                    # тестовые значения
                    losses = np.random.rand(
                        bot.get('n_epochs'),
                        bot.get('num_batches_per_epoch')
                    ).tolist()
                    # forecasts = ...
                    mase_metrics = np.random.rand(len(self.train_ds)).tolist()
                    smape_metrics = np.random.rand(len(self.train_ds)).tolist()

                # запоминаем метрики бота
                bot.metrics = {
                    'losses': losses,
                    'mase': mase_metrics,
                    'smape': smape_metrics,
                }
                # оценка бота - среднее значение ошибки sMAPE по всем временным рядам
                bot.score = np.array(smape_metrics).mean()

                # сохраняем бота на диск
                if self.save:
                    bot.save()

                # выводим статистику бота
                if self.show_graphs:
                    Show.dashboard(bot.metrics, self.test_ds, forecasts, bot,
                                   self.channel_names,
                                   total_periods=4,
                                   name=str(bot)
                                   )
                # освобождаем память
                del model, train_dataloader, test_dataloader
                torch.cuda.empty_cache()
                gc.collect()

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
            values, bot_hash = self.hp.generate(mode, self.hashes,
                                                current_values=values)
            if values is None:
                return None
            if keep_hash:
                self.hashes.append(bot_hash)

        # задаём значения и их хэш боту
        bot.activate(values, bot_hash)

        return bot

    def get_best_bots(self):
        """Определяет и возвращает словарь лучших ботов
        """
        # количество отбираемых ботов равно количеству выживших
        best_bots = _rate_bots(self.bots, self.hp.get('n_survived'))

        # # получаем пары (ID, оценка)
        # scores = [(bot_id, bot.score) for bot_id, bot in self.bots.items()]
        # # из отсортированного списка берём заданное количество лучших пар (ID, оценка)
        # best_pairs = sorted(scores, key=lambda t: t[1])[:number]
        # # возвращаем словарь лучших ботов
        # best_bots = {t[0]: self.bots[t[0]] for t in best_pairs}
        logger.info('Лучшие боты:')
        for bot in best_bots.values():
            logger.info(repr(bot))
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
            parent_ids = random.sample(best_ids, k=2)
            # получаем имена параметров для мутации
            mut_names = self._mutation()
            # print(f'Параметры для мутации: {mut_names}')
            values = bot.values.copy()
            for key in bot.changeable:
                # для мутирующих параметров оставим случайные значения,
                # остальные перезапишем от родителей
                if key not in mut_names:
                    # берём случайное значение у одного из родителей
                    values[key] = self.bots[
                        random.choice(parent_ids)
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
        return random.sample(list(self.hp.space.keys()), k=n_mutations)

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
            for bot in bots.values():
                print(repr(bot))

        else:
            print(f'Ботов на диске в пространстве имён "{namespace}" не найдено.')

        # возвращаем словарь ботов, отсортированный по ID, и ID старшего бота
        if len(bots) > 0:
            bots = dict(sorted(bots.items()))
            assert list(bots.keys())[-1] == max_id
        return bots, max_id

    # def _connect(self):
    #     """Определяет папку для хранения/чтения ботов"""
    #     # если папка на гугл-диске
    #     if 'MyDrive' in self.bots_folder:
    #         if not os.path.exists(self.bots_folder):
    #             drive.mount('/content/drive/')
    #     elif not os.path.exists(self.bots_folder):
    #         os.mkdir(self.bots_folder)

    def filepath(self):
        """Возвращает шаблон пути для считывания ботов с диска"""
        return f'{self.bots_folder}bot_{self.hp.get("namespace")}_*.jd'
