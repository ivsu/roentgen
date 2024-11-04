import os

from workplan.hyperparameters import Hyperparameters
from workplan.datamanager import DataManager
from workplan.genetic import Researcher
from workplan.show import indicators
from common.logger import Logger

logger = Logger(__name__)

PARAMS = dict(n_search='Популяций', n_bots='Ботов в популяции',
              n_survived='Количество выживающих', n_random='Количество рандомных', prediction_len='Глубина предикта',
              n_epochs='Эпох обучения', warmup_epochs='Эпох прогрева', decay_epochs='Эпох затухания LR',
              end_shifts='Этапов обучения каждого бота')


def learn_default(namespace, data_version, do_forecast):
    """Обучает бота с параметрами по умолчанию и сохраняет его в заданном пространстве имён"""

    end_shifts = [-15, -10, -5, 0]
    # end_shifts = [0]

    # создаём инстанс гиперпараметров
    hp = Hyperparameters()
    # установим параметры для единичного обучения
    hp.set('namespace', namespace)
    hp.set('prediction_len', 5)
    hp.set('n_search', 1)
    hp.set('n_bots', 1)
    hp.set('n_survived', 0)
    hp.set('n_random', 0)
    hp.set('n_epochs', 40)
    hp.set('warmup_epochs', 5)
    hp.set('decay_epochs', 12)
    hp.set('k_expand', 2.0)
    hp.set('initial_lr', 5e-5)
    hp.set('target_lr', 1e-3)
    hp.set('final_lr', 1e-4)
    hp.set('end_shifts', end_shifts)
    # для debug
    if data_version == 'debug':
        # hp.set('prediction_len', 5)
        hp.set('n_epochs', 40)
        # hp.set('k_expand', 2.0)
        # hp.set('initial_lr', 1e-5)
        # hp.set('target_lr', 1e-3)
        hp.set('final_lr', 1e-4)
        pass

    # сгенерируем набор дефолтных гиперпараметров и посмотрим на их значения
    values, _ = hp.generate(mode='default', hashes=[])
    print(hp.repr(values, mode='short'))

    # (!) plotly странно работает при первом вызове в колабе - выведем графические
    # индикаторы для первого вызова
    # if True or os.environ['RUN_ENV'] == 'COLAB':
    #     indicators(hp, PARAMS)

    # создаём менеджер данных и готовим исходный DataFrame для формирования выборок
    datamanager = DataManager()
    datamanager.read_and_prepare(
        freq=hp.get('freq'),
        prediction_len=hp.get('prediction_len'),
        data_version=data_version
    )

    # создаём Researcher и передаём ему датасеты и инстанс гиперпараметров
    researcher = Researcher(datamanager, hp,
                            mode='single',
                            show_graphs=True,
                            train=True, save_bots=True)
    researcher.run(do_forecast)


def search_hyperparameters(mode='genetic', end_shifts=None):

    # создаём инстанс гиперпараметров
    hp = Hyperparameters()
    # debug
    # hp.set('n_epochs', 3)

    if end_shifts is not None:
        hp.fixed['end_shifts'] = end_shifts

    # (!) plotly странно работает при первом вызове в колабе - выведем графические
    # индикаторы для первого вызова
    if True or os.environ['RUN_ENV'] == 'COLAB':
        indicators(hp, PARAMS)

    # создаём менеджер данных и готовим исходный DataFrame для формирования выборок
    datamanager = DataManager()
    datamanager.read_and_prepare(
        freq=hp.get('freq'),
        prediction_len=hp.get('prediction_len'),
        data_version = 'train'
    )

    # создаём Researcher и передаём ему датасеты и инстанс гиперпараметров
    researcher = Researcher(datamanager, hp,
                            mode=mode,
                            show_graphs=True,
                            train=True, save_bots=True)
    researcher.run()


def learn_best_bots(n_bots, namespace, end_shifts=None, do_forecast=False):

    # создаём инстанс гиперпараметров
    hp = Hyperparameters()
    hp.set('namespace', namespace)
    # установим большее, чем при поиске количество эпох обучения, и возьмём 7 лучших ботов
    hp.set('n_epochs', 7)
    hp.set('n_survived', n_bots)
    hp.set('warmup_epochs', 5)
    hp.set('decay_epochs', 20)
    hp.set('end_shifts', end_shifts if end_shifts else [0])

    # (!) plotly странно работает при первом вызове в колабе - выведем графические
    # индикаторы для первого вызова
    if os.environ['RUN_ENV'] == 'COLAB':
        indicators(hp, PARAMS)

    # создаём менеджер данных и готовим исходный DataFrame для формирования выборок
    datamanager = DataManager()
    datamanager.read_and_prepare(
        freq=hp.get('freq'),
        prediction_len=hp.get('prediction_len'),
        data_version='train'
    )

    # создаём Researcher и передаём ему датасеты и инстанс гиперпараметров
    researcher = Researcher(datamanager, hp,
                            mode='best',
                            show_graphs=True,
                            train=True, save_bots=False)
    researcher.run(do_forecast)


if __name__ == '__main__':

    logger.setup(level=logger.INFO, layout='debug')

    # установим количество каналов данных
    os.environ['ROENTGEN.N_CHANNELS'] = str(6)

    # установим дату начала прогноза (датасет будет урезан до неё)
    os.environ['ROENTGEN.FORECAST_START_DATE'] = '2024-04-29'

    learn_default(namespace='99', data_version='train', do_forecast=False)  # source, train, debug
    # search_hyperparameters(
    #     # mode='test',
    #     mode='genetic', end_shifts=[-5, 0]
    # )
    # learn_best_bots(n_bots=7, namespace='02')
