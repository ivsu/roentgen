import os

from workplan.hyperparameters import Hyperparameters
from workplan.datamanager import DataManager
from workplan.genetic import Researcher
from workplan.show import indicators
from common.logger import Logger

logger = Logger(__name__)


def search_hyperparameters(mode='genetic', end_shifts=None):

    # создаём инстанс гиперпараметров
    hp = Hyperparameters()
    if end_shifts is not None:
        hp.fixed['end_shifts'] = end_shifts

    # (!) plotly странно работает при первом вызове в колабе - выведем графические
    # индикаторы для первого вызова
    if os.environ['RUN_ENV'] == 'COLAB':
        indicators(
            hp,
            params=['n_epochs', 'warmup_epochs', 'decay_epochs', 'prediction_len'],
            titles=['Эпох обучения', 'Эпох прогрева', 'Эпох затухания LR', 'Глубина предикта']
        )

    # создаём менеджер данных и готовим исходный DataFrame для формирования выборок
    datamanager = DataManager()
    datamanager.read_and_prepare(
        freq=hp.get('freq'),
        prediction_len=hp.get('prediction_len')
    )

    # создаём Researcher и передаём ему датасеты и инстанс гиперпараметров
    researcher = Researcher(datamanager, hp,
                            mode=mode,
                            show_graphs=True, total_periods=7,
                            train=True, save_bots=True)
    researcher.run()


def learn_best_bots():

    # создаём инстанс гиперпараметров
    hp = Hyperparameters()
    # установим большее, чем при поиске количество эпох обучения, и возьмём 7 лучших ботов
    hp.set('n_epochs', 30)
    hp.set('n_survived', 7)
    hp.set('warmup_epochs', 5)
    hp.set('decay_epochs', 20)
    hp.set('end_shifts', [0])

    # (!) plotly странно работает при первом вызове в колабе - выведем графические
    # индикаторы для первого вызова
    if os.environ['RUN_ENV'] == 'COLAB':
        indicators(
            hp,
            params=['n_epochs', 'warmup_epochs', 'decay_epochs', 'prediction_len'],
            titles=['Эпох обучения', 'Эпох прогрева', 'Эпох затухания LR', 'Глубина предикта']
        )

    # создаём менеджер данных и готовим исходный DataFrame для формирования выборок
    datamanager = DataManager()
    datamanager.read_and_prepare(
        freq=hp.get('freq'),
        prediction_len=hp.get('prediction_len')
    )

    # создаём Researcher и передаём ему датасеты и инстанс гиперпараметров
    researcher = Researcher(datamanager, hp,
                            mode='best',
                            show_graphs=True, total_periods=2,
                            train=True, save_bots=False)
    researcher.run()


if __name__ == '__main__':

    logger.setup(level=logger.INFO, layout='debug')

    search_hyperparameters(
        # mode='test',
        mode='genetic', end_shifts=[0]
    )
    # learn_best_bots()
