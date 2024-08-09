import os

from workplan.hyperparameters import Hyperparameters
from workplan.datamanager import DataManager
from workplan.genetic import Researcher
from workplan.show import indicators
from common.logger import Logger

logger = Logger(__name__)


def search_hyperparameters(mode='genetic'):

    # создаём инстанс гиперпараметров
    hp = Hyperparameters()

    # (!) plotly странно работает при первом вызове в колабе в цикле - выведем графические
    # индикаторы и удалим инстанс
    if os.environ['RUN_ENV'] == 'COLAB':
        indicators(
            hp,
            params=['n_epochs', 'warmup_epochs', 'prediction_len'],
            titles=['Эпох обучения', 'Эпох прогрева', 'Глубина предикта']
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
                            show_graphs=True,
                            train=True, save_bots=True)
    researcher.run()


def learn_best_bots():

    # создаём инстанс гиперпараметров
    hp = Hyperparameters()
    # установим большее, чем при поиске количество эпох обучения, и возьмём 7 лучших ботов
    hp.fixed['n_epochs'] = 30
    hp.fixed['n_survived'] = 7

    # (!) plotly странно работает при первом вызове в колабе в цикле - выведем графические
    # индикаторы и удалим инстанс
    if os.environ['RUN_ENV'] == 'COLAB':
        indicators(
            hp,
            params=['n_epochs', 'warmup_epochs', 'prediction_len'],
            titles=['Эпох обучения', 'Эпох прогрева', 'Глубина предикта']
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
                            show_graphs=True,
                            train=True, save_bots=False)
    researcher.run()


if __name__ == '__main__':

    logger.setup(level=logger.INFO, layout='debug')

    # search_hyperparameters(
    #     # mode='test',
    #     mode='genetic',
    # )
    learn_best_bots()
