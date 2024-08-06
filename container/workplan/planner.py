import os

from workplan.hyperparameters import Hyperparameters
from workplan.datamanager import DataManager
from workplan.genetic import Researcher
from workplan.show import indicators
from common.logger import Logger

logger = Logger(__name__)


def planner():

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
                            # mode='genetic',
                            mode='test',
                            show_graphs=True,
                            train=True, save_bots=True)
    researcher.run()


if __name__ == '__main__':

    logger.setup(level=logger.INFO, layout='debug')

    planner()
