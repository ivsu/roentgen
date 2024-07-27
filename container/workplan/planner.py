import os

from hyperparameters import Hyperparameters
from datamanager import DataManager, CHANNEL_NAMES
from genetic import Researcher
from show import Show
from common.logger import Logger

logger = Logger(__name__)


if __name__ == '__main__':

    logger.setup(level=logger.INFO, layout='debug')

    # создаём инстанс гиперпараметров
    hp = Hyperparameters()

    # (!) plotly странно работает при первом вызове в колабе в цикле - выведем графические
    # индикаторы и удалим инстанс
    if os.environ['RUN_ENV'] == 'COLAB':
        Show.indicators(
            hp,
            params=['n_epochs', 'warmup_epochs', 'prediction_len'],
            titles=['Эпох обучения', 'Эпох прогрева', 'Глубина предикта']
        )

    # создаём менеджер данных и готовим исходный DataFrame для формирования выборок
    dm = DataManager()
    dm.prepare(
        freq=hp.get('freq'),
        prediction_len=hp.get('prediction_len')
    )
    # формируем выборки
    train_ds = dm.from_generator(splits=2, split='train')
    test_ds = dm.from_generator(splits=2, split='test')

    # создаём Researcher и передаём ему датасеты и инстанс гиперпараметров
    researcher = Researcher(train_ds, test_ds, hp,
                            CHANNEL_NAMES,
                            mode='genetic',
                            show_graphs=True,
                            train=True, save_bots=True)
    researcher.run()
