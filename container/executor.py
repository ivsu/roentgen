import os

from workplan.hyperparameters import Hyperparameters
from workplan.datamanager import DataManager
from workplan.genetic import Researcher
from workplan.show import indicators
from workplan.planner import learn_best_bots
from common.logger import Logger

logger = Logger(__name__)


def predict_and_calc_schedule():
    """
    Реализует рабочий цикл:
    - обучение модели с наилучшим набором гиперпараметров;
    - прогнозирование объёма исследований и сохранение прогноза в БД;
    - расчёт графика работы врачей на основе прогноза и сохранение его в БД.

    :param start_date: дата начала прогноза
    """
    # запускаем обучение одного (лучшего в пространстве имён) бота
    learn_best_bots(n_bots=1, namespace='99')


if __name__ == '__main__':

    logger.setup(level=logger.INFO, layout='debug')

    # установим дату начала прогноза
    os.environ['ROENTGEN.FORECAST_START_DATE'] = '2024-04-29'

    predict_and_calc_schedule()
