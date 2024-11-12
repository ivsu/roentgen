import os

from workplan.hyperparameters import Hyperparameters
from workplan.datamanager import DataManager
from workplan.genetic import Researcher
from workplan.show import indicators
from workplan.planner import learn_best_bots
# from schedule.show_schedule import get_formatted_schedule
from common.logger import Logger

logger = Logger(__name__)

"""
    TODO: полный цикл для целей диплома не нужен
    Библиотека для выполнения полного цикла прогнозирования и расчёта графика с сохранением его в БД.
    
    Алгоритм прогноза и расчёта графика:
    + сохраняем единичного бота на диск (также будет работать и при подборе гиперпараметров)
    + считываем и обучаем одного лучшего бота с параметром save_forecast=True
    + разобраться, как сформировать датасет - типа, нулями его заполнить для future_values
    + формируем номера недель для прогноза, удаляем имеющиеся за этот период данные из БД и пишем прогнозные
      с версией forecast
    - в Scheduler:
      - настроить понятный вывод результатов поиска;
      + прикрутить вывод графика в plotly
      + настрить сохранение в SQLite
      + считываем и выводим в виде plotly table
      - сделать запуск
    - запускаем scheduler
"""


def predict_and_calc_schedule():
    """
    Реализует рабочий цикл:
    - обучение модели с наилучшим набором гиперпараметров;
    - прогнозирование объёма исследований и сохранение прогноза в БД;
    - расчёт графика работы врачей на основе прогноза и сохранение его в БД.
    """
    # запускаем обучение одного (лучшего в пространстве имён) бота
    # после обучения, бот сохраняет полученных прогноз объёмов работ в БД в таблицу work_plan_summary
    # learn_best_bots(n_bots=1, namespace='99', end_shifts=[-15, -10, -5, 0], do_forecast=True)
    learn_best_bots(n_bots=1, namespace='99', end_shifts=[-5, 0], do_forecast=True)

    # TODO: запуск get_formatted_schedule


if __name__ == '__main__':

    logger.setup(level=logger.INFO, layout='debug')

    # установим количество каналов данных
    os.environ['ROENTGEN.N_CHANNELS'] = str(6)

    # установим дату начала прогноза
    os.environ['ROENTGEN.FORECAST_START_DATE'] = '2024-04-29'

    # predict_and_calc_schedule()




