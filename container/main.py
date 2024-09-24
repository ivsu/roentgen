import sys
import os

from common.showdata import show_time_series_by_year, show_doctors, show_legend, show_sample_example
from workplan.planner import search_hyperparameters, learn_best_bots
from schedule.scheduler import calculate_schedule

VALID_ARGS = [
    '--show-legend',
    '--show-doctors',
    '--show-time-series-by-year',
    '--show-sample-example',
    '--search-hyperparameters',
    '--learn-best-bots',
    '--calculate-schedule',
    '--test',
]


def print_help():
    print('Список возможных аргументов:')
    print("\n".join(VALID_ARGS))


if __name__ == '__main__':



    args = sys.argv
    if len(args) <= 1:
        raise RuntimeError('Необходимо указать аргументы.')

    mode = args[1]
    if mode not in VALID_ARGS:
        print_help()

    options_len = len(args) - 2
    if options_len > 0:
        for option in args[2:]:
            parts = option.split('=')
            if '--n_channels' == parts[0]:
                assert len(parts) == 2, "Неверно задан параметр для опции --n_channels."
                os.environ['ROENTGEN.N_CHANNELS'] = str(parts[1])

    if mode == '--show-legend':
        show_legend()
    elif mode == '--show-doctors':
        show_doctors()
    elif mode == '--show-time-series-by-year':
        show_time_series_by_year(data_version='train')
    elif mode == '--show-sample-example':
        show_sample_example(batch_size=8)
    elif mode == '--search-hyperparameters':
        search_hyperparameters()
    elif mode == '--learn-best-bots':
        learn_best_bots()
    elif mode == '--test':
        print(f'test, args: {args}')
    elif mode == '--calculate-schedule':
        raise NotImplementedError()
        # calculate_schedule(datetime(2024, 1, 1))
#
"""
    Алгоритм прогноза и расчёта графика:
    + сохраняем единичного бота на диск (также будет работать и при подборе гиперпараметров)
    - считываем и обучаем одного лучшего бота с параметром save_forecast=True
      TODO: разобраться, как сформировать датасет - типа, нулями его заполнить для future_values
    - формируем номера недель для прогноза, удаляем имеющиеся за этот период данные из БД и пишем прогнозные
      с версией forecast
    - в Scheduler:
      - настроить понятный вывод результатов поиска;
      - прикрутить вывод графика в plotly
      - настрить сохранение в SQLite
      - считываем и выводим в виде датафрейма
      - сделать запуск
    - запускаем scheduler
"""