import sys
import os

from common.showdata import show_time_series_by_year, show_doctors, show_legend, show_sample_example
from workplan.planner import search_hyperparameters, learn_best_bots, learn_default
from schedule.scheduler2 import calculate_schedule
from schedule.show_schedule import ShowSchedule

VALID_MODES = [
    '--show-legend',
    '--show-doctors',
    '--show-time-series-by-year',
    '--show-sample-example',
    '--train-model',
    '--calculate-schedule',
    '--show-schedule',
    '--search-hyperparameters',
    '--learn-best-bots',
    '--calculate-schedule',
    '--test',
]


def check_environ(variable):
    if variable not in os.environ:
        raise ValueError('Не задана переменная:', variable)


def print_help():
    print('Список возможных режимов:')
    print("\n".join(VALID_MODES))


if __name__ == '__main__':

    args = sys.argv
    if len(args) <= 1:
        raise RuntimeError('Необходимо указать аргументы.')

    mode = args[1]
    if mode not in VALID_MODES:
        print_help()

    options_len = len(args) - 2
    if options_len > 0:
        for option in args[2:]:
            parts = option.split('=')
            if '--n_channels' == parts[0]:
                assert len(parts) == 2, "Неверно задан параметр для опции --n_channels."
                os.environ['ROENTGEN.N_CHANNELS'] = str(parts[1])
            elif '--forecast_start_date' == parts[0]:
                assert len(parts) == 2, "Неверно задан параметр для опции --forecast_start_date."
                # установим дату начала прогноза (датасет будет урезан до неё)
                os.environ['ROENTGEN.FORECAST_START_DATE'] = str(parts[1])
            elif '--schedule_start_date' == parts[0]:
                assert len(parts) == 2, "Неверно задан параметр для опции --schedule_start_date."
                # установим дату начала прогноза (датасет будет урезан до неё)
                os.environ['ROENTGEN.SCHEDULE_START_DATE'] = str(parts[1])

    if mode == '--show-legend':
        check_environ('ROENTGEN.N_CHANNELS')
        show_legend()
    elif mode == '--show-doctors':
        show_doctors()
    elif mode == '--show-time-series-by-year':
        check_environ('ROENTGEN.N_CHANNELS')
        show_time_series_by_year(data_version='train')
    elif mode == '--show-sample-example':
        check_environ('ROENTGEN.FORECAST_START_DATE')
        show_sample_example(batch_size=8)
    elif mode == '--train-model':
        # TODO: починить вывод дашборда в колаб
        check_environ('ROENTGEN.N_CHANNELS')
        check_environ('ROENTGEN.FORECAST_START_DATE')
        # обучим модель с параметрами по умолчанию, выполним прогноз работ от даты начала прогноза
        # и запишем рассчитанный план работ в БД
        learn_default(namespace='99', data_version='train', do_forecast=True)  # source, train, debug
    elif mode == '--calculate-schedule':
        check_environ('ROENTGEN.SCHEDULE_START_DATE')
        calculate_schedule(
            plan_version='forecast',
            n_generations=30,
            population_size=100,
            n_survived=50
        )  # validation, forecast
    elif mode == '--show-schedule':
        # TODO: настроить вывод легенды в колабе
        check_environ('ROENTGEN.SCHEDULE_START_DATE')
        show = ShowSchedule(read_data=True)
        show.plot()
    elif mode == '--search-hyperparameters':
        search_hyperparameters()
    elif mode == '--learn-best-bots':
        learn_best_bots()
    elif mode == '--test':
        print(f'test, args: {args}')
