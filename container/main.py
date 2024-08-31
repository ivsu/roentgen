import sys

from common.showdata import show_time_series_by_year, show_doctors, show_legend
from workplan.planner import search_hyperparameters, learn_best_bots

VALID_ARGS = [
    '--show-legend',
    '--show-doctors',
    '--show-time-series-by-year',
    '--search-hyperparameters',
    '--learn-best-bots'
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

    if mode == '--show-legend':
        show_legend()
    elif mode == '--show-doctors':
        show_doctors()
    elif mode == '--show-time-series-by-year':
        show_time_series_by_year()
    elif mode == '--search-hyperparameters':
        search_hyperparameters()
    elif mode == '--learn-best-bots':
        learn_best_bots()