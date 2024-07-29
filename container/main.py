import sys

from common.showdata import show_time_series_by_year

POSSIBLE_ARGS = [
    '--show-time-series-by-year',
]


def print_help():
    print('Список возможных аргументов:')
    print("\n".join(POSSIBLE_ARGS))


if __name__ == '__main__':

    args = sys.argv
    if len(args) <= 1:
        raise RuntimeError('Необходимо указать аргументы.')

    mode = args[1]
    if mode not in POSSIBLE_ARGS:
        print_help()

    if mode == '--show-time-series-by-year':
        show_time_series_by_year()