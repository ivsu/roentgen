import hashlib
import numpy as np
import os
import copy
from gluonts.time_feature import (
    month_of_year, week_of_year
)
from common.logger import Logger
from settings import BOTS_FOLDER

logger = Logger(__name__)


# PROJECT_FOLDER = '/content/drive/MyDrive/university/roentgen/'


class Hyperparameters:
    def __init__(self):
        # словарь фиксированных параметров, которые не меняются в процессе обучения
        self.fixed = dict(
            # пространство имён ботов, чтобы различать их на диске,
            # если потребуется задать другой набор гиперпараметров
            namespace='02',
            # частота данных
            freq='W-SUN',
            # длина предсказываемой последовательности
            prediction_len=5,
            # папка для хранения ботов
            bots_folder=BOTS_FOLDER,
            # количество итераций поиска (смены популяций ботов)
            # TODO: заменить на критерий окончания поиска
            n_search=15,
            # количество эпох на обучение (испытание) одного бота
            n_epochs=15,
            # количество эпох на прогрев модели и на затухание LR
            warmup_epochs=5,
            decay_epochs=10,
            # общее количество ботов в популяции
            n_bots=15,
            # количество выживающих ботов (>=2)
            n_survived=5,
            # количество случайных ботов в каждой новой популяции
            n_random=5,
            # сдвиг на количество шагов (с конца) при обучении бота на временном ряде разной длины
            end_shifts=[-20, -15, -10, -5, 0],
            # количество батчей на эпоху - рассчитываемый
            num_batches_per_epoch=None
        )
        # имена расчётных параметров
        self.calculated = ['num_batches_per_epoch']
        # имена фиксированных параметров, которые включаются в хэш бота
        self.hashable = ['namespace', 'prediction_len', 'warmup_epochs', 'n_epochs',
                         'end_shifts', 'num_batches_per_epoch']
        # имена параметров, которые не отображаются в сокращённом выводе repr
        self.excluded_in_short_mode = ['namespace', 'bots_folder', 'n_search', 'n_bots',
                                       'n_survived', 'n_random']
        # дополнительные признаки - временные лаги - на сколько недель мы "смотрим назад"
        self.lags_sequence_set = [
            [1, 2, 3, 4, 5, 51, 52, 53, 103, 104, 105],
            [1, 2, 3, 4, 5, 50, 51, 52, 53, 54, 102, 103, 104, 105, 106],
            [51, 52, 53, 103, 104, 105],
            [52, 104],
            [50, 51, 52, 53, 54, 102, 103, 104, 105, 106],
            [49, 50, 51, 52, 101, 102, 103, 104],
            [52, 53, 54, 55, 104, 105, 106, 107],
        ]
        # временные фичи
        self.time_features_set = [
            [week_of_year],
            [week_of_year, month_of_year],
        ]
        # пространство динамических гиперпараметров, которые меняются
        # в процессе работы генетического алгоритма
        self.space = dict(
            # размер батча тренировочной выборки
            train_batch_size=[8, 16, 32, 64, 128],
            # соотношение длины контекста к длине предсказываемой последовательности
            context_ratio=[0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0],
            # дополнительные признаки - временные лаги - на сколько недель мы "смотрим назад"
            lags_sequence_index=[s for s in range(len(self.lags_sequence_set))],
            # функции применяемые в качестве временных фичей
            time_features_index=[s for s in range(len(self.time_features_set))],
            # параметры трансформера:
            embedding_dim=[1, 2, 4, 8],
            encoder_layers=[1, 2, 4, 8],
            decoder_layers=[1, 2, 4, 8],
            d_model=[8, 16, 32, 64],
            # distribution_output (string, optional, defaults to "student_t")
            # The distribution emission head for the model.
            # Could be either “student_t”, “normal” or “negative_binomial”.
        )
        # индексы динамических гиперпараметров по умолчанию (для режима default)
        self.defaults = [3, 1, 3, 0, 1, 1, 1, 0]
        assert len(self.space) == len(self.defaults)
        # ключи параметров данных (для изменения только параметров данных у лучших ботов)
        self.data_keys = ['train_batch_size', 'context_ratio',
                          'lags_sequence_index', 'time_features_index']
        # создаём генератор случайных чисел
        self.gen = np.random.default_rng()

    def get(self, key):
        """Возвращает значение фиксированного гиперпараметра по его ключу"""
        if key in self.fixed:
            return self.fixed[key]
        else:
            raise KeyError(f'Не задан фиксированный гиперпараметр с ключом: {key}')

    def set(self, key, value):
        """Задаёт значение фиксированного гиперпараметра"""
        if key in self.fixed:
            self.fixed[key] = value
        else:
            raise KeyError(f'Фиксированные гиперпараметры не содержат ключа: {key}')

    def generate(self, mode, hashes, current_values=None):
        """
        Генерирует набор значений гиперпараметров в заданном режиме.

        :param mode: режим генерации: default, random, data_nearest
        :param hashes: список хэшей имеющихся гиперпараметров для проверки
                новых ботов на уникальность
        :param current_values: имеющийся набор значений для режима data_nearest
        :returns: словарь гиперпараметров и их хэш
        """
        # сначала копируем значения фиксированных параметров
        values = copy.deepcopy(self.fixed.copy())

        # установка параметров по умолчанию
        if mode == 'default':
            for i, key in enumerate(self.space):
                values[key] = self.space[key][self.defaults[i]]
            return values, self.get_hash(values)

        # для режима поиска ближайшего значения сформируем исходную маску
        if mode == 'data_nearest':
            # формируем матрицу поиска ближайшего значения в пространстве параметров данных
            m_shape = ()
            for key in self.data_keys:
                m_shape += (len(self.space[key]),)
            matrix = np.ones(m_shape, dtype=np.int32)
            # print(f'm_shape: {m_shape}')
            # получаем положение в матрице по текущим параметрам данных
            current_pos = ()
            for key in self.data_keys:
                current_pos += (self.space[key].index(current_values[key]),)
            current_pos = np.array(current_pos)

        # будем пробовать установить уникальный набор параметров
        collisions = 0
        while True:
            # инициализуем значения гиперпараметров случайным образом
            if mode == 'random':
                for key in self.space:
                    value_index = self.gen.integers(0, len(self.space[key]))
                    values[key] = self.space[key][value_index]

            # инициализуем значения параметров данных к ближайшим значениям
            elif mode == 'data_nearest':

                # будем искать значение не далее двух шагов от текущего
                m_lookup = []
                for distance in range(1, 3):

                    mask = self._get_square_mask(m_shape, current_pos, distance)
                    m_lookup = matrix * mask
                    # получаем индексы ненулевых значений по каждому измерению
                    nonzero = np.asarray(np.nonzero(m_lookup), dtype=np.int32)
                    # если все значения кончились, расширим диапазон поиска
                    if nonzero.shape[1] == 0:
                        continue
                    # получим случайный набор индексов
                    rnd_index = self.gen.integers(0, nonzero[0].size, dtype=np.int32)
                    pos = nonzero[:, rnd_index]
                    # print(f'Изменение индексов ближайшего значения в параметрах данных: {current_pos} -> {pos}')

                    # маскируем найденное значение, чтобы при продолжении
                    # поиска, оно не попало в выбор
                    # print(f'matrix shape: {matrix.shape}')
                    matrix[tuple(i for i in pos)] = 0

                    # формируем значения параметров
                    data_param_index = 0
                    for key in self.space:
                        if key in self.data_keys:
                            values[key] = self.space[key][pos[data_param_index]]
                            data_param_index += 1
                        else:
                            values[key] = current_values[key]

                    # параметры подобраны
                    break

                # если найти новое значение не удалось, сообщим об этом коллеру
                if m_lookup.sum() == 0:
                    return None, None

            else:
                raise KeyError("Неверный режим: " + mode)

            # формируем хэш полученного набора значений
            bot_hash = self.get_hash(values)
            # проверяем на уникальность
            if bot_hash in hashes:
                collisions += 1
                print(f'Бот существует. Повторная попытка создания нового бота: {collisions}')
                if collisions > 20:
                    print(f'Завершены попытки найти уникальную конфигурацию бота,'
                          f' mode: {mode}, hash: {bot_hash}.')
                    return None, None
                continue
            break

        return values, bot_hash

    @classmethod
    def _get_square_mask(cls, shape, pos, distance):
        """
        Формирует квадратную маску по краям заданной области.
        Args:
            shape: форма массива, в рамках которого формируется маска
            pos: координаты точки в массиве, вокруг которой формируется маска
            distance: удалённость рамки от исходной точки
        """
        assert len(shape) == len(pos)
        assert distance > 0
        dims = len(shape)
        mask = np.zeros(shape, dtype=np.int32)
        zero_limits = []

        # определяем слайсы для каждого измерения
        slicing = ()
        for dim in range(dims):
            left = max(pos[dim] - distance, 0)
            right = min(pos[dim] + distance + 1, shape[dim])
            slicing += (slice(left, right),)
            # запоминаем диапазон для заполнения нулями
            left_shift = 1 if pos[dim] - distance >= 0 else 0
            right_shift = 1 if pos[dim] + distance < shape[dim] else 0
            zero_limits.append([left + left_shift, right - right_shift])

        # маскируем полученную область единицами
        mask[slicing] = 1

        # определяем слайсы для внутренней области
        slicing = ()
        for dim in range(dims):
            left = zero_limits[dim][0]
            right = zero_limits[dim][1]
            slicing += (slice(left, right),)

        # маскируем область внутри квадрата нулями
        mask[slicing] = 0

        return mask

    def get_hash(self, values):
        """Возвращает хэш значений параметров"""
        if values:
            keys = sorted(values.keys())
            s = "".join(
                f"{str(k)}={str(values[k])}"
                for k in keys
                if k not in self.fixed or k in self.hashable
            )
            return hashlib.sha256(s.encode("utf-8")).hexdigest()[:32]
        raise Exception('Попытка вычисления хэша на незаданных values')

    def repr(self, values, mode=None):
        output = "Параметры:\n"
        max_len = max([len(key) for key in dict(self.fixed, **self.space).keys()])
        for k, v in values.items():
            if mode == 'short' and k in self.excluded_in_short_mode:
                continue
            if k == 'lags_sequence_index':
                k, v = 'lags_sequence', self.lags_sequence_set[v]
            if k == 'time_features_index':
                k, v = 'time_features', self.time_features_set[v]
            output += f'{k:>{max_len}}: {v}\n'
        return output


if __name__ == '__main__':
    os.chdir('..')
    logger.setup(level=logger.INFO, layout='debug')

    check_hp = Hyperparameters()

    # сгенерируем набор дефолтных гиперпараметров и посмотрим на их значения
    check_values, check_bot_hash = check_hp.generate(mode='default', hashes=[])
    print(check_hp.repr(check_values))
