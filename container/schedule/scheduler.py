import copy

import uuid
import os
import gc
import re
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import collections
import matplotlib.pyplot as plt
from matplotlib_inline.config import InlineBackend
from plotly import graph_objects as go

from schedule.dataloader import DataLoader, get_month_layout, MODALITIES, SCHEDULE_TYPES
from settings import DB_VERSION
from common.logger import Logger
from schedule.dataset import load_doctor_availability

if DB_VERSION == 'PG':
    from common.db import DB
else:
    from common.dblite import DB
# from common.db import Transaction  # TODO: сделать это для SQLLite + save_bot

logger = Logger(__name__)
InlineBackend.figure_format = 'retina'  # TODO: переделать на plotly

RANDOM_SCHEDULE_TYPE = False
SECONDS = 3600.  # количество секунд в часах (устанавливается в 1 для режима test)

SELECTION_METHOD = 'tournament'  # метод отбора популяции: best_bots, tournament
MATE_RATE = 0.01  # пропорция генов, которыми обмениваются родители при скрещивании
MUTATE_PROBABILITY = 0.2  # вероятносить применения мутации
MUTATE_RATE = 0.01  # количество случайных переставляемых генов при мутации

TMP_FILE_PATH = '/Users/ivan/Documents/CIFROPRO/Проекты/Нейронки/Расписание рентген-центра/'


def calculate_schedule(plan_version, n_generations=30, population_size=100, n_survived=50,
                       generate_doctor_availability=False,
                       correct_doctor_table=False):

    assert 'ROENTGEN.FORECAST_START_DATE' in os.environ, 'В переменных среды не задана дата начала расчёта графика.'
    schedule_month_start = datetime.fromisoformat(os.environ['ROENTGEN.FORECAST_START_DATE'])

    # генерируем график доступности врачей
    if generate_doctor_availability:
        load_doctor_availability('doctor_availability', schedule_month_start, version='base',
                                 msg=f'Записано в БД строк графика работы врачей на период '
                                 f'{schedule_month_start.strftime("%B %Y")}')

    scheduler = Scheduler(
        plan_version=plan_version,
        n_generations=n_generations,
        population_size=population_size,
        n_survived=n_survived,
        mode='main',
        correct_doctor_table=correct_doctor_table
    )
    scheduler.run(save=True)


def plot(x, y_dict, title):
    for key in y_dict.keys():
        y = y_dict[key]
        plt.plot(x, y)
    plt.title(title)
    plt.legend(y_dict.keys())
    plt.show()


class Container:
    """Класс для согласованного хранения расписаний и связанных данных"""

    def __init__(self):
        self.bots = np.array([])
        self.bots_data = np.array([])
        self.scores = np.array([])
        self.generation = None
        self.gen = np.random.default_rng()

    def insert(self, bots, bots_data=None, scores=None, index=None, source=None, start_id=0, generation=None):

        if generation is not None:
            self.generation = generation

        if index is None:
            index = len(self.bots)
        assert index <= len(
            self.bots), f'Переданный индекс превышает количество имеющихся ботов: {index} / {len(self.bots)}'

        n_exists = len(self.bots)

        self.bots = np.vstack([self.bots[:index], bots, self.bots[index:]]) \
            if n_exists > 0 else bots

        if bots_data is None:
            ids = self._get_ids(bots.shape[0], start_id)
            bots_data = np.array([{'id': bot_id, 'source': source} for bot_id in ids])

        self.bots_data = np.hstack([self.bots_data[:index], bots_data, self.bots_data[index:]]) \
            if n_exists > 0 else bots_data

        if scores is not None:
            assert len(self.scores) == n_exists, \
                f'Не заданы оценки для уже имеющихся ботов: {len(self.scores)} / {n_exists}'
            self.scores = np.hstack([self.scores[:index], scores, self.scores[index:]]) \
                if n_exists > 0 else scores

    def _get_ids(self, range_len, start_id=0):
        assert self.generation is not None
        id_range = range(start_id, start_id + range_len)
        return [f'{self.generation:04d}/{i:03d}' for i in id_range]

    def unpack(self, copy=False):
        if copy:
            return self.bots.copy(), self.bots_data.copy(), self.scores.copy()
        else:
            return self.bots, self.bots_data, self.scores

    def tournament(self, num):
        """Отбирает ботов методов турнирного отбора. num должно быть кратно population_size"""
        assert len(self.scores) == len(self.bots) and self.scores.size
        # assert len(self.scores) % num == 0
        indices = np.arange(len(self.scores))
        self.gen.shuffle(indices)
        scores = self.scores[indices].copy()
        rest = len(scores) % num
        # дополним массивы индексов и оценок значениями, которые в дальнейшем не попадут выборку
        if rest > 0:
            indices = np.hstack([indices, np.repeat(-1, num - rest)])
            scores = np.hstack([scores, np.repeat(-1., num - rest)])
        indices = indices.reshape((num, -1), order='F')
        scores = scores.reshape((num, -1), order='F')
        # print(f'indices:\n{indices}')
        # print(f'scores:\n{scores}')

        best_indices_in_group = np.argpartition(scores, -1)[:, -1:].flatten()
        # print(f'best_indices_in_group: {best_indices_in_group}')
        best_indices_in_scores = indices[np.arange(best_indices_in_group.shape[0]), best_indices_in_group]
        # print(f'best_indices_in_scores: {best_indices_in_scores}')
        index = np.flip(best_indices_in_scores[np.argsort(self.scores[best_indices_in_scores])])
        # print(f'index (sorted): {index}')
        # print(f'scores (winners): {self.scores[index]}')
        # print(f'self.scores: {self.scores}')

        return self.bots[index].copy(), self.bots_data[index].copy(), self.scores[index].copy()

    def best_bots(self, num=None, copy=False):
        """Возвращает num лучших ботов"""
        if num is None:
            num = len(self.scores)

        # получаем индексы лучших ботов, отсортированные в порядке убывания оценки
        best_indices = np.argpartition(self.scores, -num)[-num:]
        index = np.flip(best_indices[np.argsort(self.scores[best_indices])])

        if copy:
            return self.bots[index].copy(), self.bots_data[index].copy(), self.scores[index].copy()
        else:
            return self.bots[index], self.bots_data[index], self.scores[index]

    def get_extended_data(self):
        assert len(self.scores) == len(self.bots_data)

        _, bots_data, scores = self.best_bots()

        return np.array([{
            'id': bots_data[i]["id"],
            'source': bots_data[i]["source"],
            'score': scores[i]
        } for i in range(len(bots_data))])

    def extract(self, amount):
        bots = self.bots[-amount:].copy()
        bots_data = self.bots_data[-amount:].copy()
        self.bots, self.bots_data = self.bots[:-amount], self.bots_data[:-amount]

        if self.scores.size == 0:
            scores = self.scores.copy()
        else:
            scores = self.scores[-amount:].copy()
            self.scores = self.scores[:-amount]
        return bots, bots_data, scores

    def get_scores(self):
        return self.scores

    def set_scores(self, scores):
        self.scores = scores

    def print(self, mode=None):
        logger.info(f'Форма вектора ботов (bots): {self.bots.shape}')
        logger.info(f'Форма вектора данных (bots_data): {self.bots_data.shape}')
        if mode == 'test':
            logger.info(f'bots:\n{self.bots}')


class Scheduler:
    """
    Класс реализует генетический алгоритм подбора расписания работы врачей на месяц
    для выполнения заданного плана работ.
    Алгоритм:
    - формируем подложку - рандомное расписание для каждого врача с учётом его графика работы, ставки
        и доступных модальностей, но без учёта выходных дней и дней отсутствия (все дни рабочие)
    - операции мутации:
        - смещение выходных дней (возможно, на один день вправо-влево)
        - изменение модальности (при наличии) в подложке
    - оценка: для каждой модальности и дня смотрим насколько выполняется план и для каждого врача оцениваем
        его вклад в выполнение плана:
        - если врач работает по модальности, по которой план недовыполняется - оценка выше
        - если врач работает по модальности, по которой план перевыполняется - оценка ниже
        - если у врача выходной, когда нужно выполнять план по одной из доступных ему модальностей, то оценка ниже.
    - скрещивание:
        - для каждого врача получаем интегральную оценку, и из двух ботов формируем потомка
            с наилучшими генами (оцененное расписание каждого врача).

    Основной объект v_bot: [bot_id, doctor_id, modality, day_index] = time_volume
    Вектор модальностей v_mod: [1, doctor_id, modality, 1] = 0, 1, 2 (основная)
    Вектор доступности по базовому графику v_base_avail: [1, doctor_id, 1, day_index] = -1, 0, 1
    Вектор врачей v_doctor: [1, doctor_id, 1, props] = any (нормальная продолжительность рабочего дня, ...)
    """

    def __init__(self, plan_version, n_generations, population_size, n_survived, mode,
                 correct_doctor_table=False):
        """mode: main, test"""
        self.n_generations = n_generations
        self.population_size = population_size
        self.n_survived = n_survived
        self.mode = mode
        self.n_mods = len(MODALITIES)
        self.db_schema = None
        if DB_VERSION == 'PG':
            self.db_schema = 'test' if mode == 'test' else 'roentgen'
        self.gen = np.random.default_rng()
        self.dataloader = DataLoader(self.db_schema)
        print('Количество поколений:', n_generations)
        print('Размер популяции:', population_size)
        print('Количество выживающих:', n_survived)
        ax = np.newaxis

        # получаем таблицу врачей
        doctor_df = self.dataloader.get_doctors()
        doctor_df['schedule_type_index'] = doctor_df['schedule_type'].apply(lambda s: SCHEDULE_TYPES.index(s))
        self.doctor_df = doctor_df
        self.time_rate = doctor_df['time_rate'].to_numpy(dtype=np.float32, copy=True)[ax, :, ax, ax]

        # получаем вектор модальностей врачей: [1, num_doctors, self.n_mods, 1]
        self.v_mod = self._get_doctors_mods()

        if 'ROENTGEN.SCHEDULE_START_DATE' not in os.environ:
            raise ValueError('В переменных среды не задана дата начала прогнозного плана работ.')
        month_start = datetime.fromisoformat(os.environ['ROENTGEN.SCHEDULE_START_DATE'])
        if month_start.day != 1:
            # начинаем расчёт графика с начала следующего месяца
            month_start = (month_start.replace(day=1) + timedelta(days=32)).replace(day=1)

        self.month_layout = get_month_layout(month_start)

        # считываем план на текущий месяц по дням и модальностям (в секундах)
        day_plan_df = self.dataloader.get_day_plan(plan_version, month_start, with_ce=False)
        # pd.set_option('display.expand_frame_repr', False)
        # print(day_plan_df[day_plan_df['day_index'] >= 29].iloc[:20])
        day_plan_df = day_plan_df[day_plan_df['month'] == month_start.month]
        row_index = {mod: i for i, mod in enumerate(MODALITIES)}
        day_plan_df = day_plan_df.pivot(
            index=['modality'], columns=['day_index'], values=['time_volume']
        ).rename(index=row_index)
        day_plan_df.sort_index(inplace=True)
        # print(day_plan_df.head(7))
        self.v_plan = day_plan_df.to_numpy(copy=True, dtype=np.float32)[ax, ax, :, :]
        self.v_plan *= SECONDS
        # вектор весов модальностей, отражающий распределение работ по модальностям в плане работ
        # используется для генерации ботов в populate и при мутации ботов в сторону соответствия плану
        self.plan_mod_weights = self.v_plan.sum(axis=3, keepdims=True) / self.v_plan.sum()  # -> [1, 1, n_mods, 1]
        self.bot_mod_weights = None

        # print(f'v_plan:\n{self.v_plan[0, 0, :, :5]}')

        # получаем желаемый график работы (базовое расписание)
        base_schedule = self.dataloader.get_schedule('base', month_start, data_layer='day')
        # print(base_schedule.iloc[:20])
        self.v_base_avail = base_schedule.pivot(
            index=['doctor_id'], columns=['day_index'], values=['availability']
        ).to_numpy(dtype=np.int32, copy=True)[ax, :, ax, :]
        # обрезаем до количества дней в плане (актуально для режима test)
        n_days = self.v_plan.shape[3]
        self.v_base_avail = self.v_base_avail[..., :n_days]

        if correct_doctor_table and self.mode != 'test':
            available = self.correct_doctors()
            # корректировка данных
            self.doctor_df['available'] = pd.Series(available)
            self.doctor_df = self.doctor_df[self.doctor_df['available']]
            self.time_rate = self.time_rate[:, available, :, :]
            self.v_mod = self.v_mod[:, available, :, :]
            # self.v_plan = self.v_plan[:, available, :, :]
            self.v_base_avail = self.v_base_avail[:, available, :, :]

        # TODO: делаем сравнение ресурсов врачей с единственной модальностью к плану
        eval_for_single_mod = self.v_mod.sum

        # вектора для генерации расписаний с разным видом графика
        def gen_workday_matrix(template, repeats):
            wdm = []
            wdm_source = np.tile(np.array(template, dtype=np.int32), repeats)
            for day in (range(len(template))):
                wdm.append(wdm_source[day:day + n_days])
            return np.array(wdm, dtype=np.int32)

        self.wd52 = gen_workday_matrix([0, 0, 1, 1, 1, 1, 1], 6)
        self.wd22 = gen_workday_matrix([0, 0, 1, 1], 9)

        # вектор популяции ботов
        self.v_bot = None
        # вектор рейтингов всех генов в текущей популяции
        self.v_score = None
        # итоговая оценка каждого бота
        self.bot_scores = None

        self.total_score = None
        self.best_bots = None

    def populate(self, n_bots):
        """
        Создаёт вектор популяции врачей:
        v_bot: [n_bots, n_doctors, modality, day_index] = time_volume
        """
        # self.doctor_df, self.month_layout, self.v_base_avail, self.mode
        # -> container: v_bot
        np.set_printoptions(formatter=dict(float=lambda x: f'{x:7.5f}'))
        # print(f'n_bots: {n_bots}')

        n_doctors = self.v_mod.shape[1]
        n_days = self.v_plan.shape[3]
        work_days = np.zeros((n_bots, n_doctors, n_days), dtype=np.float32)

        schedule_type_index = self.doctor_df['schedule_type_index'].to_numpy(dtype=np.int32)
        # print(f'schedule_type_index: {schedule_type_index.shape}')

        for i, schedule_type in enumerate(SCHEDULE_TYPES):

            # print(f'i: {i}, schedule_type: {schedule_type}')
            if schedule_type == '5/2':
                schedule_variance = 7
                workdate_template = self.wd52
                time_volume = np.array([8.], dtype=np.float32)
            elif schedule_type == '2/2':
                schedule_variance = 4
                workdate_template = self.wd22
                time_volume = np.array([12.], dtype=np.float32)
            else:
                raise RuntimeError("Неизвестный вид графика:", schedule_type)

            current_schedule_index = np.argwhere(schedule_type_index == i)[:, 0]
            # print(f'current_schedule_index: {current_schedule_index.shape}')

            index_len = len(current_schedule_index)
            # формируем случайные индексы расписаний текущего вида графика в workdate_template
            random_schedule_index = self.gen.integers(
                0, schedule_variance, size=n_bots * index_len, dtype=np.int32
            )
            # print(f'random_schedule_index: {random_schedule_index.shape}')

            # заполняем расписания только с текущим видом графика, добавляем рабочее время
            work_days[:, current_schedule_index, :] = workdate_template[random_schedule_index] \
                .reshape(n_bots, index_len, n_days).astype(np.float32) * time_volume
            # print(f'work_days sample:\n{work_days[0, :10, :]}')

        # print(f'work_days {work_days.shape}')
        # создаём бота - добавляем ось модальностей
        v_bot = np.repeat(work_days[:, :, np.newaxis, :], repeats=self.n_mods, axis=2)
        # получаем вектор индексов случайной модальности из числа доступных
        v_random_mod_index = self._generate_doctors_mods(v_bot.shape)
        # print(f'v_random_mod_index: {v_random_mod_index.shape}')
        # TODO: отладка
        # unique, counts = np.unique(v_random_mod_index, return_counts=True)
        # print(f'unique: {unique}')
        # print(f'[populate] mods: {counts / counts.sum()}')
        # df = pd.DataFrame(v_random_mod_index)
        v_random_mod_index = v_random_mod_index.transpose((0, 1, 3, 2)).reshape((-1,))
        # v_bot
        v_bot_mod_mask = np.zeros(v_bot.shape).reshape((-1, self.n_mods))
        # print(f'v_bot_mod_mask: {v_bot_mod_mask.shape}')

        # advanced integer indexing
        v_bot_mod_mask[np.arange(v_bot_mod_mask.shape[0]), v_random_mod_index] = 1.
        v_bot_mod_mask = v_bot_mod_mask \
            .reshape((v_bot.shape[0], v_bot.shape[1], v_bot.shape[3], self.n_mods)) \
            .transpose((0, 1, 3, 2))
        # print(f'v_bot_mod_mask: {v_bot_mod_mask.shape}')
        # print('mod_mask:', mod_mask.shape)
        # print(f'mod_mask:\n{mod_mask[0]}')

        # оставляем рабочее время только на заданной модальности
        v_bot *= v_bot_mod_mask
        # умножаем на ставку рабочего времени
        v_bot *= self.time_rate
        # зануляем рабочее время в недоступные дни
        mask = self.v_base_avail.repeat(n_bots, axis=0).repeat(self.n_mods, axis=2)
        v_bot[mask < 0] = 0.
        # зануляем часть ботов при переизбытке ресурсов
        # if v_used is not None:
        #     v_bot *= v_used
        # переводим в секунды
        v_bot *= SECONDS

        # logger.info(f'initial bots schedule by modalities {bots.shape}:\n{bots.sum(axis=(1, 3)) / 3600}')
        # print(f'v_bot {v_bot.shape}')
        # print(f'v_bot sample:\n{v_bot[0, :6, :, :10]}')
        # print(f'v_base_avail sample:\n{self.v_base_avail[0, :6, 0, :10]}')

        return v_bot

    def evaluate(self, container: Container, counter):
        """
        ...
        Работы v_bot занулены в дни недоступности (self.v_base_avail = -1)

        :param container:
        :return:
        TODO: оценку выполнять понедельно
        """

        # time_pw = -10  # степень функции для оценки нормы рабочего времени в рамках дня
        fact_over_plan_pw = -2  # степень затухания оценки при превышении факта над планом
        works_plan_pw = -1  # степень затухания оценки соотношения графика работ и плана
        extra_mod_k = 0.98  # снижение оценки за работы не по основной модальности
        conflict_with_base_k = 0.98  # снижение оценки за несовпадение с базовым графиком

        v_bot, bots_data, _ = container.unpack()

        np.set_printoptions(formatter=dict(float=lambda x: f'{x:6.3f}'))

        def power_zero_and_norm(value, normal, power):
            # степенная функция, равная 1 в нуле и в значении нормы
            return np.power(value / normal * 2 - 1, power)

        def power_norm_and_more(value, normal, power):
            # степенная функция, равная 1 в значении нормы и затем затухающая
            # при отрицательной степени
            mask = value > 0
            out = value.copy()
            # TODO: исключить ошибку при делении на 0. (когда нет плана на день по модальности)
            # print(f'out: {out.shape}')
            # print(f'normal: {normal.shape}')
            # out[np.isclose(normal, 0.)] = 0.
            np.power(value / normal, power, out=out, where=mask)
            return out

        def linear_zero_and_norm(value, normal):
            # линейная функция, равная 0 в нуле и 1 в значении нормы времени
            return value / normal

        f_mod = self.v_mod > 0
        f_extra_mod = self.v_mod == 1  # 1 - дополнительная модальность, 2 - основная
        f_noworks = np.isclose(v_bot, 0.)
        # маска нерабочих дней в базовом графике
        f_not_working_days_in_base = self.v_base_avail < 1
        # f_not_working_days_in_base = f_not_working_days_in_base \
        #     .repeat(self.population_size, axis=0) \
        #     .repeat(self.n_mods, axis=2)
        # есть работы по своим модальностям
        f_work_frame = ~f_noworks & f_mod
        logger.debug(f'f_work_frame: {f_work_frame.shape}')

        # v_score = np.zeros(v_bot.shape)
        # начальная оценка 1. за всё время, на которое назначены работы
        # v_score[f_work_frame] = 1.
        # logger.debug(f'v_score: {v_score.shape}')

        # немного снижаем оценку за работы по дополнительной модальности
        # logger.info(f'v_score before extra_mod_k:\n{v_score}')
        # v_score[f_work_frame & f_extra_mod] *= extra_mod_k

        """ код для оценки рабочего дня с суммированием модальностей,
            но выяснилось, что по нескольким модальностям врач не работает в рамках смены
        print(f'work_day_duration: {work_day_duration.shape}')
        f_in_norm = work_day_duration <= f_day_time_norm
        print(f'f_in_norm: {f_in_norm.shape}')
        f_day_frame = np.any(f_work_frame, axis=2, keepdims=True)
        print(f'f_day_frame: {f_day_frame.shape}')

        result = linear_zero_and_norm(work_day_duration, f_day_time_norm)
        print(f'result: {result.shape}')
        v_score *= np.where(f_day_frame & f_in_norm, result, 1.)

        result = power_norm_and_more(work_day_duration, f_day_time_norm, work_day_duration_pw)
        # print(f'result: {result.shape}')
        v_score *= np.where(f_day_frame & ~f_in_norm, result, 1.)
        """

        # снижаем оценку за работы в несовпадающие с исходным графиком дни
        # logger.info(f'v_score before conflict_with_base_k:\n{v_score}')
        # v_score[f_work_frame & f_not_working_days_in_base] *= conflict_with_base_k

        # *
        # добавляем оценку выполнения плана
        # оценка линейно растёт при приближении факта к плану
        # и убывает при перевыполнении
        """
        day_fact = np.sum(v_bot, axis=1, keepdims=True)
        logger.debug(f'day_fact: {day_fact.shape}')
        in_plan = day_fact <= self.v_plan
        logger.debug(f'in_plan: {in_plan.shape}')
        plan_fact_scores = np.ones(in_plan.shape)

        # td = 3
        # print(f':: day_fact: {day_fact[0, 0, 0, td]}')
        # print(f':: self.v_plan: {self.v_plan[0, 0, 0, td]}')
        # print(f':: power_norm_and_more: {power_norm_and_more(day_fact[0, 0, 0, td], self.v_plan[0, 0, 0, td], fact_over_plan_pw)}')

        logger.debug(f'plan_fact_scores: {plan_fact_scores.shape}')
        # TODO: попробовать заменить на нормальное распределение со средним значением в точке плана
        #   и при факте = 0, значение должно быть 0.1, например.
        plan_fact_scores[in_plan] = linear_zero_and_norm(day_fact, self.v_plan)[in_plan]
        plan_fact_scores[~in_plan] = power_norm_and_more(
            day_fact, self.v_plan, fact_over_plan_pw
        )[~in_plan]
        # logger.info(f'v_score before plan_fact_scores:\n{v_score}')
        # logger.info(f'plan_fact_scores:\n{plan_fact_scores[:, 0, :, :]}')
        # v_score *= plan_fact_scores
        """

        #
        # вариант 2
        """
        day_works_by_mod = np.sum(v_bot, axis=1, keepdims=True)  # -> [:, 1, :, :]
        day_works_total = day_works_by_mod.sum(axis=2, keepdims=True)  # -> [:, 1, 1, :]
        print(f'\nday_works_by_mod:\n{day_works_by_mod[:, 0, :, :]}')
        print(f'\nday_works_total:\n{day_works_total[:, 0, 0, :]}')
        # общее отношение ресурсов к плану - соотношение, к которому должны стремиться все распределения
        resource_ratio = day_works_total.sum(axis=(1, 2, 3), keepdims=True) / self.v_plan.sum()  # -> [:, 1, 1, 1]
        print(f'\nresource_ratio: {resource_ratio[:, 0, 0, 0]}')

        exec_ratio_by_mod = day_works_by_mod / self.v_plan  # -> [:, 1, :, :]
        exec_ratio_total = day_works_total / self.v_plan.sum(axis=2, keepdims=True)  # -> [:, 1, 1, :]
        print(f'\nexec_ratio_by_mod:\n{exec_ratio_by_mod[:, 0, :, :]}')
        print(f'\nexec_ratio_total:\n{exec_ratio_total[:, 0, 0, :]}')

        # расстояние отклонения соотношения факт/план по каждой модальности от этого же соотношения
        # по всем модальностям в рамках каждого дня -> [:, 1, 1, :]
        works_plan_distance = np.sqrt(np.sum(np.square(exec_ratio_by_mod - exec_ratio_total), axis=2, keepdims=True))
        print(f'\nworks_plan_distance:\n{works_plan_distance[:, 0, 0, :]}')
        works_plan_scores = power_norm_and_more(
            works_plan_distance + 1., resource_ratio, works_plan_pw
        )
        print(f'\nworks_plan_scores:\n{works_plan_scores[:, 0, 0, :]}')
        v_score *= works_plan_scores
        print(f'\nv_score:\n{v_score}')

        # итоговая оценка по каждому боту
        bot_scores = np.sqrt(np.sum(np.square(v_score), axis=(1, 2, 3)))
        print(f'\nbot_scores: {bot_scores}')
        """

        #
        # вариант 3
        # print(f'evaluate bots_data: {bots_data}')

        # extra_mod_k = 0.9999  # снижение оценки за работы не по основной модальности
        # conflict_with_base_k = 0.9999  # снижение оценки за несовпадение с базовым графиком
        max_extra_mod_k = 0.99  # максимальный штраф за работы по неосновной модальности
        max_base_mismatch_k = 0.99  # максимальный штраф за несовпадение с базовым графиком

        day_works_by_mod = np.sum(v_bot, axis=1, keepdims=True)  # -> [:, 1, :, :]
        diff = day_works_by_mod - self.v_plan
        # print(f'diff: {diff.shape}')
        diff_minus = np.sum(diff, axis=(1, 2, 3), where=(diff < 0))
        diff_plus = np.sum(diff, axis=(1, 2, 3), where=(diff > 0))
        counter['diff'].append(np.min(diff_plus - diff_minus) / SECONDS)

        # общее количество назначенных работ по дням
        total_works = np.count_nonzero(v_bot, axis=(1, 2), keepdims=True)

        # рассчитываем штраф за работы по неосновной модальности
        extra_mod = np.zeros(v_bot.shape, dtype=np.int32)
        extra_mod[f_work_frame & f_extra_mod] = 1
        extra_mod = extra_mod.sum(axis=(1, 2), keepdims=True)
        extra_mod_rate = np.divide(extra_mod, total_works,
                                   where=(total_works > 0))
        # print(f'extra_mod_rate:\n{extra_mod_rate}')
        extra_mod_penalty = max_extra_mod_k + (1 - extra_mod_rate) * (1 - max_extra_mod_k)
        # print(f'extra_mod_penalty:\n{extra_mod_penalty}')
        del extra_mod, extra_mod_rate
        gc.collect()

        # рассчитываем штраф за работы/отдых в несовпадающие с исходным графиком дни
        doctor_workdays = np.where(v_bot.sum(axis=2, dtype=np.int32, keepdims=True) > 0, True, False)
        # TODO: уточнить со значением -1
        bool_avail = np.where(self.v_base_avail == 1, True, False)
        base_mismatch_rate = np.sum(doctor_workdays ^ bool_avail, axis=1, keepdims=True) / doctor_workdays.shape[1]
        base_mismatch_penalty = max_base_mismatch_k + (1 - base_mismatch_rate) * (1 - max_base_mismatch_k)
        # print(f'base_mismatch_rate:\n{base_mismatch_rate[:, 0, 0, :]}')
        # print(f'base_mismatch_penalty:\n{base_mismatch_penalty[:, 0, 0, :]}')
        # del doctor_workdays, bool_availv, base_mismatch_rate
        # gc.collect()

        # применяем понижающие коэффициенты:
        diff *= 1 / (extra_mod_penalty * base_mismatch_penalty)

        diff_minus = np.sum(diff, axis=(1, 2, 3), where=(diff < 0))
        diff_plus = np.sum(diff, axis=(1, 2, 3), where=(diff > 0))
        diff_total = diff_plus - diff_minus
        # print(f'diff by bots:\n{diff_minus}\n{diff_plus}\n{diff_plus - diff_minus}')
        counter['diff_minus'].append(-diff_minus.max() / SECONDS)
        counter['diff_plus'].append(diff_plus.min() / SECONDS)
        counter['diff_weighted'].append(diff_total.min() / SECONDS)

        m = 1e3 if mode == 'test' else 1e10
        bot_scores = m / diff_total
        # print(f'evaluated bot_scores: {bot_scores}')

        return bot_scores

    def select(self, container: Container, method=None, generation=None) -> Container:

        assert method in ['best_bots', 'tournament'], f'Неизвестный метод отбора: {method}'

        if method == 'tournament':
            best_bots, best_bots_data, scores = container.tournament(self.n_survived)
        else:
            best_bots, best_bots_data, scores = container.best_bots(self.n_survived, copy=True)

        # кладём лучших ботов в другой контейнер
        selected = Container()
        selected.insert(best_bots, best_bots_data, scores, generation=generation)
        # logger.info(f'scores best scores: {scores}')
        return selected

    def mate(self, bots):
        """
        Выполняет скрещивание каждой пары ботов путём обмена модальностями между ними.
        """
        assert self.n_survived % 2 == 0
        np.set_printoptions(formatter=dict(float=lambda x: f'{x:.1f}'))
        n_best_bots = bots.shape[0]
        n_doctors = bots.shape[1]
        n_days = bots.shape[3]
        # количество скрещивающихся генов (максимальное, т.к. применяется unique по
        n_mate_gens = n_doctors // 2

        # определяем, кто с кем будет скрещиватья
        mate_indices = np.arange(n_best_bots)
        self.gen.shuffle(mate_indices)
        logger.debug(f'mate_indices:\n{mate_indices}')

        # *
        # получаем вектор случайного скрещивания для всех пар родителей
        v_mate_len = self.n_survived // 2
        v_mate = np.zeros((v_mate_len, n_doctors, n_days), dtype=np.int32)

        # добавляем случайные индексы второго родителя
        # v_mate.shape = (v_mate_len, -1)
        # for i in range(v_mate.shape[0]):
        #     indices = np.unique(self.gen.integers(0, n_doctors, size=n_doctors // 2))
        #     v_mate[i][indices] = 1
        mate_gens = self.gen.random(size=(v_mate_len, n_doctors, n_days)) < MATE_RATE
        v_mate[mate_gens] = 1
        # v_mate.shape = (v_mate_len, n_doctors, bots.shape[3])

        new_bots = []
        logger.debug(f'v_mate {v_mate.shape}:\n{v_mate}')

        # производим мутацию для каждой пары согласно вектору случайного скрещивания
        for i in range(0, len(mate_indices), 2):
            mate_index = i // 2
            b0, b1 = bots[i], bots[i + 1]
            new_bot0, new_bot1 = b0.copy(), b1.copy()

            # print(f'f1: {f1.shape}')

            def change_modalities(bot0, bot1):
                """
                Переставляет работы по модальности у двух ботов.
                Для перестановки берётся модальность с максимальным объёмом работ.
                """
                # print('change_modalities bot0:\n', bot0)
                # print('change_modalities bot1:\n', bot1)
                mod_index0 = bot0.argmax(axis=1)
                mod_index1 = bot1.argmax(axis=1)
                new_index0 = mod_index0.copy()
                new_index1 = mod_index1.copy()

                mask = np.ones(mod_index0.shape, dtype=np.bool_)
                # TODO: ? возможно стоит обмениваться там, где есть работы
                # группируем работы бота-источника по врачам, модальностям
                # зануляем в маске работы, которых нет у бота источника
                # (?) умножаем на матрицу модальностей врачей
                # mask = mask & np.isclose(src_bot.sum(axis=2) * self.v_mod[0, :, :, 0], 0.)[:, :, np.newaxis]

                n_mods = self.n_mods
                v_mod = self.v_mod[0, :, :, :]
                v_mod = np.repeat(v_mod, bot0.shape[2], axis=-1).transpose((0, 2, 1))
                v_mod = v_mod.reshape((-1, n_mods))
                mod_allowed = v_mod[np.arange(v_mod.shape[0]), mod_index0.reshape(-1, )]
                mod_allowed.shape = mod_index0.shape
                logger.debug(f'mod_allowed {mod_allowed.shape}:\n{mod_allowed}')

                # оставляем модальности согласно вектору случайного скрещивания
                mask = mask & v_mate[mate_index] == 1
                logger.debug(f'mask:\n{mask}')
                # обмен индексами
                new_index0[mask] = mod_index1[mask]
                new_index1[mask] = mod_index0[mask]
                # перезапись модальнойстей ботов
                logger.debug(f'mod_index0:\n{mod_index0}')
                logger.debug(f'new_index0:\n{new_index0}')
                logger.debug(f'mod_index1:\n{mod_index1}')
                logger.debug(f'new_index1:\n{new_index1}')
                mod_index0.shape = (-1,)
                mod_index1.shape = (-1,)
                new_index0.shape = (-1,)
                new_index1.shape = (-1,)

                v_len = new_index0.shape[0]

                def prepare_for_advanced_indexing(bot):
                    """что-то не так с contiguous - создаётся новый объект"""
                    return bot.transpose((0, 2, 1)).reshape((-1, self.n_mods))

                def turn_back(transposed, original):
                    """Возврат бота к первоначальной форме"""
                    shape = original.shape
                    return transposed \
                        .reshape(shape[0], shape[2], shape[1]) \
                        .transpose((0, 2, 1))

                def swith_modalities(transposed, mod_index, new_index):
                    keeper = transposed[np.arange(v_len), new_index].copy()
                    transposed[np.arange(v_len), new_index] = transposed[np.arange(v_len), mod_index]
                    transposed[np.arange(v_len), mod_index] = keeper

                transposed0 = prepare_for_advanced_indexing(bot0)
                transposed1 = prepare_for_advanced_indexing(bot1)

                swith_modalities(transposed0, mod_index0, new_index0)
                swith_modalities(transposed1, mod_index1, new_index1)
                # print(f'transposed0:\n{transposed0}')
                # print(f'transposed1:\n{transposed1}')
                gc.collect()

                return turn_back(transposed0, bot0), turn_back(transposed1, bot1)

            def mate_pair(src_bot, new_bot):
                """
                возьмём из соседнего бота работы только по тем модальностям, которые есть
                в модальностях принимающего бота
                :param src_bot:
                :param new_bot:
                """
                # изначально разрешён полный перенос
                mask = np.ones(src_bot.shape, dtype=np.bool_)
                # группируем работы бота-источника по врачам, модальностям
                # зануляем в маске работы, которых нет у бота источника
                # (?) умножаем на матрицу модальностей врачей
                mask = mask & np.isclose(src_bot.sum(axis=2) * self.v_mod[0, :, :, 0], 0.)[:, :, np.newaxis]
                # оставляем работы согласно вектору случайного скрещивания
                mask = mask & v_mate[mate_index][:, np.newaxis, np.newaxis] == 1

                # new_bot[mask] = src_bot[mask]

            new_bot0, new_bot1 = change_modalities(new_bot0, new_bot1)
            logger.debug(f'new_bot0:\n{new_bot0}')
            logger.debug(f'new_bot1:\n{new_bot1}')
            gc.collect()

            logger.debug(f'b0 / new_bot0 sum: {b0.sum()} / {new_bot0.sum()}')
            logger.debug(f'b1 / new_bot1 sum: {b1.sum()} / {new_bot1.sum()}')

            # print(f'new_bot0: {new_bot0.shape}')
            # print(f'new_bot1: {new_bot1.shape}')
            new_bots.append(new_bot0)
            new_bots.append(new_bot1)

            logger.debug(f'mate_indices:\n{mate_indices[i:i + 2]}')

        # print(f'v_mod:\n{self.v_mod[0, :3]}')
        new_bots = np.array(new_bots)
        # logger.info(f'new_bots: {new_bots.shape}')
        return new_bots

    def mutate(self, bots):
        """Выполняет мутацию ботов путём переключения модальностей врачей из числа им доступных
        :deprecated: заменено на mutate_v2
        """
        # print(f'source bots {bots.shape}:\n{bots}')

        # вектор весов модальностей, отражающий выполнение ботами плана работ
        # [1, 1, n_mods, n_days] / [n_bots, 1, n_mods, n_days] -> [n_bots, 1, n_mods, n_days]
        mod_weights = self.v_plan / bots.sum(axis=1, keepdims=True)
        mod_weights_monitor = self.v_plan.sum(axis=(1, 3)) / bots.sum(axis=(1, 3))  # (!) для просмотра
        # print(f'work_weights: {mod_weights.shape}')
        # print(f'Выполнение плана (plan / schedule):\n{mod_weights.sum(axis=(1, 3))}')
        # print(f'v_plan[0, 0, 2, :10]: {self.v_plan[0, 0, 2, :10]}')
        # print(f'mod_weights[0, 0, 2, :10]: {mod_weights[0, 0, 2, :10]}')
        # print(f'bots.sum(axis=1, keepdims=True)[0, 0, 2, :10]: {bots.sum(axis=1, keepdims=True)[0, 0, 2, :10]}')
        # print(f'mod_weights_monitor:\n{mod_weights_monitor}')

        # берём индекс модальности с максимальным объёмом работ
        src_mod_indices = bots.argmax(axis=2).reshape((-1,))
        # print(f'src_mod_indices {src_mod_indices.shape}:\n{src_mod_indices}')

        # получаем индекс случайной модальности из доступных для врача
        rnd_mod_indices = self._generate_doctors_mods(bots.shape).reshape((-1,))
        # print(f'rnd_mod_indices {rnd_mod_indices.shape}:\n{rnd_mod_indices}')

        bots = bots.transpose((0, 1, 3, 2))
        bots_keep_shape = bots.shape
        bots = bots.reshape((-1, self.n_mods))
        long_index = np.arange(bots.shape[0])

        v_mutate = self.gen.random(size=len(long_index)) < MUTATE_RATE

        keep_values = bots[long_index, rnd_mod_indices].copy()
        # print(f'keep_values {keep_values.shape}:\n{keep_values}')

        bots[long_index, rnd_mod_indices] = np.where(
            v_mutate, bots[long_index, src_mod_indices], bots[long_index, rnd_mod_indices]
        )
        bots[long_index, src_mod_indices] = np.where(
            v_mutate, keep_values, bots[long_index, src_mod_indices]
        )
        # print('v_bot[long_index, rnd_mod_indices]:')
        # print(bots[long_index, rnd_mod_indices][v_mutate])
        # print(f'v_bot {bots.shape}:\n{bots}')
        bots.shape = bots_keep_shape
        bots = bots.transpose((0, 1, 3, 2))
        # print(f'v_mutate {v_mutate.shape}:\n{v_mutate}')
        # print(f'result bots {bots.shape}:\n{bots}')

        return bots

    def mutate_v2(self, bots, donors):
        # заменяем работы по модальностям и дням у части ботов/врачей
        mutate_k = 0.1  # доля мутирующих генов
        mask = self.gen.random(size=bots.shape[:2]) < mutate_k
        bots[mask] = donors[mask]
        return bots

    def run(self, save=False):
        logger.info(f'Форма вектора модальностей врачей (v_mod): {self.v_mod.shape}')
        logger.info(f'Форма вектора запланированных работ (v_plan): {self.v_plan.shape}')
        logger.info(f'Форма вектора доступности врачей (v_base_avail): {self.v_base_avail.shape}')
        logger.info(f'Вектор весов модальностей (mod_weights): {self.plan_mod_weights[0, 0, :, 0]},'
                    f' Сумма: {self.plan_mod_weights.sum():.4f}')
        counter = {'diff': [], 'diff_minus': [], 'diff_plus': [], 'diff_weighted': []}

        def print_stat(container: Container, title, first=None):
            g = container.generation
            bots, _, _ = container.best_bots()
            data = container.get_extended_data()
            cnt = collections.Counter()
            for d in data:
                cnt[d['source']] += 1
            if first is not None:
                data = data[:first]
            print(f'Поколение #{g}/{self.n_generations}, {title}:')
            print(f'Источники ботов в популяции: {dict(cnt)}')
            print(f'{"Бот":>12}{"Источник":>11}{"Оценка":>11}  {"Соотношение ресурсы/работы по модальностям"}')
            print('\n'.join(f'id: {d["id"]} {d["source"]:>10}  {d["score"]:9.5f}'
                            f'  {bots[i].sum(axis=(0, 2)) / self.v_plan.sum(axis=(0, 1, 3))}'
                            for i, d in enumerate(data)))

        random_bots = self.populate(n_bots=self.population_size)
        container = Container()
        container.insert(random_bots, source='populate', generation=0)
        container.print()
        # TODO: проверка назначения модальностей
        debug_doctor_works = random_bots.sum(axis=(0, 3))
        # debug_doctor_works = random_bots[0].sum(axis=2)
        for debug_doctor_index in range(0, 10):
            print(f'doctor: {debug_doctor_index}, mods: {self.v_mod[0, debug_doctor_index, :, 0]},'
                  f' works: {debug_doctor_works[debug_doctor_index] / SECONDS},'
                  f' total: {debug_doctor_works[debug_doctor_index].sum() / SECONDS}')

        # вычисляем оценку ботов
        bot_scores = self.evaluate(container, counter)
        # logger.info(f'bot_scores:\n{bot_scores}')
        container.set_scores(bot_scores)

        # производим отбор
        container = self.select(container, method=SELECTION_METHOD, generation=0)
        gc.collect()

        print(f'Первичное соотношение ресурсы/работы:'
              f' {np.mean(random_bots.sum(axis=(1, 2, 3)) / self.v_plan.sum(axis=(1, 2, 3))):.4f}')

        # debug
        print_stat(container, 'Лучшие боты')

        for gn in range(self.n_generations):
            generation = gn + 1
            # print(f'Поколение #{generation}')
            # получаем ботов из контейнера - там остались лучшие
            bots, bots_data, best_bots_scores = container.unpack()
            # формируем вектор ресурсов по модальностям (для корректировки вероятностей в populate)
            self.bot_mod_weights = bots.sum(axis=(0, 1, 3), keepdims=True) / bots.sum()

            # формируем случайных ботов
            # with timer('populate'):
            random_bots = self.populate(n_bots=self.population_size)
            container = Container()
            container.insert(random_bots, source='populate', generation=generation)
            # print(f'next_container bots: {next_container.bots.shape}')
            donors, _, _ = container.extract(self.n_survived)
            # print(f'next_container bots after extract: {next_container.bots.shape}')
            # print(f'donors bots: {donors.shape}')

            # скрещиваем лучших ботов
            # with timer('mate'):
            # TODO: доработать
            source = 'mate'
            descendants = self.mate(bots)
            # descendants = bots
            # print(f'descendants after mate: {descendants.shape}')

            # производим мутацию
            # with timer('mutate'):
            if self.gen.random() < MUTATE_PROBABILITY:
                descendants = self.mutate_v2(descendants, donors)
                source = 'mutate'
            # print(f'descendants after mutate_v2: {descendants.shape}')

            container.insert(descendants, source=source, start_id=self.n_survived, generation=generation)

            # рассчитываем оценку новых ботов
            # with timer('evaluate') as t:
            scores = self.evaluate(container, counter)
            container.set_scores(scores)
            print_stat(container, 'Оценки новых ботов (лучшие)', first=5)

            # формируем общий контейнер всех ботов: сначала лучшие, затем полученные в данном цикле
            container.insert(bots, bots_data, best_bots_scores, index=0)
            # print(f'next_container best_bots_scores: {next_container.best_bots_scores}')

            # производим отбор из всех ботов:
            container = self.select(container, method=SELECTION_METHOD, generation=generation)
            print_stat(container, 'Лучшие боты', first=5)

            gc.collect()

        bots, bots_data, scores = container.best_bots(1)
        best_bot, best_bot_score = bots[0], scores[0]
        best_bot_id, best_bot_source = bots_data[0]['id'], bots_data[0]['source']
        schedule = best_bot.sum(axis=1)
        v_plan = self.v_plan[0, 0, :, :]
        v_schedule = best_bot.sum(axis=0)
        diff = v_schedule - v_plan
        diff_k = v_schedule / v_plan

        fmt_4_1 = dict(float=lambda x: f'{x:4.1f}')
        fmt_6_4 = dict(float=lambda x: f'{x:6.4f}')
        fmt_6_1 = dict(float=lambda x: f'{x:6.1f}')

        print(f'\nЛучший бот: {best_bot_id} [{best_bot_source}], оценка: {best_bot_score:8.5f}')
        self.print_schedule(best_bot)

        # m = 3600
        if self.mode == 'test':
            # m = 1
            logger.info(f'self.v_base_avail:\n{self.v_base_avail[0, :, 0, :]}')
            # logger.info(f'v_plan:\n{v_plan}')
            logger.info(f'best_bot:\n{best_bot}')
        np.set_printoptions(formatter=fmt_4_1)
        # logger.info(f'schedule:\n{schedule / SECONDS}')
        np.set_printoptions(formatter=fmt_6_1)
        logger.info(f'Суммарный подневный график работы врачей по 6-ти модальностям (часы):\n{v_schedule / SECONDS}')
        logger.info(f'Суммарный подневный план объёмов работ по 6-ти модальностям (часы):\n{v_plan / SECONDS}')
        logger.info(f'Разница графика работы и плана по 6-ти модальностям (часы):\n{diff / SECONDS}')
        np.set_printoptions(formatter=fmt_6_4)
        logger.info(f'Относительное отклонение рассчитанного графика работы от начального плана:\n{diff_k}')
        np.set_printoptions(formatter=fmt_6_1)
        logger.info(f'Суммарный месячный график работ по 6-ти модальностям (часы):\n{v_schedule.sum(axis=1) / SECONDS}')
        logger.info(
            f'Суммарный месячный план объёмов работ по 6-ти модальностям (часы):\n{v_plan.sum(axis=1) / SECONDS}')
        logger.info(
            f'Суммарная месячная разница графика работы и плана по 6-ти модальностям (часы):\n{diff.sum(axis=1) / SECONDS}')
        logger.info(
            f'Итоговое относительное отклонение рассчитанного графика работы от начального плана:\n{v_schedule.sum(axis=1) / v_plan.sum(axis=1)}')
        logger.info(f'Общая разница рассчитанного графика и плана работ: {diff.sum() / SECONDS:.1f}')
        logger.info(f'Относительная разница рассчитанного графика и плана работ: {v_schedule.sum() / v_plan.sum():.3f}')

        plot(range(self.n_generations + 1), counter, 'Мин. разница график-план')
        # пишем лучшего бота в базу
        if save:
            self.save_bot(best_bot)

    def save_bot(self, bot):

        doctors = self.dataloader.get_doctors_for_schedule_save()
        doctors = doctors.set_index('row_index').T.to_dict()

        output = []
        for doctor_index in range(len(bot)):
            for day_index in range(len(bot[0, 0])):
                time_volumes = []
                mods = []
                for mod_index in range(len(bot[0])):
                    time_volume = timedelta(seconds=bot[doctor_index, mod_index, day_index].item())
                    if time_volume > timedelta(seconds=0):
                        time_volumes.append(time_volume)
                        mods.append(mod_index)
                output.append([doctor_index, mods, day_index, time_volumes])
        df = pd.DataFrame(output, columns=['doctor_index', 'mods', 'day_index', 'time_volumes'])

        version = 'final'
        doctor_day_plan = []

        def set_row(row):
            row['uid'] = uuid.uuid4()
            doctor_index = row['doctor_index']
            doctor = doctors[doctor_index]
            row['doctor'] = doctor['uid']
            row['day_start'] = datetime.combine(
                self.month_layout['month_start'] + timedelta(days=row['day_index']), doctor['day_start_time'])
            row['availability'] = self.v_base_avail[0, doctor_index, 0, row['day_index']]

            if row['availability'] == 1:
                day_time = timedelta(hours=8)
                if doctor['schedule_type'] == '2/2':
                    day_time = timedelta(hours=12)
                row['time_volume'] = day_time * doctor['time_rate']
            else:
                row['time_volume'] = timedelta(seconds=0)

            # формируем связанную таблицу по модальностям
            if len(row['mods']) > 0:
                mods = row['mods']
                for mod_index in mods:
                    # TODO: в БД пишем без КУ
                    doctor_day_plan.append(
                        [version, row['uid'], MODALITIES[mod_index], 'none', row['time_volume']]
                    )
            return row

        df['uid'] = None
        df['version'] = version
        df['doctor'] = None
        df['day_start'] = None
        df['availability'] = None
        df['time_volume'] = None
        df = df.apply(set_row, axis=1)
        df.drop(columns=['doctor_index', 'mods', 'day_index', 'time_volumes'])

        df_day = pd.DataFrame(doctor_day_plan, columns=[
            'version', 'doctor_availability', 'modality', 'contrast_enhancement', 'time_volume'])

        q = f"delete from {self.db_schema}.doctor_availability where version = '{version}'"
        q_day = f"delete from {self.db_schema}.doctor_day_plan where version = '{version}'"
        db = DB(self.db_schema)
        with db.get_cursor() as cursor:
            cursor.execute(q)
            cursor.execute(q_day)

        # t = Transaction(db)
        # t.set('upsert_with_cursor', df, 'doctor_availability', unique=['uid'])
        # t.set('upsert_with_cursor', df_day, 'doctor_day_plan',
        #       unique=['version', 'doctor_availability', 'modality', 'contrast_enhancement'])
        # t.call()
        db.upsert(df, 'doctor_availability', unique=['uid'])
        db.upsert(df_day, 'doctor_day_plan',
                  unique=['version', 'doctor_availability', 'modality', 'contrast_enhancement'])
        db.close()
        logger.info('Данные записаны.')

    def _get_doctors_mods(self):
        """
        Формирует вектор доступных модальностей врочей: [1, n_doctors, n_mods, 1] = 0, 1, 2,
            где 2 означает основную модальность, 1 - дополнительную, 0 - отсутствие модальности
        """
        self.doctor_df.reset_index(drop=True, inplace=True)
        # print(self.doctor_df.head())
        v_mod = np.zeros((len(self.doctor_df), self.n_mods), dtype=np.int32)
        mod_np = np.array(MODALITIES[:self.n_mods])

        def set_mod(row):
            row_index = row.name
            if len(row['modalities']) > 0:
                _, indices, _ = np.intersect1d(mod_np, row['modalities'], return_indices=True)
                v_mod[row_index][indices] = 1
            main_mod = np.where(mod_np == row['main_modality'])
            v_mod[row_index][main_mod] = 2

        # конвертируем строковое представление массива модальностей '{kt,mrt}' в массив
        remove_brackets = re.compile('[{}]')
        self.doctor_df['modalities'] = self.doctor_df['modalities'].apply(
            lambda m: [mod for mod in re.sub(remove_brackets, '', m).split(',')]
        ).apply(
            # убираем значение в виде пустой строки
            lambda m: m if len(m[0]) > 0 else []
        )

        self.doctor_df.apply(set_mod, axis=1)
        # print(f'v_mod:\n{v_mod[:10]}')
        return v_mod[np.newaxis, :, :, np.newaxis]

    def _generate_doctors_mods(self, bots_shape) -> np.ndarray:
        """
        Формирует вектор одной случайной модальности врачей из числа им доступных:
            [:, :, 1, :],
        Модальности формируются с распределением вероятностей, заданным в self.plan_mod_weights.
        """
        mod_weights = copy.deepcopy(self.plan_mod_weights)
        n_doctors = self.v_mod.shape[1]
        # print(f'mod_weights {mod_weights.shape} sum: {mod_weights.sum()}')

        # корректируем веса модальностей в соответствии с вектором текущего распределения
        if self.bot_mod_weights is not None:
            mod_weights /= self.bot_mod_weights / mod_weights
            # 0.03 / 0.01 = 3, 0.01 / 0.03 = 0.33,

        # формируем начальный вектор весов модальностей по каждому врачу
        weights = mod_weights.repeat(n_doctors, axis=1)
        # зануляем веса недоступных для врачей модальностей
        weights[self.v_mod == 0] = 0.
        # выравниваем веса до единицы и немного корректируем (иначе сумма может получиться >1)
        weights /= weights.sum(axis=2, keepdims=True)
        weights *= 0.9999

        # размножаем веса по ботам
        weights = weights.repeat(bots_shape[0], axis=0)
        # print(f'weights {weights.shape}:\n{weights[0, :15, :, 0]}')
        # print(f'weights > 1:\n{weights[weights.sum(axis=-1) > 1].sum(axis=-1)}')

        # проверки
        # for b in range(weights.shape[0]):
        #     for d in range(weights.shape[1]):
        #         s = weights[b][d].sum()
        #         for m in range(weights.shape[2]):
        #             pval = weights[b][d][m]
        #             if pval < 0 or pval > 1 or pval == np.nan:
        #                 print(f'[error] {s}: {weights[b][d]}')
        #                 break
        #         if s > 1:
        #             print(f'[>1] {s}: {weights[b][d]}')

        # размножаем веса по дням и переносим ось модальностей в конец
        weights = weights.repeat(bots_shape[3], axis=3).transpose((0, 1, 3, 2))

        # по одному эксперименту для каждого врача в каждом боте на каждый день
        mods_shape = bots_shape[:2] + (bots_shape[3],)
        n = np.ones(mods_shape, dtype=np.int32)
        # формируем вектор индексов случайных модальностей по каждому врачу в каждом боте
        mn = self.gen.multinomial(n, weights, size=mods_shape)
        rnd_index = np.argmax(mn, axis=-1, keepdims=True)
        # print(f'rnd_index {rnd_index.shape}:\n{rnd_index[0, :15, :, 0]}')
        return rnd_index.transpose((0, 1, 2, 3))

        """  прежняя версия - без учёта доступных модальностей
        # print(f'self.v_mod: {self.v_mod.shape}')
        mods_shape = bots_shape[:2] + (bots_shape[3],)
        random_mods = self.gen.random(size=mods_shape)  # (n_bots, n_doctors, n_days)
        # вектор распределения вероятностей с накоплением от 0 до 1 - добавляем ноль слева
        v_prop = np.cumsum(np.hstack([0., mod_weights[0, 0, :, 0]]))
        v_prop[self.n_mods] = 1.01  # на случай, если random выдаст ровно единицу (?)
        # print(f'v_prop: {v_prop}')
        random_mods = np.searchsorted(v_prop, random_mods) - 1

        return random_mods[:, :, np.newaxis, :]
        """

    def correct_doctors(self):
        """Минимизирует состав врачей с определёнными модальностями.
        Используется для проверки качества работы алгоритма.
        :return одномерный булевый вектор, отражающий включение/исключение врача
        """

        # доля исключаемых врачей по основной модальности
        corrections = [
            dict(mod=0, reduction=0.3),
            dict(mod=2, reduction=0.28),
        ]

        np.set_printoptions(formatter=dict(float=lambda x: f'{x:.2f}'))
        # pd.set_option('display.max_rows', 500)
        pd.set_option('display.max_columns', 500)
        pd.set_option('display.width', 1000)

        v_mod = self.v_mod
        # print(f'v_mod: {v_mod.shape}:\n{v_mod[0, :10, :, 0]}')
        n_doctors = v_mod.shape[1]
        doctor_df = copy.deepcopy(self.doctor_df)
        # добавляем колонку индекса основной модальности
        doctor_df["main_modality_index"] = doctor_df['main_modality'].apply(lambda m: MODALITIES.index(m))
        main_mod_index = doctor_df["main_modality_index"].to_numpy(dtype=np.int32)
        # print(f'main_mod_index ({len(main_mod_index)}):\n{main_mod_index}')
        schedule_type_index = doctor_df['schedule_type_index'].to_numpy(dtype=np.int32)
        # print(doctor_df.columns)
        ax = np.newaxis

        # получаем месячные объёмы работ по модальностям
        plan_volume = self.v_plan.sum(axis=3, keepdims=True) / SECONDS
        plan_volume_total = plan_volume.sum()
        print('\nКорректировка состава врачей...')
        print(f'plan_volume {plan_volume.shape}, hours.: {plan_volume[0, 0, :, 0]}, total: {plan_volume_total:.1f}')

        def compare_plan_vs_resources(avail, current_mask):
            """Рассчитывает суммарное рабочее время каждого врача за месяц, в соответствии
            с графиком базовой доступности, графиком работы и ставкой.

            :param avail: вектор достности врачей
            :param current_mask: одномерный вектор врачей для исключения их данных из
            avail и day_work_time
            :return: вектор рабочего времени [1, n_doctors, 1, 1]
                и соотношение ресурсы/план (число).
            """
            exclude_mask = ~current_mask
            # получаем месячный объём ресурсов //по модальностям
            avail[:, exclude_mask, :, :] = 0
            avail[avail < 1] = 0  # оставляем только рабочие дня базового графика
            # print(f'avail {avail.shape}: {avail[0, 0, 0]}')

            # print(f'time_rate {self.time_rate.shape}: *')
            # print(f'time_rate[:10]: {self.time_rate[0, :10, 0, 0]}')
            # print(f'avail[5]: {avail[0, 5, 0, :]}')
            day_work_time = np.where(schedule_type_index == 0, 8., 12.)[ax, :, ax, ax]
            day_work_time[:, exclude_mask, :, :] = 0
            work_time = day_work_time * self.time_rate * avail.sum(axis=3, keepdims=True)
            work_time_total = work_time.sum()
            over_resource = work_time_total / plan_volume_total
            print(f'work_time[:10] {work_time.shape}: {work_time[0,:10, 0, 0]}')
            print(f'work_time_total, hours: {work_time_total:.1f}, over_resource: {over_resource:.3f}')
            return work_time, over_resource

        def evaluate_resources(work_time):
            """Выводит отчёт об оценке ресурсов по основной и дополнительной модальностям"""
            # определение ресурсов по основной модальности
            doctor_df["work_time"] = pd.Series(work_time[0, :, 0, 0])
            # print(doctor_df[['main_modality', 'main_modality_index', 'work_time']].head(7))

            # Вариант 1. Сначала считаем ресурс по основной модальности, затем джойним дополнительные.
            #   Имеем дополнительную колонку с ресурсом по основной модальности.

            work_time_df = pd.DataFrame(doctor_df[['main_modality_index', 'work_time']]) \
                .groupby(['main_modality_index'], as_index=True).sum()
            # print(f'work_time_df:\n{work_time_df}\ntotal: {work_time_df["work_time"].sum()}')

            # считаем полный ресурс по каждой модальности отдельно
            extra_mod = np.where(v_mod == 1, work_time, 0.)
            # print(f'extra_mod: {extra_mod.shape}:\n{extra_mod[0, :10, :, 0]}')
            extra_mod_df = pd.DataFrame(extra_mod[0, :, :, 0])
            extra_mod_df['main_mod'] = pd.Series(main_mod_index)
            # print(f'extra_mod_df:\n{extra_mod_df.head()}')
            extra_mod_df = extra_mod_df.groupby(['main_mod'], as_index=True).sum()
            # print(f'extra_mod_df:\n{extra_mod_df}')

            work_time_df = work_time_df.join(extra_mod_df)
            print(f'work_time_df:\n{work_time_df}')

            # Вариант2. Расчёт ресурса и по основной и по дополнительной модальности сразу.

            # считаем полный ресурс по каждой модальности отдельно
            available_mod = np.where(v_mod > 0, work_time, 0.)
            # print(f'available_mod: {available_mod.shape}:\n{available_mod[0, :10, :, 0]}')
            available_mod_df = pd.DataFrame(available_mod[0, :, :, 0])
            available_mod_df['main_mod'] = pd.Series(main_mod_index)
            # print(f'extra_mod_df:\n{extra_mod_df.head()}')
            available_mod_df = available_mod_df.groupby(['main_mod'], as_index=True).sum()
            # print(f'available_mod_df:\n{available_mod_df}')
            # print(f'available_mod_df:\n{available_mod_df}')

        def correct(available, avail, correction):
            part = self.gen.random(size=(n_doctors,)) < correction['reduction']
            mask = (main_mod_index == correction['mod']) & part
            # print(f'mask:      {np.where(mask, 1, 0)[:30]}')
            available[mask] = 0
            print(f'available[:30]: {np.where(available, 1, 0)[:30]}')
            # print(f'part:      {np.where(part, 1, 0)[:30]}')
            # print()
            work_time, _ = compare_plan_vs_resources(avail, available)
            evaluate_resources(work_time)

        # смотрим, какой ресурс по врачам с одной модальностью
        print('\nВрачи с одной модальностью')
        single_mod = doctor_df['modalities'].apply(lambda m: len(m) == 0)
        avail = copy.deepcopy(self.v_base_avail)
        work_time, _ = compare_plan_vs_resources(avail, single_mod)
        evaluate_resources(work_time)

        # начальная оценка ресурсов
        print('\nНачальная оценка')
        available = np.ones(shape=v_mod.shape[1], dtype=bool)
        avail = copy.deepcopy(self.v_base_avail)
        work_time, _ = compare_plan_vs_resources(avail, available)
        evaluate_resources(work_time)

        for corr in corrections:
            print(f'\nКорректировка по модальности: {MODALITIES[corr["mod"]]}')
            correct(available, avail, corr)

        # # print(f'main_mod:  {main_mod_index[:30]}')
        # part = self.gen.random(size=(n_doctors,)) < mod_0_minus
        # mask = (main_mod_index == 0) & part
        # # print(f'mask:      {np.where(mask, 1, 0)[:30]}')
        # available[mask] = 0
        # print(f'available: {np.where(available, 1, 0)[:30]}')
        # # print(f'part:      {np.where(part, 1, 0)[:30]}')
        # print()
        # work_time, over_resource = compare_plan_vs_resources(~available)
        # evaluate_resources(work_time)
        #
        # part = self.gen.random(size=n_doctors) < mod_2_minus
        # mask = (main_mod_index == 2) & part
        # available[mask] = 0
        # print(f'available: {np.where(available, 1, 0)[:30]}')
        # print()
        # work_time, over_resource = compare_plan_vs_resources(~available)
        # evaluate_resources(work_time)

        return available

    def print_schedule(self, bot):
        # np.set_printoptions(formatter=dict(float=lambda x: f'{x:>3.0f}' if x != 0 else f'{"-":>3}'))
        bad_layout = ((bot != 0).sum(axis=1) > 1).sum()
        if bad_layout:
            raise ValueError(f'Найдены ошибки распределения по модальностям: {bad_layout}')
        for d in range(bot.shape[0]):
            mod_index = np.argmax(bot[d], axis=0)
            month_schedule = bot[d].sum(axis=0) / SECONDS
            print(''.join([f'{MODALITIES[i]:>5}' if h != 0 else f'{" ":>5}' for i, h in zip(mod_index, month_schedule)]))
            print(''.join([f'{h:>5.0f}' if h != 0 else f'{"-":>5}' for h in month_schedule]))



if __name__ == '__main__':

    logger.setup(level=logger.INFO, layout='debug')
    np.set_printoptions(edgeitems=30, linewidth=100000,
                        formatter=dict(float=lambda x: f'{x:.5f}'))

    # main_month_start = datetime(2024, 1, 1)
    # установим дату начала расчёта графика работы
    os.environ['ROENTGEN.SCHEDULE_START_DATE'] = '2024-05-01'

    mode = 'main'  # main, test (запись и чтение в тестовой БД - будет ошибка, если нет данных)
    if mode == 'test':  # ошибку выдаёт, day_plan_df некорректно формируется
        MODALITIES = ['kt', 'mrt']
        SECONDS = 1.
        n_generations = 10
        population_size = 4
        n_survived = 2
    else:
        n_generations = 1  # 10
        population_size = 100  # 100
        n_survived = 50  # 50

    # генерируем график доступности врачей
    start_of_month = datetime.fromisoformat(os.environ['ROENTGEN.SCHEDULE_START_DATE'])
    # load_doctor_availability('doctor_availability', start_of_month, version='base',
    #                          msg=f'Записано в БД строк графика работы врачей на период '
    #                              f'{start_of_month.strftime("%B %Y")}')

    main_scheduler = Scheduler(
        plan_version='forecast',
        n_generations=n_generations,
        population_size=population_size,
        n_survived=n_survived,
        mode=mode,
        correct_doctor_table=True,
    )
    # main_scheduler._generate_doctors_mods()
    # main_scheduler.populate(n_bots=main_scheduler.population_size)
    # main_scheduler.correct_doctors_table()
    main_scheduler.run(save=False)

    # schedule = dataloader.get_schedule('base', month_start, data_layer='day')
    # schedule = dataloader.get_schedule('base', month_start, data_layer='day_modality')
    # schedule = dataloader.get_schedule('base', month_start, data_layer='day_modality_ce')
    # print(schedule.head(20))
