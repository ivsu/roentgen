import uuid
import os
import gc
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import collections
import matplotlib.pyplot as plt
from matplotlib_inline.config import InlineBackend
from lib.dataloader import DataLoader, get_month_layout, MODALITIES, SCHEDULE_TYPES
from common.db import DB, Transaction
from common.logger import Logger

logger = Logger(__name__)
InlineBackend.figure_format = 'retina'

RANDOM_SCHEDULE_TYPE = False
SECONDS = 3600.  # количество секунд в часах (устанавливается в 1 для режима test)

SELECTION_METHOD = 'tournament'  # метод отбора популяции: best_bots, tournament
MATE_RATE = 0.01  # пропорция генов, которыми обмениваются родители при скрещивании
MUTATE_PROBABILITY = 0.2  # вероятносить применения мутации
MUTATE_RATE = 0.01  # количество случайных переставляемых генов при мутации

TMP_FILE_PATH = '/Users/ivan/Documents/CIFROPRO/Проекты/Нейронки/Расписание рентген-центра/'


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

    # def append(self, bots, bots_data=None, source=None, start_id=0, generation=None):
    #     self.bots = np.vstack([self.bots, bots]) if self.bots.size else bots
    #     if generation is not None:
    #         self.generation = generation
    #     if bots_data is None:
    #         ids = self._get_ids(bots.shape[0], start_id)
    #         bots_data = np.array([{'id': bot_id, 'source': source} for bot_id in ids])
    #
    #     self.bots_data = np.hstack([self.bots_data, bots_data]) if self.bots.size else bots_data

    def insert(self, bots, bots_data=None, scores=None, index=None, source=None, start_id=0, generation=None):

        if generation is not None:
            self.generation = generation

        if index is None:
            index = len(self.bots)
        assert index <= len(self.bots), f'Переданный индекс превышает количество имеющихся ботов: {index} / {len(self.bots)}'

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
        logger.info(f'bots shape: {self.bots.shape}')
        logger.info(f'bots_data shape: {self.bots_data.shape}')
        if mode == 'test':
            logger.info(f'bots:\n{self.bots}')


class Scheduler:
    """
    Класс представляет популяцию расписания работы врачей на месяц.
    Основной объект v_bot: [bot_id, doctor_id, modality, day_index] = time_volume
    Вектор модальностей v_mod: [1, doctor_id, modality, 1] = 0, 1, 2 (основная)
    Вектор доступности по базовому графику v_base_avail: [1, doctor_id, 1, day_index] = -1, 0, 1
    Вектор врачей v_doctor: [1, doctor_id, 1, props] = any (нормальная продолжительность рабочего дня, ...)
    """

    def __init__(self, month_start, plan_version, n_generations, population_size, n_survived, mode):
        """mode: main, test"""
        self.n_generations = n_generations
        self.population_size = population_size
        self.n_survived = n_survived
        self.mode = mode
        self.n_mods = len(MODALITIES)
        self.db_schema = 'test' if mode == 'test' else 'roentgen'
        self.dataloader = DataLoader(self.db_schema)

        # получаем таблицу врачей
        doctor_df = self.dataloader.get_doctors()
        doctor_df['schedule_type_index'] = doctor_df['schedule_type'].apply(lambda s: SCHEDULE_TYPES.index(s))
        self.doctor_df = doctor_df
        self.n_doctors = len(doctor_df)
        # self.v_doctor = doctor_df['schedule_type']
        self.time_rate = doctor_df['time_rate'].to_numpy(dtype=np.float32, copy=True)\
            [np.newaxis, :, np.newaxis, np.newaxis]

        # получаем вектор весов модальностей врачей: [1, num_doctors, self.n_mods, 1]
        self.v_mod = self._get_doctors_mods()

        self.month_layout = get_month_layout(month_start)

        # считываем план на текущий месяц по дням, врачам и модальностям (в секундах)
        day_plan_df = self.dataloader.get_day_plan(plan_version, month_start, with_ce=False)
        # print(day_plan_df.iloc[0])
        day_plan_df = day_plan_df[day_plan_df['month'] == month_start.month]
        row_index = {mod: i for i, mod in enumerate(MODALITIES)}
        day_plan_df = day_plan_df.pivot(
            index=['modality'], columns=['day_index'], values=['time_volume']
        ).rename(index=row_index)
        day_plan_df.sort_index(inplace=True)
        # print(day_plan_df.head(7))
        self.v_plan = day_plan_df.to_numpy(copy=True, dtype=np.float32)[np.newaxis, np.newaxis, :, :]
        self.v_plan *= SECONDS
        self.n_days = self.v_plan.shape[3]
        # print(f'v_plan:\n{self.v_plan[0, 0, :, :5]}')

        # получаем желаемый график работы (базовое расписание)
        base_schedule = self.dataloader.get_schedule('base', month_start, data_layer='day')
        self.v_base_avail = base_schedule \
            .pivot(index=['doctor_id'], columns=['day_index'], values=['availability']) \
            .to_numpy(dtype=np.int32, copy=True)[np.newaxis, :, np.newaxis, :]
        # обрезаем до количества дней в плане (актуально для режима test)
        self.v_base_avail = self.v_base_avail[..., :self.n_days]

        # вектора для генерации расписаний с разным видом графика
        def gen_workday_matrix(template, repeats):
            wdm = []
            wdm_source = np.tile(np.array(template, dtype=np.int32), repeats)
            for day in (range(len(template))):
                wdm.append(wdm_source[day:day + self.n_days])
            return np.array(wdm, dtype=np.int32)

        self.wd52 = gen_workday_matrix([0, 0, 1, 1, 1, 1, 1], 6)
        self.wd22 = gen_workday_matrix([0, 0, 1, 1], 9)

        self.gen = np.random.default_rng()

        # вектор популяции ботов
        self.v_bot = None
        # вектор рейтингов всех генов в текущей популяции
        self.v_score = None
        # итоговая оценка каждого бота
        self.bot_scores = None

        self.total_score = None
        self.best_bots = None

    def populate(self, n_bots, mod_weights=None):
        # self.doctor_df, self.month_layout, self.v_base_avail, self.mode
        # -> container: v_bot
        np.set_printoptions(formatter=dict(float=lambda x: f'{x:7.5f}'))
        # print(f'n_bots: {n_bots}')
        # print(f'n_doctors: {self.n_doctors}')

        work_days = np.zeros((n_bots, self.n_doctors, self.n_days), dtype=np.float32)

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
                .reshape(n_bots, index_len, self.n_days).astype(np.float32) * time_volume
            # print(f'work_days sample:\n{work_days[0, :10, :]}')

        # print(f'work_days {work_days.shape}')
        # создаём бота - добавляем ось модальностей
        v_bot = np.repeat(work_days[:, :, np.newaxis, :], repeats=self.n_mods, axis=2)
        # получаем вектор индексов случайной модальности
        v_random_mod_index = self._generate_doctors_mods(v_bot.shape, mod_weights)
        # print(f'v_random_mod_index: {v_random_mod_index.shape}')
        # TODO: отладка
        unique, counts = np.unique(v_random_mod_index, return_counts=True)
        # print(f'unique: {unique}')
        print(f'[populate] mods: {counts / counts.sum()}')
        # df = pd.DataFrame(v_random_mod_index)
        v_random_mod_index = v_random_mod_index.transpose((0, 1, 3, 2)).reshape((-1,))
        # v_bot
        v_bot_mod_mask = np.zeros(v_bot.shape).reshape((-1, self.n_mods))
        # print(f'v_bot_mod_mask: {v_bot_mod_mask.shape}')

        # advanced integer indexing
        v_bot_mod_mask[np.arange(v_bot_mod_mask.shape[0]), v_random_mod_index] = 1.
        v_bot_mod_mask = v_bot_mod_mask \
            .reshape((v_bot.shape[0], v_bot.shape[1], v_bot.shape[3], self.n_mods))\
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
        max_base_mismatch_k = 0.99 # максимальный штраф за несовпадение с базовым графиком

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
        """Выполняет мутацию ботов путём переключения модальностей врачей из числа им доступных"""
        # print(f'source bots {bots.shape}:\n{bots}')

        # вектор весов модальностей, отражающий выполнение ботами плана работ
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
        rnd_mod_indices = self._generate_doctors_mods(bots.shape, mod_weights).reshape((-1,))
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
        mutate_k = 0.1  # доля мутирующих генов
        mask = self.gen.random(size=bots.shape[:2]) < mutate_k
        bots[mask] = donors[mask]
        return bots

    def run(self, save=False):
        logger.info(f'v_mod: {self.v_mod.shape}')
        logger.info(f'v_plan: {self.v_plan.shape}')
        logger.info(f'v_base_avail: {self.v_base_avail.shape}')
        counter = {'diff': [], 'diff_minus': [], 'diff_plus': [], 'diff_weighted': []}
        # TODO: тест
        if self.mode != 'test':
            self.v_plan[:, :, 2] *= 1.43
            pass

        # вектор весов модальностей, отражающий распределение работ по модальностям в плане работ
        # используется для генерации ботов в populate
        mod_weights = self.v_plan.sum(axis=3, keepdims=True) / self.v_plan.sum()
        print(f'mod_weights: {mod_weights[0, 0, :, 0]}, sum: {mod_weights.sum()}')

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
            print(f'Соотношение: {dict(cnt)}')
            print('\n'.join(f'id: {d["id"]} {"[" + d["source"] + "]":>10}, score: {d["score"]:9.5f}'
                            f', schedule/plan by mods: {bots[i].sum(axis=(0, 2)) / self.v_plan.sum(axis=(0, 1, 3))}'
                            for i, d in enumerate(data)))

        random_bots = self.populate(n_bots=self.population_size, mod_weights=mod_weights)
        container = Container()
        container.insert(random_bots, source='populate', generation=0)
        container.print()

        # вычисляем оценку ботов
        bot_scores = self.evaluate(container, counter)
        # logger.info(f'bot_scores:\n{bot_scores}')
        container.set_scores(bot_scores)

        # производим отбор
        container = self.select(container, method=SELECTION_METHOD, generation=0)
        gc.collect()

        print(f'Первичное соотношение ресурсы/план:'
              f' {np.mean(random_bots.sum(axis=(1, 2, 3)) / self.v_plan.sum(axis=(1, 2, 3))):.4f}')

        # debug
        print_stat(container, 'best bots')

        for gn in range(self.n_generations):
            generation = gn + 1
            # print(f'Поколение #{generation}')
            # получаем ботов из контейнера - там остались лучшие
            bots, bots_data, best_bots_scores = container.unpack()

            # вектор использования врачей для зануления работ при переизбытке ресурсов
            v_used = None
            # over_resources = bots.sum(axis=(1, 2, 3)).mean() / self.v_plan.sum()
            """
            if over_resources > 1.:
                v_used = np.ones((1, bots.shape[1], 1, 1), dtype=np.float32)
                k = 1. - 1. / over_resources
                v_used[self.gen.random(size=v_used.shape) < k] = 0.
                print(f'v_used count: {v_used.sum(where=v_used > 0.)}')

                # применяем аналогичный вектор к первому поколению
                if generation == 0:
                    bots *= v_used
            """

            # вектор весов модальностей, отражающий выполнение плана работ текущей популяцией
            # - усредняем по имеющимся ботам для задания тренда новым создаваемым в populate
            # mod_weights = None
            # mod_weights = self.v_plan / bots.sum(axis=1, keepdims=True)
            # mod_weights = np.mean(mod_weights, axis=0)

            # формируем случайных ботов
            # with timer('populate'):
            random_bots = self.populate(n_bots=self.population_size, mod_weights=mod_weights)
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
            print_stat(container, 'evaluated bots', first=5)

            # формируем общий контейнер всех ботов: сначала лучшие, затем полученные в данном цикле
            container.insert(bots, bots_data, best_bots_scores, index=0)
            # print(f'next_container best_bots_scores: {next_container.best_bots_scores}')

            # производим отбор из всех ботов:
            container = self.select(container, method=SELECTION_METHOD, generation=generation)
            print_stat(container, 'best bots', first=5)

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

        print(f'\nbest bot: {best_bot_id} [{best_bot_source}], score: {best_bot_score:8.5f}')

        # m = 3600
        if self.mode == 'test':
            # m = 1
            logger.info(f'self.v_base_avail:\n{self.v_base_avail[0, :, 0, :]}')
            # logger.info(f'v_plan:\n{v_plan}')
            logger.info(f'best_bot:\n{best_bot}')
        np.set_printoptions(formatter=fmt_4_1)
        # logger.info(f'schedule:\n{schedule / SECONDS}')
        np.set_printoptions(formatter=fmt_6_1)
        logger.info(f'schedule:\n{v_schedule / SECONDS}')
        logger.info(f'v_plan:\n{v_plan / SECONDS}')
        logger.info(f'difference (schedule-plan):\n{diff / SECONDS}')
        np.set_printoptions(formatter=fmt_6_4)
        logger.info(f'difference (schedule/plan):\n{diff_k}')
        np.set_printoptions(formatter=fmt_6_1)
        logger.info(f'schedule by modalities:\n{v_schedule.sum(axis=1) / SECONDS}')
        logger.info(f'plan by modalities:\n{v_plan.sum(axis=1) / SECONDS}')
        logger.info(f'difference by modalities (schedule-plan):\n{diff.sum(axis=1) / SECONDS}')
        logger.info(f'difference by modalities (schedule/plan):\n{v_schedule.sum(axis=1) / v_plan.sum(axis=1)}')
        logger.info(f'difference total (schedule-plan): {diff.sum() / SECONDS:.1f}')
        logger.info(f'difference total (schedule/plan): {v_schedule.sum() / v_plan.sum():.3f}')

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

        t = Transaction(db)
        t.set('upsert_with_cursor', df, 'doctor_availability', unique=['uid'])
        t.set('upsert_with_cursor', df_day, 'doctor_day_plan',
              unique=['version', 'doctor_availability', 'modality', 'contrast_enhancement'])
        t.call()
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

        self.doctor_df.apply(set_mod, axis=1)
        # print(f'v_mod:\n{v_mod[:10]}')
        return v_mod[np.newaxis, :, :, np.newaxis]

    def _generate_doctors_mods(self, bots_shape, mod_weights) -> np.ndarray:
        """
        Формирует вектор одной случайной модальности врачей из числа им доступных:
            [:, :, 1, :],
        """
        # print(f'self.v_mod: {self.v_mod.shape}')
        mods_shape = bots_shape[:2] + (bots_shape[3],)
        random_mods = self.gen.random(size=mods_shape)
        # кумулятивный вектор распределения вероятностей
        v_prop = np.cumsum(np.hstack([0., mod_weights[0, 0, :, 0]]))
        v_prop[self.n_mods] = 1.01  # на случай, если random выдаст ровно единицу
        print(f'v_prop: {v_prop}')
        random_mods = np.searchsorted(v_prop, random_mods) - 1

        return random_mods[:, :, np.newaxis, :]


if __name__ == '__main__':
    os.chdir('..')
    logger.setup(level=logger.INFO, layout='debug')
    np.set_printoptions(edgeitems=30, linewidth=100000,
                        formatter=dict(float=lambda x: f'{x:.5f}'))

    main_month_start = datetime(2024, 1, 1)

    mode = 'main'
    if mode == 'test':
        MODALITIES = ['kt', 'mrt']
        SECONDS = 1.
        n_generations = 10
        population_size = 4
        n_survived = 2
    else:
        n_generations = 30
        population_size = 100
        n_survived = 40

    main_scheduler = Scheduler(
        main_month_start,
        plan_version='validation',
        n_generations=n_generations,
        population_size=population_size,
        n_survived=n_survived,
        mode=mode,
    )
    # main_scheduler._generate_doctors_mods()
    # main_scheduler.populate(n_bots=main_scheduler.population_size)
    main_scheduler.run(save=False)

    # schedule = dataloader.get_schedule('base', month_start, data_layer='day')
    # schedule = dataloader.get_schedule('base', month_start, data_layer='day_modality')
    # schedule = dataloader.get_schedule('base', month_start, data_layer='day_modality_ce')
    # print(schedule.head(20))
