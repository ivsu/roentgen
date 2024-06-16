import uuid

import numpy as np

import os
import gc
from datetime import datetime, timedelta
import calendar
import pandas as pd
import numpy as np
from lib.timeutils import time_to_interval, time_to_second
from lib.dataloader import get_day_plan, get_schedule, get_month_layout, \
    generate_doctor_month_schedule, \
    DOCTOR_COLUMNS, DB_SCHEMA, MODALITIES, SCHEDULE_TYPES
from lib.db import DB, Transaction, get_all

RANDOM_SCHEDULE_TYPE = False


def generate_bot(doctors, month_layout, base_schedule):

    columns = ['doctor_id', 'day_index', 'availability', 'modality', 'time_volume']

    month_start = month_layout['month_start']
    month_start_weekday= month_layout['month_start_weekday']
    month_end_weekday= month_layout['month_end_weekday']
    num_days = month_layout['num_days']
    schedule_template = month_layout['schedule_template']
    working_days_stack = month_layout['working_days_stack']

    assert len(doctors) * num_days == len(base_schedule)

    gen = np.random.default_rng()
    doctor_data = []

    time_table = []
    for i, doctor in doctors.iterrows():
        doctor_id = doctor['id']
        time_rate = doctor['time_rate']
        modalities = [doctor['main_modality']] + doctor['modalities']

        schedule_type = gen.choice(SCHEDULE_TYPES) if RANDOM_SCHEDULE_TYPE else doctor['schedule_type']

        # случайный день начала выходных
        weekend_start = gen.integers(0, 6, endpoint=True)
        working_days_masks = working_days_stack[schedule_type]

        month_schedule = generate_doctor_month_schedule(
            schedule_template, schedule_type, time_rate, working_days_masks, weekend_start
        )
        # преобразуем в одномерный вектор
        month_schedule.shape = (-1,)
        # оставляем только дни текущего месяца
        if month_end_weekday < 7:
            month_schedule = month_schedule[month_start_weekday - 1: -7 + month_end_weekday]
        else:
            month_schedule = month_schedule[month_start_weekday - 1:]

        for day_index in range(num_days):
            # флаг доступности: -1 - недоступен для распределения, 0 - выходной день, 1 - рабочий день
            try:
                availability = base_schedule.iloc[i * num_days + day_index]['availability']
            except Exception as e:
                print(f'i: {i}, day_index: {day_index}')
                raise
            # случайно выбираем модальность из доступных
            modality = MODALITIES.index(gen.choice(modalities))
            time_volume = timedelta(hours=month_schedule[day_index]).seconds
            time_table.append([doctor_id, day_index, availability, modality, time_volume])

        doctor_data.append([
            # нормальная продолжительность рабочего дня в зависимости от вида графика
            [8. * 3600 * time_rate, 12. * 3600 * time_rate][np.where(SCHEDULE_TYPES == schedule_type)[0][0]]
        ])

    # print(f'time_table:\n{time_table[:10]}')

    return pd.DataFrame(time_table, columns=columns), doctor_data


class Container:
    """Класс для согласованного хранения расписаний и связанных данных"""
    def __init__(self):
        self.bots = None
        self.avails = None
        self.doctors = None
        self.scores = None

    def keep(self, bots, avails, doctors):
        self.bots = bots
        self.avails = avails
        self.doctors = doctors

    def get(self, indices=None, copy=False):
        if indices is None:
            return self.bots, self.avails, self.doctors, self.scores
        elif copy:
            return self.bots[indices], self.avails[indices], self.doctors[indices], self.scores
        else:
            return self.bots[indices].copy(), self.avails[indices].copy(), \
                   self.doctors[indices].copy(), self.scores[indices].copy()

    def get_scores(self):
        return self.scores

    def set_scores(self, scores):
        self.scores = scores

    def append(self, bots, avails, doctors):
        # bots, avails, doctors, scores = container.get()
        self.bots = np.vstack([self.bots, bots])
        self.avails = np.vstack([self.avails, avails])
        self.doctors = np.vstack([self.doctors, doctors])
        # self.scores = np.vstack([self.scores, scores])

    def print_shapes(self):
        print(f'bots: {self.bots.shape}')
        print(f'avails: {self.avails.shape}')
        print(f'doctors: {self.doctors.shape}')


class Scheduler:

    """
    Класс представляет популяцию расписания работы врачей на месяц.
    Основной объект v_bot - матрица: (bot_id, doctor_id, day_index, modality) = time_volume
    Вектор весов модальностей v_mod - main_modality: (bot_id, doctor_id, modality) = 1.0 / 0.8 / 0.0
    Вектор доступности v_avail - (bot_id, doctor_id, day_index) = -1, 0, 1
    Вектор врачей v_doctor - (bot_id, doctor_id, props) = any
    """

    def __init__(self, month_start, plan_version, n_generations, population_size, n_survived):
        self.n_generations = n_generations
        self.population_size = population_size
        self.n_survived = n_survived

        # получаем таблицу врачей
        db = DB()
        q = (f"SELECT id, {', '.join(DOCTOR_COLUMNS)}"
             f" FROM {DB_SCHEMA}.doctor"
             f" WHERE is_active"
             f" ORDER BY id"
             )
        with db.get_cursor() as cursor:
            cursor.execute(q)
            doctors: pd.DataFrame = get_all(cursor)
        self.doctors = doctors

        self.base_schedule = get_schedule('base', month_start, data_layer='day')

        # создание вектора весов модальностей врачей: (num_doctors, num_mods)
        mod_np = np.array(MODALITIES)
        mod_template = np.zeros((len(MODALITIES)), dtype=np.int32)
        mod_list = []

        def set_mod(row):
            mod = mod_template.copy()
            if len(row['modalities']) > 0:
                _, indices, _ = np.intersect1d(mod_np, row['modalities'], return_indices=True)
                mod[indices] = 1
            main_mod = np.where(mod_np == row['main_modality'])
            mod[main_mod] = 2
            mod_list.append(mod)

        doctors.apply(set_mod, axis=1)
        v_mod = np.array(mod_list)[np.newaxis, :, :, np.newaxis]
        print(f'v_mod: {v_mod.shape}')
        # print(v_mod[:5])
        self.v_mod = v_mod
        self.v_avail = None # сейчас не используется в алгоритме
        self.v_bot = None
        self.v_doctor = None

        self.month_layout = get_month_layout(month_start)
        # считываем план на текущий месяц по дням, врачам и модальностям (в секундах)
        day_plan_df = get_day_plan(plan_version, month_start, with_ce=False)
        # print(day_plan_df.iloc[0])
        day_plan_df = day_plan_df[day_plan_df['month'] == month_start.month]
        row_index = {mod: i for i, mod in enumerate(MODALITIES)}
        day_plan_df = day_plan_df.pivot(
            columns=['day_index'], index=['modality'], values=['time_volume']
        ).rename(index=row_index)
        day_plan_df.sort_index(inplace=True)
        # print(day_plan_df.head(7))
        self.v_plan = np.expand_dims(day_plan_df.to_numpy(copy=True, dtype=np.float32),
                                     axis=(0, 1)) * 3600
        print(f'v_plan: {self.v_plan.shape}')
        # print(f'v_plan:\n{self.v_plan[0, 0, :, :5]}')

        # вектор рейтингов всех генов в текущей популяции
        self.v_score = None
        # итоговая оценка каждого бота
        self.bot_scores = None

        self.total_score = None
        self.best_bots = None

    def initialize(self, n_bots):
        if n_bots == 0:
            raise RuntimeError('Получено нулевое количество ботов для инициализации.')

        np.set_printoptions(formatter=dict(float=lambda x: f'{x:.1f}'))
        num_mods = len(MODALITIES)
        bot_list = []
        avail_list = []
        v_doctor = None
        for bot_index in range(n_bots):

            # выход: [doctor_id, day_index, availability, modality, time_volume]
            src_df, doctor_data = generate_bot(self.doctors, self.month_layout, self.base_schedule)
            # src_df['time_volume'] = src_df['time_volume'].apply(lambda s: s.seconds)
            # print(f'src_df:\n{src_df.head(10)}')
            # print(src_df.iloc[0])
            doctor_np = np.array(doctor_data)[np.newaxis, :]
            v_doctor = doctor_np if v_doctor is None else np.vstack([v_doctor, doctor_np])

            base_df = src_df.pivot(columns=['day_index'], index=['doctor_id'], values=['time_volume'])
            # print(f'base_df:\n{base_df.head()}')
            bot = base_df.to_numpy(copy=True, dtype=np.float32)
            # раскрываем расписание врача по модальностям: (num_doctors, num_mods, num_days)
            bot = np.repeat(bot[:, np.newaxis, :], num_mods, axis=1)
            # print(f'bot: {bot.shape}')
            # print(f'bot:\n{bot[:2, 0, :]}')

            # создаём маску для зануления незадействованных модальностей
            # (модальности заданы на все дни - рабочие, нерабочие)
            mod_df = src_df.pivot(columns=['day_index'], index=['doctor_id'], values=['modality'])
            mod_index = mod_df.to_numpy(dtype=np.int32)
            # print('mod_index:', mod_index.shape)
            # print(f'mod_index:\n{mod_index[:2]}')
            mod_index.shape = (-1,)
            # print('mod_index:', mod_index.shape)

            mod_mask = np.zeros(bot.shape).reshape((-1, num_mods))
            # print('mod_mask:', mod_mask.shape)

            # advanced integer indexing
            mod_mask[np.arange(mod_mask.shape[0]), mod_index] = 1.
            mod_mask = mod_mask.reshape((bot.shape[0], bot.shape[2], num_mods)).transpose((0, 2, 1))
            # print('mod_mask:', mod_mask.shape)
            # print(f'mod_mask:\n{mod_mask[0]}')

            # оставляем рабочее время только на заданной модальности
            bot *= mod_mask
            print(f'bot works:\n{bot.sum(axis=1)[:3]}')
            bot_list.append(bot)
            # for i in [0, 2]:
            #     print(f'doctor: {i}')
            #     for j in range(num_mods):
            #         print(f'mod: {j}')
            #         print(bot[i, j])
            # print(f'bot:\n{bot[0:num_mods:2]}')

            avail_df = src_df.pivot(columns=['day_index'], index=['doctor_id'], values=['availability'])
            avail_list.append(avail_df.to_numpy(dtype=np.float32, copy=True))
            # print(avail_df.head())

        container = Container()
        container.keep(
            np.array(bot_list),
            np.array(avail_list),
            v_doctor
        )
        return container

    def evaluate(self, container: Container):

        time_pw = -10  # степень функции для оценки нормы рабочего времени в рамках дня
        extra_mod_k = 0.9  # снижение оценки за работы не по основной модальности
        fact_over_plan_pw = -2  # степень затухания оценки при превышении факта над планом

        v_bot, _, v_doctor,_ = container.get()

        np.set_printoptions(formatter=dict(float=lambda x: f'{x:.3f}'))
        # TODO: debug
        # v_bot[0, 0, 0, 0:5] = 20000.
        # v_bot[0, 0, 0, 6:8] = 1000.
        # v_bot[0, 0, 2, 0:5] = 10000.
        # v_bot[0, 2, 0, 3:5] = 50000.
        # v_bot[0, 2, 1, 3:5] = 10000.

        f_mod = self.v_mod > 0
        f_extra_mod = self.v_mod == 1  # 1 - дополнительная модальность, 2 - основная
        f_noworks = np.isclose(v_bot, 0.)
        # 0 - норма времени (сек.) согласно виду графика и ставке
        f_day_time_norm = v_doctor[:, :, 0, np.newaxis, np.newaxis]
        print(f'f_day_time_norm: {f_day_time_norm.shape}')
        # print(f'f_day_time_norm:\n{f_day_time_norm[0, :3, 0, 0]}')

        def power_zero_and_norm(value, normal, power):
            # степенная функция, равная 1 в нуле и в значении нормы времени
            return np.power(value / normal * 2 - 1, power)

        def power_norm_and_more(value, normal, power):
            # степенная функция, равная 1 в значении нормы времени и затем затухающая
            # return np.power(-2. + value / normal, power)
            # степенная функция, равная 1 в значении нормы времени и затем затухающая
            # при отрицательной степени
            mask = value > 0
            out = value.copy()
            # print(f'out: {type(out)}, shape: {out.shape}')
            np.power(value / normal, power, out=out, where=mask)
            return out

        def linear_zero_and_norm(value, normal):
            # линейная функция, равная 0 в нуле и 1 в значении нормы времени
            return value / normal

        v_score = np.zeros(v_bot.shape)
        # *
        # оценка отдельного рабочего дня врача в рамках модальности:

        # если работ нет - ставим 1
        # TODO: возможно, следует занулить
        v_score[f_noworks & f_mod] = 0.
        # есть работы по своим модальностям
        f_work_frame = ~f_noworks & f_mod
        print(f'f_work_frame: {f_work_frame.shape}')
        # отбираем работы со временем меньше нормативного
        f_in_norm = v_bot <= f_day_time_norm
        # оценку 1 получают работы по модальности не превышающие нормативное время
        v_score[f_work_frame & f_in_norm] = 1.
        # работы по модальности со временем больше нормативного получают убывающую оценку
        v_score[f_work_frame & ~f_in_norm] = power_norm_and_more(
            v_bot, f_day_time_norm, time_pw)[f_work_frame & ~f_in_norm]
        print(f'v_score: {v_score.shape}')

        # немного снижаем оценку за работы по дополнительной модальности
        v_score[f_work_frame & f_extra_mod] *= extra_mod_k

        # *
        # добавляем оценку продолжительности рабочего дня (суммируем по модальностям)
        # работы близкие к нормативному времени получают оценку близкую к 1
        # работы более нормативного времени - убывающую оценку
        # work_day_duration = np.sum(self.v_bot, axis=2, keepdims=True)

        # td = 6
        # print(f':: f_work_frame: {f_work_frame[0, 0, 0, td]}')
        # print(f':: work_day_duration: {work_day_duration[0, 0, 0, td]}')
        # print(f':: f_day_time_norm: {f_day_time_norm[0, 0, 0, 0]}')
        # print(f':: self.v_bot: {self.v_bot[0, 0, 0, td]}')

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

        # TODO: оценка соответствия исходному графику

        # *
        # добавляем оценку выполнения плана
        # оценка линейно растёт при приближении факта к плану
        # и убывает при перевыполнении
        day_fact = np.sum(v_bot, axis=1, keepdims=True)
        print(f'day_fact: {day_fact.shape}')
        in_plan = day_fact <= self.v_plan
        print(f'in_plan: {in_plan.shape}')
        multiplier = np.ones(in_plan.shape)

        # td = 3
        # print(f':: day_fact: {day_fact[0, 0, 0, td]}')
        # print(f':: self.v_plan: {self.v_plan[0, 0, 0, td]}')
        # print(f':: power_norm_and_more: {power_norm_and_more(day_fact[0, 0, 0, td], self.v_plan[0, 0, 0, td], fact_over_plan_pw)}')

        print(f'multiplier: {multiplier.shape}')
        multiplier[in_plan] = linear_zero_and_norm(day_fact, self.v_plan)[in_plan]
        multiplier[~in_plan] = power_norm_and_more(
            day_fact, self.v_plan, fact_over_plan_pw
        )[~in_plan]
        print(f'multiplier:\n{multiplier[0, 0, :, :10]}')
        v_score *= multiplier
        # итоговая оценка
        # TODO: настроить
        bot_scores = v_score.sum(axis=(1, 2, 3))
        print('total_score:', bot_scores)

        # check = self.v_plan
        # print('Проверка плана ...')
        # for b in range(check.shape[0]):
        #     for m in range(check.shape[2]):
        #         for day in range(check.shape[3]):
        #             if check[b, 0, m, day] == 0:
        #                 print(f'bot: {b}, mod: {m}, day: {day}')
        return bot_scores

    def select(self, container: Container) -> Container:
        bot_scores = container.get_scores()
        best_indices = bot_scores.argsort()[-self.n_survived:]
        best_bots, avails, doctors, scores = container.get(best_indices, copy=True)
        # кладём лучших ботов в другой контейнер
        selected = Container()
        selected.keep(best_bots, avails, doctors)
        selected.set_scores(scores)
        container = None
        gc.collect()
        return selected

    def mate(self, bots):
        assert self.n_survived % 2 == 0
        np.set_printoptions(formatter=dict(float=lambda x: f'{x:.1f}'))
        n_best_bots = bots.shape[0]
        gen = np.random.default_rng()

        # определяем, кто с кем будет скрещиватья
        mate_indices = np.arange(n_best_bots)
        gen.shuffle(mate_indices)
        print(f'mate_indices:\n{mate_indices}')

        # *
        # получаем вектор случайного скрещивания для всех пар родителей
        v_mate_len = self.n_survived // 2
        v_mate = np.zeros((v_mate_len, bots.shape[1], bots.shape[3]), dtype=np.int32)

        # добавляем случайные индексы второго родителя
        v_mate.shape = (v_mate_len, -1)
        for i in range(v_mate.shape[0]):
            indices = np.unique(gen.integers(0, v_mate.shape[1], size=v_mate.shape[1] // 2))
            v_mate[i][indices] = 1
        v_mate.shape = (v_mate_len, bots.shape[1], bots.shape[3])
        new_bots = None
        print(f'v_mate {v_mate.shape}:\n{v_mate}')

        # производим мутацию
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

                n_mods = len(MODALITIES)
                v_mod = self.v_mod[0, :, :, :]
                v_mod = np.repeat(v_mod, bot0.shape[2], axis=-1).transpose((0, 2, 1))
                v_mod = v_mod.reshape((-1, n_mods))
                mod_allowed = v_mod[np.arange(v_mod.shape[0]), mod_index0.reshape(-1,)]
                mod_allowed.shape = mod_index0.shape
                print(f'mod_allowed {mod_allowed.shape}:\n{mod_allowed}')

                # оставляем модальности согласно вектору случайного скрещивания
                mask = mask & v_mate[mate_index] == 1
                print(f'mask:\n{mask}')
                # обмен индексами
                new_index0[mask] = mod_index1[mask]
                new_index1[mask] = mod_index0[mask]
                # перезапись модальнойстей ботов
                print(f'mod_index0:\n{mod_index0}')
                print(f'new_index0:\n{new_index0}')
                print(f'mod_index1:\n{mod_index1}')
                print(f'new_index1:\n{new_index1}')
                mod_index0.shape = (-1,)
                mod_index1.shape = (-1,)
                new_index0.shape = (-1,)
                new_index1.shape = (-1,)

                v_len = new_index0.shape[0]

                def prepare_for_advanced_indexing(bot):
                    """что-то не так с contiguous - создаётся новый объект"""
                    return bot.transpose((0, 2, 1)).reshape((-1, len(MODALITIES)))

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

            # mate_pair(b1, new_bot0)
            # mate_pair(b0, new_bot1)
            new_bot0, new_bot1 = change_modalities(new_bot0, new_bot1)
            print(f'new_bot0:\n{new_bot0}')
            print(f'new_bot1:\n{new_bot1}')
            gc.collect()

            print(f'b0 / new_bot0 sum: {b0.sum()} / {new_bot0.sum()}')
            print(f'b1 / new_bot1 sum: {b1.sum()} / {new_bot1.sum()}')

            # print(f'new_bot0: {new_bot0.shape}')
            # print(f'new_bot1: {new_bot1.shape}')
            new_pair = np.vstack([new_bot0[np.newaxis, :, :, :], new_bot1[np.newaxis, :, :, :]])
            if new_bots is None:
                new_bots = new_pair
                print(f'new_bots: {new_bots.shape}')
            else:
                new_bots = np.vstack([new_bots, new_pair])

            print(f'mate_indices:\n{mate_indices[i:i + 2]}')
            # print(f'new_pair[0]:\n{new_pair[0][0, :10]}')
            # print(f'new_pair[1]:\n{new_pair[1][0, :10]}')

        # print(f'v_mod:\n{self.v_mod[0, :3]}')
        print(f'new_bots: {new_bots.shape}')
        return new_bots

    def mutate(self, bots):
        pass
        return bots

    def run(self):

        container = self.initialize(n_bots=self.population_size)
        container.print_shapes()

        # вычисляем оценку ботов
        bot_scores = self.evaluate(container)
        container.set_scores(bot_scores)

        # производим отбор
        container = self.select(container)
        gc.collect()

        for generation in range(self.n_generations - 1):
            # print(f'Поколение #{generation}')
            # скрещиваем ботов
            bots, avails, doctors, scores = container.get()
            new_bots = self.mate(bots)
            # производим мутацию
            new_bots = self.mutate(new_bots)

            # добавляем случайных ботов
            next_container = self.initialize(
                n_bots=self.population_size - self.n_survived
            )
            # доступность врачей и график их работы не меняются
            next_container.append(new_bots, avails.copy(), doctors.copy())

            # рассчитываем оценку новых ботов
            next_scores = self.evaluate(next_container)

            # формируем общий контейнер всех ботов: рассчитанных в данном цикле и стека лучших
            next_container.append(bots, avails, doctors)
            next_container.set_scores(next_scores + scores)

            # производим отбор из всех ботов:
            container = self.select(next_container)
            gc.collect()

        # пишем лучшего бота в базу
        # TODO: нужно также сохранять доступность и норму времени (длительность рабочего дня) - ок
        best_bots, avails, _, _ = container.get([0])
        self.save_bot(best_bots[0], avails[0])

    def save_bot(self, bot, avail):

        q = f"""
            select id, uid, day_start_time, schedule_type, time_rate,
            row_number() over (order by id) - 1 as row_index 
            from {DB_SCHEMA}.doctor order by id;
        """
        db = DB()
        with db.get_cursor() as cursor:
            cursor.execute(q)
            doctors = get_all(cursor)
            doctors = doctors.set_index('row_index').T.to_dict()
        db.close()

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
            row['availability'] = avail[doctor_index][row['day_index']]

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
                        [version, row['uid'], MODALITIES[mod_index], 'none', row['time_volume']])
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

        q = f"delete from {DB_SCHEMA}.doctor_availability where version = '{version}'"
        q_day = f"delete from {DB_SCHEMA}.doctor_day_plan where version = '{version}'"
        with db.get_cursor() as cursor:
            cursor.execute(q)
            cursor.execute(q_day)

        t = Transaction(db)
        t.set('upsert_with_cursor', df, 'doctor_availability', unique=['uid'])
        t.set('upsert_with_cursor', df_day, 'doctor_day_plan',
              unique=['version', 'doctor_availability', 'modality', 'contrast_enhancement'])
        t.call()
        db.close()
        print('Данные записаны')


class Test:

    def __init__(self, month_start):
        self.month_start = month_start

    def get_bots(self):
        gen = np.random.default_rng()
        bots = np.arange(144)
        zero_indices = gen.integers(0, len(bots), size=len(bots) // 2)
        bots[zero_indices] = 0
        gen.shuffle(bots)
        bots.shape = (2, 3, len(MODALITIES), 4)

        return bots

    def run(self):
        # self.n_generations = n_generations
        # self.population_size = population_size
        # self.n_survived = n_survived

        scheduler = Scheduler(
            self.month_start,
            plan_version='validation',
            n_generations=2,
            population_size=3,
            n_survived=2,
        )
        bots = self.get_bots()
        v_mod = np.array([
            [1, 0, 1, 2, 1, 1],
            [0, 2, 0, 1, 1, 0],
            [2, 1, 1, 0, 0, 1]
        ])

        mask = np.ones(bots.shape, dtype=np.bool_)
        mask = mask & v_mod[np.newaxis, :, :, np.newaxis] == 0
        bots[mask] = 0
        scheduler.v_mod = v_mod[np.newaxis, :, :, np.newaxis]

        print(f'initial bots:\n{bots}')
        scheduler.mate(bots)
        # print(f'bot0:\n{bots[0]}')
        # print(f'bot1:\n{bots[1]}')


if __name__ == '__main__':
    os.chdir('..')
    np.set_printoptions(edgeitems=30, linewidth=100000,
                        formatter=dict(float=lambda x: f'{x:.5f}'))
    month_start = datetime(2024, 1, 1)

    # test = Test(month_start)
    # test.run()

    scheduler = Scheduler(
        month_start,
        plan_version='validation',
        n_generations=2,
        population_size=3,
        n_survived=2,
    )
    scheduler.run()

    # schedule = get_schedule('base', month_start, data_layer='day')
    # schedule = get_schedule('base', month_start, data_layer='day_modality')
    # schedule = get_schedule('base', month_start, data_layer='day_modality_ce')
    # print(schedule.head(20))
