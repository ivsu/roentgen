import uuid
import os
import gc
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from lib.dataloader import get_day_plan, get_schedule, get_month_layout, \
    generate_doctor_month_schedule, get_weekday, \
    DOCTOR_COLUMNS, DB_SCHEMA, MODALITIES, SCHEDULE_TYPES
from lib.db import DB, Transaction, get_all
from lib.timeutils import timer
from common.logger import Logger

logger = Logger(__name__)

RANDOM_SCHEDULE_TYPE = False


def generate_bot(doctors, month_layout, v_avail, mode):
    columns = ['doctor_id', 'day_index', 'availability', 'modality', 'time_volume']

    # month_start = month_layout['month_start']
    month_start_weekday = month_layout['month_start_weekday']
    month_end_weekday = month_layout['month_end_weekday']
    num_days = month_layout['num_days']
    schedule_template = month_layout['schedule_template']
    working_days_matrices = month_layout['working_days_matrices']
    workday_seconds = [8., 12.] if mode == 'test' else [8. * 3600, 12. * 3600]
    assert len(doctors) * num_days == v_avail.shape[0] * v_avail.shape[1]

    gen = np.random.default_rng()
    doctor_data = []

    time_table = []
    for i, doctor in doctors.iterrows():
        doctor_id = doctor['id']
        time_rate = doctor['time_rate']
        modalities = [doctor['main_modality']] + doctor['modalities']

        schedule_type = gen.choice(SCHEDULE_TYPES) if RANDOM_SCHEDULE_TYPE else doctor['schedule_type']
        day_norm = time_rate * workday_seconds[np.where(SCHEDULE_TYPES == schedule_type)[0][0]]

        # случайный день начала выходных
        weekend_start = gen.integers(0, 6, endpoint=True)
        working_days_matrix = working_days_matrices[schedule_type]

        month_schedule = generate_doctor_month_schedule(
            schedule_template, schedule_type, time_rate, working_days_matrix, weekend_start
        )
        # print(f'month_schedule:\n{month_schedule}')
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
                availability = v_avail[i][day_index]  # iloc[i * num_days + day_index]['availability']
            except Exception as e:
                logger.error(f'i: {i}, day_index: {day_index}, {repr(e)}')
                raise
            # случайно выбираем модальность из доступных
            modality = MODALITIES.index(gen.choice(modalities))
            # зануляем рабочее время в недоступные дни
            time_volume = 0.
            if availability != -1:
                time_volume = timedelta(hours=month_schedule[day_index]).seconds
            if mode == 'test':
                time_volume = time_volume / 3600
            time_table.append([doctor_id, day_index, availability, modality, time_volume])

        # нормальная продолжительность рабочего дня в зависимости от вида графика
        doctor_data.append([day_norm])

    # print(f'time_table:\n{time_table[:10]}')

    return pd.DataFrame(time_table, columns=columns), doctor_data


class Container:
    """Класс для согласованного хранения расписаний и связанных данных"""

    def __init__(self):
        self.bots = None
        # self.avails = None
        self.doctors = None
        self.scores = None

    def keep(self, bots, doctors):
        self.bots = bots
        # self.avails = avails
        self.doctors = doctors

    def unpack(self, copy=False):
        if copy:
            return self.bots.copy(), self.doctors.copy(), self.scores.copy()
        else:
            return self.bots, self.doctors, self.scores

    def unpack_best(self, num, copy=False):

        # получаем индексы лучших ботов, отсортированные в порядке убывания оценки
        best_indices = np.argpartition(self.scores, -num)[-num:]
        index = np.flip(best_indices[np.argsort(self.scores[best_indices])])

        if copy:
            return self.bots[index].copy(), self.doctors[index].copy(), self.scores[index].copy()
        else:
            return self.bots[index], self.doctors[index], self.scores[index]

    def get_scores(self):
        return self.scores

    def set_scores(self, scores):
        self.scores = scores

    def append(self, bots, doctors):
        self.bots = np.vstack([self.bots, bots])
        # self.avails = np.vstack([self.avails, avails])
        self.doctors = np.vstack([self.doctors, doctors])

    def print_shapes(self):
        logger.info(f'bots: {self.bots.shape}')
        # print(f'avails: {self.avails.shape}')
        logger.info(f'doctors: {self.doctors.shape}')


class Scheduler:
    """
    Класс представляет популяцию расписания работы врачей на месяц.
    Основной объект v_bot - матрица: (bot_id, doctor_id, day_index, modality) = time_volume
    Вектор весов модальностей v_mod - main_modality: (bot_id, doctor_id, modality) = 1.0 / 0.8 / 0.0
    Вектор доступности v_base_avail - (bot_id, doctor_id, day_index) = -1, 0, 1
    Вектор врачей v_doctor - (bot_id, doctor_id, props) = any
    """

    def __init__(self, month_start, plan_version, n_generations, population_size, n_survived, mode):
        """mode: main, test"""
        self.n_generations = n_generations
        self.population_size = population_size
        self.n_survived = n_survived
        self.mode = mode
        self.n_mods = len(MODALITIES)

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

        base_schedule = get_schedule('base', month_start, data_layer='day')
        self.v_base_avail = base_schedule.pivot(index=['doctor_id'], columns=['day_index'], values=['availability'])
        self.v_base_avail = self.v_base_avail.to_numpy(dtype=np.int32, copy=True)

        # создание вектора весов модальностей врачей: (num_doctors, self.n_mods)
        mod_np = np.array(MODALITIES[:self.n_mods])
        mod_template = np.zeros((self.n_mods,), dtype=np.int32)
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
        # print(v_mod[:5])
        self.v_mod = v_mod
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
        # print(f'v_plan:\n{self.v_plan[0, 0, :, :5]}')

        self.gen = np.random.default_rng()

        # вектор рейтингов всех генов в текущей популяции
        self.v_score = None
        # итоговая оценка каждого бота
        self.bot_scores = None

        self.total_score = None
        self.best_bots = None

    def initialize(self, n_bots):
        if n_bots == 0:
            raise RuntimeError('Получено нулевое количество ботов для инициализации.')

        np.set_printoptions(formatter=dict(float=lambda x: f'{x:.0f}'))
        bot_list = []
        v_doctor = None
        for bot_index in range(n_bots):
            # выход: [doctor_id, day_index, availability, modality, time_volume]
            src_df, doctor_data = generate_bot(self.doctors, self.month_layout, self.v_base_avail, self.mode)
            # src_df['time_volume'] = src_df['time_volume'].apply(lambda s: s.seconds)
            # print(f'src_df:\n{src_df.head(10)}')
            # print(src_df.iloc[0])
            doctor_np = np.array(doctor_data)[np.newaxis, :]
            v_doctor = doctor_np if v_doctor is None else np.vstack([v_doctor, doctor_np])

            base_df = src_df.pivot(columns=['day_index'], index=['doctor_id'], values=['time_volume'])
            # print(f'base_df:\n{base_df.head()}')
            bot = base_df.to_numpy(copy=True, dtype=np.float32)
            # раскрываем расписание врача по модальностям: (num_doctors, self.n_mods, num_days)
            bot = np.repeat(bot[:, np.newaxis, :], self.n_mods, axis=1)
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

            mod_mask = np.zeros(bot.shape).reshape((-1, self.n_mods))
            # print('mod_mask:', mod_mask.shape)

            # advanced integer indexing
            mod_mask[np.arange(mod_mask.shape[0]), mod_index] = 1.
            mod_mask = mod_mask.reshape((bot.shape[0], bot.shape[2], self.n_mods)).transpose((0, 2, 1))
            # print('mod_mask:', mod_mask.shape)
            # print(f'mod_mask:\n{mod_mask[0]}')

            # оставляем рабочее время только на заданной модальности
            bot *= mod_mask
            logger.debug(f'bot[{bot_index}] works:\n{bot.sum(axis=1)[:3]}')
            bot_list.append(bot)
            # for i in [0, 2]:
            #     print(f'doctor: {i}')
            #     for j in range(self.n_mods):
            #         print(f'mod: {j}')
            #         print(bot[i, j])
            # print(f'bot:\n{bot[0:self.n_mods:2]}')

            # avail_df = src_df.pivot(columns=['day_index'], index=['doctor_id'], values=['availability'])
            # avail_list.append(avail_df.to_numpy(dtype=np.float32, copy=True))
            # print(avail_df.head())

        container = Container()
        bots = np.array(bot_list)
        # logger.info(f'initial bots schedule by modalities {bots.shape}:\n{bots.sum(axis=(1, 3)) / 3600}')

        # logger.info(f'bots schedule:\n{bots.sum(axis=2)}')
        container.keep(
            bots,
            # self.v_base_avail,
            v_doctor
        )
        return container

    def evaluate(self, container: Container):
        """
        ...
        Работы v_bot занулены в дни недоступности (self.v_base_avail = -1)

        :param container:
        :return:
        """

        # time_pw = -10  # степень функции для оценки нормы рабочего времени в рамках дня
        fact_over_plan_pw = -1  # степень затухания оценки при превышении факта над планом
        extra_mod_k = 0.98  # снижение оценки за работы не по основной модальности
        conflict_with_base_k = 0.98  # снижение оценки за несовпадение с базовым графиком

        v_bot, v_doctor, _ = container.unpack()

        np.set_printoptions(formatter=dict(float=lambda x: f'{x:.3f}'))

        # TODO: debug
        # v_bot[0, 0, 0, 0:5] = 20000.
        # v_bot[0, 0, 0, 6:8] = 1000.
        # v_bot[0, 0, 2, 0:5] = 10000.
        # v_bot[0, 2, 0, 3:5] = 50000.
        # v_bot[0, 2, 1, 3:5] = 10000.

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
        f_not_working_days_in_base = self.v_base_avail[np.newaxis, :, np.newaxis, :] < 1
        f_not_working_days_in_base = f_not_working_days_in_base \
            .repeat(self.population_size, axis=0) \
            .repeat(self.n_mods, axis=2)
        # 0 - норма времени (сек.) согласно виду графика и ставке
        # f_day_time_norm = v_doctor[:, :, 0, np.newaxis, np.newaxis]
        # print(f'f_day_time_norm: {f_day_time_norm.shape}')
        # print(f'f_day_time_norm:\n{f_day_time_norm[0, :3, 0, 0]}')

        v_score = np.zeros(v_bot.shape)
        logger.debug(f'v_score: {v_score.shape}')

        # *
        # оценка отдельного рабочего дня врача в рамках модальности:
        # logger.info(f'v_bot:\n{v_bot}')

        # если работ нет - ставим 1
        # TODO: возможно, следует занулить
        # v_score[f_noworks & f_mod] = 0.
        # есть работы по своим модальностям
        f_work_frame = ~f_noworks & f_mod
        logger.debug(f'f_work_frame: {f_work_frame.shape}')

        # (!) следующее не нужно, пока мы оперируем только нормативным временем
        # фильтр работ со временем меньше или равном нормативному
        # f_in_norm = v_bot <= f_day_time_norm
        # оценку 1 получают работы по модальности не превышающие нормативное время
        # TODO: наверное, следует применить power_norm_and_more (это для случая изменения рабочего времени алгоритмом)
        # v_score[f_work_frame & f_in_norm] = 1.
        v_score[f_work_frame] = 1.
        # работы по модальности со временем больше нормативного получают убывающую оценку
        # v_score[f_work_frame & ~f_in_norm] = power_norm_and_more(
        #     v_bot, f_day_time_norm, time_pw)[f_work_frame & ~f_in_norm]

        # немного снижаем оценку за работы по дополнительной модальности
        # logger.info(f'v_score before extra_mod_k:\n{v_score}')
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

        # снижаем оценку за работы в несовпадающие и исходным графиком дни
        # logger.info(f'v_score before conflict_with_base_k:\n{v_score}')
        v_score[f_not_working_days_in_base] *= conflict_with_base_k

        # *
        # добавляем оценку выполнения плана
        # оценка линейно растёт при приближении факта к плану
        # и убывает при перевыполнении

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
        plan_fact_scores[in_plan] = linear_zero_and_norm(day_fact, self.v_plan)[in_plan]
        plan_fact_scores[~in_plan] = power_norm_and_more(
            day_fact, self.v_plan, fact_over_plan_pw
        )[~in_plan]
        # logger.info(f'v_score before plan_fact_scores:\n{v_score}')
        # logger.info(f'plan_fact_scores:\n{plan_fact_scores[:, 0, :, :]}')
        v_score *= plan_fact_scores

        # итоговая оценка
        bot_scores = np.sqrt(np.sum(np.power(v_score, 2), axis=(1, 2, 3)))
        # logger.info(f'power:\n{np.power(v_score, 2)}')
        # logger.info(f'sum: {np.sum(np.power(v_score, 2), axis=(1, 2, 3))}')

        # абсолютные отклонения факта от плана
        diff = - day_fact + self.v_plan
        atol = 1e-1
        f_under_plan = diff < -atol
        f_by_plan = np.isclose(diff, 0., atol=atol)

        # for i in range(len(bot_scores)):
        #     logger.info(f'bot[{i}] score: {bot_scores[i]:.8f},'
        #                 f' недовыполнение: {np.sum(diff[i], where=f_under_plan[i]):6.2f},'
        #                 f' перевыполнение: {np.sum(diff[i], where=~f_under_plan[i] & f_by_plan[i]):6.2f},'
        #                 f' по плану: {np.sum(diff[i], where=f_by_plan[i]):6.2f}.')

        # check = self.v_plan
        # print('Проверка плана ...')
        # for b in range(check.shape[0]):
        #     for m in range(check.shape[2]):
        #         for day in range(check.shape[3]):
        #             if check[b, 0, m, day] == 0:
        #                 print(f'bot: {b}, mod: {m}, day: {day}')
        return bot_scores

    def select(self, container: Container) -> Container:

        best_bots, doctors, scores = container.unpack_best(self.n_survived, copy=True)

        # кладём лучших ботов в другой контейнер
        selected = Container()
        selected.keep(best_bots, doctors)
        selected.set_scores(scores)
        del container
        gc.collect()
        # logger.info(f'scores best scores: {scores}')
        return selected

    def mate(self, bots):
        assert self.n_survived % 2 == 0
        np.set_printoptions(formatter=dict(float=lambda x: f'{x:.1f}'))
        n_best_bots = bots.shape[0]
        gen = self.gen

        # определяем, кто с кем будет скрещиватья
        mate_indices = np.arange(n_best_bots)
        gen.shuffle(mate_indices)
        logger.debug(f'mate_indices:\n{mate_indices}')

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
        logger.debug(f'v_mate {v_mate.shape}:\n{v_mate}')

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

            # mate_pair(b1, new_bot0)
            # mate_pair(b0, new_bot1)
            new_bot0, new_bot1 = change_modalities(new_bot0, new_bot1)
            logger.debug(f'new_bot0:\n{new_bot0}')
            logger.debug(f'new_bot1:\n{new_bot1}')
            gc.collect()

            logger.debug(f'b0 / new_bot0 sum: {b0.sum()} / {new_bot0.sum()}')
            logger.debug(f'b1 / new_bot1 sum: {b1.sum()} / {new_bot1.sum()}')

            # print(f'new_bot0: {new_bot0.shape}')
            # print(f'new_bot1: {new_bot1.shape}')
            new_pair = np.vstack([new_bot0[np.newaxis, :, :, :], new_bot1[np.newaxis, :, :, :]])
            if new_bots is None:
                new_bots = new_pair
                logger.debug(f'new_bots: {new_bots.shape}')
            else:
                new_bots = np.vstack([new_bots, new_pair])

            logger.debug(f'mate_indices:\n{mate_indices[i:i + 2]}')
            # print(f'new_pair[0]:\n{new_pair[0][0, :10]}')
            # print(f'new_pair[1]:\n{new_pair[1][0, :10]}')

        # print(f'v_mod:\n{self.v_mod[0, :3]}')
        logger.debug(f'new_bots: {new_bots.shape}')
        return new_bots

    def mutate(self, bots):

        mutate_k = 0.3  # доля мутирующих генов

        def choice(arr1d):
            """Случайный выбор модальности из доступных"""
            return self.gen.choice(np.where(arr1d > 0)[0])

        n_mod = self.v_mod.shape[2]
        v_mod = self.v_mod.repeat(bots.shape[3], axis=3)
        v_mod = v_mod.repeat(bots.shape[0], axis=0)
        # print(f'source bots {bots.shape}:\n{bots}')
        # print(f'v_mod: {v_mod.shape}')

        src_mod_indices = bots.argmax(axis=2)
        # print(f'src_mod_indices {src_mod_indices.shape}:\n{src_mod_indices}')
        src_mod_indices = src_mod_indices.reshape((-1,))

        rnd_mod_indices = np.apply_along_axis(choice, 2, v_mod)
        # print(f'rnd_mod_indices {rnd_mod_indices.shape}:\n{rnd_mod_indices}')
        rnd_mod_indices = rnd_mod_indices.reshape((-1,))
        # print(f'rnd_mod_indices {rnd_mod_indices.shape}:\n{rnd_mod_indices}')

        bots = bots.transpose((0, 1, 3, 2))
        bots_keep_shape = bots.shape
        bots = bots.reshape((-1, n_mod))
        long_index = np.arange(bots.shape[0])

        v_mutate = self.gen.random(size=len(long_index)) < mutate_k

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

    def run(self, save=False):
        logger.info(f'v_mod: {self.v_mod.shape}')
        logger.info(f'v_plan: {self.v_plan.shape}')
        logger.info(f'v_base_avail: {self.v_base_avail.shape}')

        container = self.initialize(n_bots=self.population_size)
        container.print_shapes()

        # вычисляем оценку ботов
        bot_scores = self.evaluate(container)
        # logger.info(f'bot_scores:\n{bot_scores}')
        container.set_scores(bot_scores)

        # производим отбор
        container = self.select(container)
        gc.collect()

        for generation in range(self.n_generations - 1):
            # print(f'Поколение #{generation}')
            # получаем ботов из контейнера - там остались лучшие
            bots, doctors, scores = container.unpack()
            logger.info(f'Поколение #{generation + 1}/{self.n_generations}, best scores: {scores}')
            # скрещиваем ботов
            with timer('mate'):
                new_bots = self.mate(bots)
            # new_bots = bots
            # производим мутацию
            with timer('mutate'):
                new_bots = self.mutate(new_bots)

            # добавляем случайных ботов
            with timer('initialize'):
                next_container = self.initialize(
                    n_bots=self.population_size - self.n_survived
                )
            # график их работы врачей не меняются
            next_container.append(new_bots, doctors.copy())

            # рассчитываем оценку новых ботов
            # with timer('evaluate') as t:
            next_scores = self.evaluate(next_container)

            # формируем общий контейнер всех ботов: рассчитанных в данном цикле и стека лучших
            next_container.append(bots, doctors)
            next_container.set_scores(np.hstack([next_scores, scores]))

            # производим отбор из всех ботов:
            # with timer('select') as t:
            container = self.select(next_container)
            gc.collect()

        # пишем лучшего бота в базу
        logger.info(f'best scores: {container.get_scores()}')
        best_bots, _, _ = container.unpack_best(1)
        best_bot = best_bots[0]
        schedule = best_bot.sum(axis=1)
        v_plan = self.v_plan[0, 0, :, :]
        diff = best_bot.sum(axis=0) - v_plan
        fmt_4_1 = dict(float=lambda x: f'{x:4.1f}')
        fmt_6_0 = dict(float=lambda x: f'{x:6.0f}')
        fmt_6_1 = dict(float=lambda x: f'{x:6.1f}')
        m = 3600
        if self.mode == 'test':
            m = 1
            logger.info(f'v_base_avail:\n{self.v_base_avail}')
            # logger.info(f'v_plan:\n{v_plan}')
            logger.info(f'best_bot:\n{best_bot}')
        np.set_printoptions(formatter=fmt_4_1)
        logger.info(f'schedule:\n{schedule / m}')
        np.set_printoptions(formatter=fmt_6_1)
        logger.info(f'difference (schedule - plan):\n{diff / m}')
        logger.info(f'v_plan:\n{v_plan / m}')
        np.set_printoptions(formatter=fmt_6_1)
        logger.info(f'schedule by modalities:\n{best_bot.sum(axis=(0, 2)) / m}')
        logger.info(f'plan by modalities:\n{v_plan.sum(axis=-1) / m}')
        logger.info(f'difference by modalities (schedule - plan):\n{diff.sum(axis=-1) / m}')
        logger.info(f'difference total (schedule - plan): {diff.sum() / m:.1f}')
        if save:
            self.save_bot(best_bots[0])

    def save_bot(self, bot):

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
            row['availability'] = self.v_base_avail[doctor_index][row['day_index']]

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
        logger.info('Данные записаны')


class Test:

    def __init__(self, month_start):
        self.n_generations = 10
        self.n_survived = 30

        self.population_size = 50
        self.n_doctors = 3
        self.n_mods = 2
        self.n_days = 7

        self.bots_shape = (self.population_size, self.n_doctors, self.n_mods, self.n_days)
        self.total_size = 1
        for i in self.bots_shape:
            self.total_size *= i
        # self.month_layout = get_month_layout(month_start)
        self.doctors = pd.DataFrame([
            # {'id': 1, 'main_modality': 'flg', 'modalities': ['kt', 'rg', 'mmg', 'dens'],
            {'id': 1, 'main_modality': 'kt', 'modalities': ['kt', 'mrt'],
             'schedule_type': '5/2', 'time_rate': 1.},
            {'id': 3, 'main_modality': 'kt', 'modalities': [],
             'schedule_type': '5/2', 'time_rate': 1.},
            # {'id': 4, 'main_modality': 'kt', 'modalities': ['mrt', 'rg', 'flg', 'dens'],
            #  'schedule_type': '5/2', 'time_rate': 0.8},
            {'id': 4, 'main_modality': 'mrt', 'modalities': [],
             'schedule_type': '2/2', 'time_rate': 0.8},
        ])

        self.month_start = month_start
        self.v_mod = np.array([  # ['kt', 'mrt', 'rg', 'flg', 'mmg', 'dens']
            # [1, 0, 1, 2, 1, 1],
            # [0, 2, 0, 1, 1, 0],
            # [2, 1, 1, 0, 0, 1],
            [2, 1],
            [2, 0],
            [0, 2],
        ])
        self.v_base_avail = np.array([
            [1, 1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 0, 0, 1]
        ])
        self.v_plan = np.array([
            [8, 8, 8, 8, 8, 8, 8],
            [9.6, 9.6, 8, 8, 9.6, 9.6, 8],
            # [0.1, 4, 0.1, 0.1],
            # [8, 0.1, 0.1, 0.1],
            # [0.1, 8, 12, 0.1],
            # [0.1, 4, 1, 7],
        ])[np.newaxis, np.newaxis, :, :]

        mask = np.ones(self.bots_shape, dtype=np.bool_)
        self.bots_mask = mask & self.v_mod[np.newaxis, :, :, np.newaxis] == 0

        self.gen = np.random.default_rng()
        # self.v_doctor = [self.gen.choice([8, 12]) for _ in range(self.n_doctors)]

    def populate(self):
        gen = self.gen
        bots = np.arange(self.total_size)
        zero_indices = gen.integers(0, len(bots), size=len(bots) // 2)
        bots[zero_indices] = 0
        gen.shuffle(bots)
        bots.shape = self.bots_shape

        return bots

    def run(self):
        scheduler = Scheduler(
            self.month_start,
            plan_version='validation',
            n_generations=self.n_generations,
            population_size=self.population_size,
            n_survived=self.n_survived,
            mode='test',
        )
        # print('month_layout:', scheduler.month_layout)
        month_layout = scheduler.month_layout
        month_layout['month_end'] = month_layout['month_start'] + timedelta(days=self.n_days)
        month_layout['month_end_weekday'] = get_weekday(month_layout['month_end'])
        month_layout['num_days'] = self.n_days

        scheduler.doctors = self.doctors
        scheduler.v_mod = self.v_mod[np.newaxis, :, :, np.newaxis]
        scheduler.v_base_avail = self.v_base_avail
        scheduler.n_mods = self.n_mods

        scheduler.v_plan = self.v_plan

        # scheduler.mate(bots)
        scheduler.run(save=False)
        # print(f'bot0:\n{bots[0]}')
        # print(f'bot1:\n{bots[1]}')


if __name__ == '__main__':
    os.chdir('..')
    logger.setup(level=logger.INFO, layout='debug')
    np.set_printoptions(edgeitems=30, linewidth=100000,
                        formatter=dict(float=lambda x: f'{x:.5f}'))

    main_month_start = datetime(2024, 1, 1)

    DB_SCHEMA = 'test'
    test = Test(main_month_start)
    test.run()

    # main_scheduler = Scheduler(
    #     main_month_start,
    #     plan_version='validation',
    #     n_generations=50,
    #     population_size=100,
    #     n_survived=60,
    #     mode='main',
    # )
    # main_scheduler.run(save=False)

    # schedule = get_schedule('base', month_start, data_layer='day')
    # schedule = get_schedule('base', month_start, data_layer='day_modality')
    # schedule = get_schedule('base', month_start, data_layer='day_modality_ce')
    # print(schedule.head(20))
