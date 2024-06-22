import os
import uuid
import random
import hashlib
import calendar
from datetime import time, datetime, timedelta
import pandas as pd
import numpy as np

from lib.db import DB, get_all
from lib.timeutils import get_weekday

XLS_FILEPATH = '/Users/ivan/Documents/CIFROPRO/Проекты/Нейронки/Расписание рентген-центра/dataset/'
DB_SCHEMA = 'roentgen'
MODALITIES: list[str] = ['kt', 'mrt', 'rg', 'flg', 'mmg', 'dens']
MODALITIES_MAP = {
    'кт': 'kt',
    'мрт': 'mrt',
    'рг': 'rg',
    'флг': 'flg',
    'ммг': 'mmg',
    'денс': 'dens',
    'денситометрия': 'dens'
}
SCHEDULE_TYPES = np.array(['5/2', '2/2'])
HOURS_PER_NORM = 8  # количество часов, на которое определён норматив работ
DOCTOR_COLUMNS = ['uid', 'name', 'main_modality', 'modalities', 'schedule_type', 'time_rate',
                  'is_active', 'day_start_time', 'day_end_time', 'rest_time']
DOCTOR_DAY_PLAN_COLUMNS = ['version', 'doctor_availability', 'modality', 'contrast_enhancement', 'time_volume']


def get_uuid_from_columns(df_row, *args):
    """Генерирует UUID из значений заданных колонок строки датафрейма"""
    source = "/".join([str(df_row[column_name]) for column_name in args])
    row_hash = hashlib.md5(source.encode('utf-8'))
    return uuid.UUID(row_hash.hexdigest())


def get_working_days_matrices():
    """
    Возвращает маски рабочих дней для двух видов графиков: 5/2, 2/2, – упорядоченные
    по началу первого выходного дня в неделе.
    """
    return {
        '5/2': np.array([
            [0, 0, 1, 1, 1, 1, 1],
            [1, 0, 0, 1, 1, 1, 1],
            [1, 1, 0, 0, 1, 1, 1],
            [1, 1, 1, 0, 0, 1, 1],
            [1, 1, 1, 1, 0, 0, 1],
            [1, 1, 1, 1, 1, 0, 0],
            [0, 1, 1, 1, 1, 1, 0]
        ]),
        '2/2': np.array([
            [0, 0, 1, 1, 0, 0, 1],
            [1, 0, 0, 1, 1, 0, 0],
            [1, 1, 0, 0, 1, 1, 0],
            [0, 1, 1, 0, 0, 1, 1]
        ])
    }


def get_month_layout(month_start) -> dict:
    """
    Формирует месячный шаблон для графика с полным количеством недель,
    где дни соответствующего месяца заполнены единицами, а дни соседних месяцев нулями.
    """
    month_start_weekday = get_weekday(month_start)
    num_days = calendar.monthrange(month_start.year, month_start.month)[1]
    if month_start_weekday > 1:
        schedule_template = np.hstack([
            np.repeat(0., month_start_weekday - 1),
            np.repeat(1., num_days)
        ])
    else:
        schedule_template = np.repeat(1., num_days)

    month_end = datetime(month_start.year, month_start.month, num_days)
    month_end_weekday = get_weekday(month_end)
    if month_end_weekday < 7:
        schedule_template = np.hstack([schedule_template, np.repeat(0., 7 - month_end_weekday)])
    schedule_template.shape = (-1, 7)

    return {
        'month_start': month_start,
        'month_end': month_end,
        'month_start_weekday': month_start_weekday,
        'month_end_weekday': month_end_weekday,
        'num_days': num_days,
        'schedule_template': schedule_template.reshape((-1, 7)),
        'working_days_matrices': get_working_days_matrices()
    }


def generate_doctor_month_schedule(schedule_template, schedule_type, time_rate, working_days_matrix, weekend_start):
    """
    Генерирует месячный график работы.

    :param schedule_template: шаблон месяца с полными неделями, где 1 - текущий месяц, 0 - соседний
    :param schedule_type: вид графика 5/2, 2/2
    :param time_rate: ставка <= 1.0
    :param working_days_matrix: месячная маска
    :param weekend_start:
    :return:
    """
    month_schedule = schedule_template.copy()
    if schedule_type == '5/2':
        return month_schedule * working_days_matrix[weekend_start] * 8. * time_rate
    elif schedule_type == '2/2':
        n_masks = len(working_days_matrix)
        week_shift = weekend_start % 7
        for week in range(len(schedule_template)):
            index = (week + week_shift) % n_masks
            month_schedule[week] = month_schedule[week] * working_days_matrix[index] * 12. * time_rate
        return month_schedule
    else:
        raise RuntimeError('Неизвестное обозначение вида графика:', schedule_type)


def generate_schedule(doctors, month_layout):

    columns = ['uid', 'doctor', 'day_start', 'availability', 'time_volume']

    month_start = month_layout['month_start']
    month_start_weekday= month_layout['month_start_weekday']
    month_end_weekday= month_layout['month_end_weekday']
    num_days = month_layout['num_days']
    schedule_template = month_layout['schedule_template']
    working_days_matrices = month_layout['working_days_matrices']

    gen = np.random.default_rng()
    doctor_data = []

    time_table = []
    for i, doctor in doctors.iterrows():
        doctor_id = doctor['id']
        doctor_uid = doctor['uid']
        time_rate = doctor['time_rate']
        # schedule_type = gen.choice(SCHEDULE_TYPES) if random_schedule_type else doctor['schedule_type']
        schedule_type = doctor['schedule_type']

        # случайный день начала выходных
        weekend_start = random.randint(0, 6)
        working_days_matrix = working_days_matrices[schedule_type]

        month_schedule = generate_doctor_month_schedule(
            schedule_template, schedule_type, time_rate, working_days_matrix, weekend_start
        )
        # преобразуем в одномерный вектор
        month_schedule.shape = (-1,)
        # оставляем только дни текущего месяца
        if month_end_weekday < 7:
            month_schedule = month_schedule[month_start_weekday - 1: -7 + month_end_weekday]
        else:
            month_schedule = month_schedule[month_start_weekday - 1:]

        for day_number in range(num_days):
            # TODO: сгенерировать отпуска
            day_start = datetime.combine(month_start + timedelta(days=day_number), doctor['day_start_time'])
            # флаг активности: -1 - недоступен для распределения, 0 - выходной день, 1 - рабочий день
            if month_schedule[day_number] > 0:
                availability = -1 if gen.random() < 0.2 else 1
            else:
                availability = 0
            time_volume = timedelta(hours=month_schedule[day_number]) if availability > 0 else time(0)
            uid = uuid.uuid4()
            time_table.append([uid, doctor_uid, day_start, availability, time_volume])

    return pd.DataFrame(time_table, columns=columns), doctor_data


def get_day_plan(version, month_start: datetime, with_ce=False):

    month_start_data = month_start.isocalendar()  # datetime.IsoCalendarDate(year=2024, week=1, weekday=1)
    month_start_week = month_start_data[1]
    start_weekday = month_start_data[2]
    year = month_start_data[0]
    num_days = calendar.monthrange(year, month_start.month)[1]

    month_end = datetime(year, month_start.month, num_days)
    month_end_data = month_end.isocalendar()  # datetime.IsoCalendarDate(year=2024, week=1, weekday=1)
    month_end_week = month_end_data[1]

    contrast_enhancement = 'p.contrast_enhancement,' if with_ce else ''

    query = f"""
        WITH raw_plan as (
            SELECT
                -- вычисляем дату от начала недели
                '{month_start.strftime('%Y-%m-%d')}'::date - {start_weekday}
                + row_number() over (
                    partition by {contrast_enhancement} p.modality 
                    order by p.week, d.weekday
                    )::integer as plan_date,
                p.week,
                d.weekday,
                p.modality,
                {contrast_enhancement}
                sum(p.amount / 7.) as work_amount,
                sum(p.amount / n.max_value * 8 / 7) as time_volume
            FROM "roentgen".work_plan_summary as p
            LEFT JOIN "roentgen".time_norm as n
                ON n.modality = p.modality AND n.contrast_enhancement = p.contrast_enhancement
            FULL JOIN (VALUES (1), (2), (3), (4), (5), (6), (7)) as d(weekday)
                ON true
            WHERE
                p.version = '{version}' and year = {year} 
                and p.week >= {month_start_week} and p.week <= {month_end_week}
            GROUP BY
                p.modality,
                {contrast_enhancement}        
                d.weekday,
                p.week
            ORDER BY
                p.week,
                d.weekday
        )
        SELECT *,
            extract(month from plan_date) as month,
            extract(day from plan_date) as day_number,
            extract(day from plan_date) - 1 as day_index
        FROM raw_plan;
    """
    db = DB()
    with db.get_cursor() as cursor:
        cursor.execute(query)
        day_plan = get_all(cursor)
    db.close()
    return day_plan


def get_schedule(version, month_start, data_layer):

    if data_layer == 'day':
        query = f"""
            -- расписание по врачам на месяц по дням
            SELECT
                d.id as doctor_id,
                d.name,
                date_trunc('day', da.day_start) as day,
                extract(day from da.day_start) - 1 as day_index,
                da.availability,
                da.time_volume
            FROM "roentgen".doctor_availability as da
            LEFT JOIN "roentgen".doctor as d
                ON d.uid = da.doctor
            WHERE
                da.version = '{version}' 
                and date_trunc('month', da.day_start) = '{month_start.strftime('%Y-%m-%d')}'::date
            ORDER BY
                d.id,
                extract(day from da.day_start)
            ;
        """
    elif data_layer == 'day_modality':
        query = f"""
            -- расписание по врачам на месяц по дням в разрезе модальностей
            SELECT
                d.id as doctor_id,
                d.name,
                date_trunc('day', da.day_start) as day,
                da.availability,
                --da.time_volume
                ddp.modality,
                sum(ddp.time_volume)::time as time_volume
            FROM "roentgen".doctor_availability as da
            LEFT JOIN "roentgen".doctor as d
                ON d.uid = da.doctor
            LEFT JOIN "roentgen".doctor_day_plan as ddp
                ON ddp.doctor_availability = da.uid
            WHERE
                da.version = '{version}' 
                and date_trunc('month', da.day_start) = '{month_start.strftime('%Y-%m-%d')}'::date
            GROUP BY
                d.id,
                d.name,
                date_trunc('day', da.day_start),
                da.availability,
                ddp.modality
            ;
        """
    elif data_layer == 'day_modality_ce':
        query = f"""
            -- расписание по врачам на месяц по дням в разрезе модальностей и КУ
            SELECT
                d.name,
                date_trunc('day', da.day_start) as day,
                da.availability,
                ddp.modality,
                ddp.contrast_enhancement,
                sum(ddp.time_volume)::time as time_volume
            FROM "roentgen".doctor_availability as da
            LEFT JOIN "roentgen".doctor as d
                ON d.uid = da.doctor
            LEFT JOIN "roentgen".doctor_day_plan as ddp
                ON ddp.doctor_availability = da.uid
            WHERE
                da.version = '{version}' 
                and date_trunc('month', da.day_start) = '{month_start.strftime('%Y-%m-%d')}'::date
            GROUP BY
                d.name,
                date_trunc('day', da.day_start),
                da.availability,
                ddp.modality,
                ddp.contrast_enhancement
            ;
        """
    else:
        raise RuntimeError('Неизвестный вид запроса данных:', data_layer)

    db = DB()
    with db.get_cursor() as cursor:
        cursor.execute(query)
        schedule = get_all(cursor)
    db.close()
    return schedule


if __name__ == '__main__':
    os.chdir('..')

    # загрузка факта работ из Excel
    # load_summary('work_summary', 'work_summary.xlsx', 'Chart data')
    # загрузка плана работ из Excel
    # load_summary('work_plan_summary', 'work_plan.xlsx', 'Chart data', version='validation')

    # загрузка врачей из Excel
    # load_doctor('doctor', 'Пример табеля.xlsx', 'for_load')
    # генерация таблицы доступности врачей
    start_of_month = datetime(2024, 1, 1)
    # load_doctor_availability('doctor_availability', start_of_month, version='base')
    # генерация записей по работе врача в течение дня
    # load_doctor_day_plan('doctor_day_plan', start_of_month, version='base')
    # генерация таблицы плана по дням
    # create_day_plan('day_plan', start_of_month, 'validation')

    # получение сводного плана
    # month_start = datetime(2024, 2, 1)
    # day_plan = get_day_plan('validation', month_start, with_ce=False)
    # print(day_plan[:20])
    # day_plan.to_excel(XLS_FILEPATH + 'day_plan_df.xlsx')



