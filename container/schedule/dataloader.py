import os
import calendar
from datetime import datetime
import pandas as pd
import numpy as np

from settings import DB_VERSION
from common.timeutils import get_weekday
if DB_VERSION == 'PG':
    from common.db import DB, get_all
else:
    from common.dblite import DB, get_all

XLS_FILEPATH = '/Users/ivan/Documents/CIFROPRO/Проекты/Нейронки/Расписание рентген-центра/dataset/'
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
SCHEDULE_TYPES = ['5/2', '2/2']
HOURS_PER_NORM = 8  # количество часов, на которое определён норматив работ
DOCTOR_COLUMNS = ['uid', 'name', 'main_modality', 'modalities', 'schedule_type', 'time_rate',
                  'is_active', 'day_start_time', 'day_end_time', 'rest_time']
DOCTOR_DAY_PLAN_COLUMNS = ['version', 'doctor_availability', 'modality', 'contrast_enhancement', 'time_volume']


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


class DataLoader:

    def __init__(self, db_schema=None):

        self.schema_placeholder = ''
        if db_schema:
            self.schema_placeholder = db_schema + '.'

    def get_doctors(self) -> pd.DataFrame:
        """Возращает датафрейм врачей"""
        db = DB()
        if DB_VERSION == 'PG':
            q = (f"SELECT id, {', '.join(DOCTOR_COLUMNS)}"
                 f" FROM {self.schema_placeholder}doctor"
                 f" WHERE is_active"
                 f" ORDER BY id"
                 )
        else:
            q = (f"SELECT id, {', '.join(DOCTOR_COLUMNS)}"
                 f" FROM {self.schema_placeholder}doctor"
                 f" WHERE is_active = 't'"
                 f" ORDER BY id"
                 )
        with db.get_cursor() as cursor:
            cursor.execute(q)
            doctor_df = get_all(cursor)
        db.close()
        return doctor_df

    def get_doctors_for_schedule_save(self) -> pd.DataFrame:
        q = f"""
            select id, uid, day_start_time, schedule_type, time_rate,
            row_number() over (order by id) - 1 as row_index 
            from {self.schema_placeholder}doctor order by id;
        """
        db = DB()
        with db.get_cursor() as cursor:
            cursor.execute(q)
            doctors = get_all(cursor)
        db.close()
        return doctors

    def get_day_plan(self, version, month_start: datetime, with_ce=False):

        month_start_data = month_start.isocalendar()  # datetime.IsoCalendarDate(year=2024, week=1, weekday=1)
        month_start_week = month_start_data[1]
        start_weekday = month_start_data[2]
        year = month_start_data[0]
        num_days = calendar.monthrange(year, month_start.month)[1]

        month_end = datetime(year, month_start.month, num_days)
        month_end_data = month_end.isocalendar()  # datetime.IsoCalendarDate(year=2024, week=1, weekday=1)
        month_end_week = month_end_data[1]

        contrast_enhancement = 'p.contrast_enhancement,' if with_ce else ''

        if DB_VERSION == 'PG':
            query = f"""
                WITH raw_plan as (
                    SELECT
                        -- вычисляем дату от начала недели
                        '{month_start.strftime('%Y-%m-%d')}'::date - {start_weekday}
                        + row_number() over (
                            partition by {contrast_enhancement} p.modality
                            order by p.week, d.weekday
                        )::int as plan_date,
                        p.week,
                        d.weekday,
                        p.modality,
                        {contrast_enhancement}
                        sum(p.amount / 7.) as work_amount,
                        sum(p.amount / n.max_value * 8 / 7) as time_volume
                    FROM {self.schema_placeholder}work_plan_summary as p
                    LEFT JOIN {self.schema_placeholder}time_norm as n
                        ON n.modality = p.modality AND n.contrast_enhancement = p.contrast_enhancement
                    FULL JOIN (SELECT column1 as weekday FROM (
                            VALUES (1), (2), (3), (4), (5), (6), (7)
                            ) as val) as d
                        ON true
                    WHERE
                        p.version = '{version}' and p.year = {year} 
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
        else:
            query = f"""
                WITH plan_prepare as (
                    SELECT
                        -- вычисляем дату от начала недели
                        row_number() over (
                            partition by {contrast_enhancement} p.modality
                            order by p.week, d.weekday
                            ) as row_number,
                        p.week,
                        d.weekday,
                        p.modality,
                        {contrast_enhancement}
                        sum(p.amount / 7.) as work_amount,
                        sum(p.amount / n.max_value * 8 / 7) as time_volume
                    FROM {self.schema_placeholder}work_plan_summary as p,
                        (SELECT column1 as weekday FROM (
                            VALUES (1), (2), (3), (4), (5), (6), (7)
                            ) as val) as d
                    LEFT JOIN {self.schema_placeholder}time_norm as n
                        ON n.modality = p.modality AND n.contrast_enhancement = p.contrast_enhancement
                    WHERE
                        p.version = '{version}' and p.year = {year} 
                        and p.week >= {month_start_week} and p.week <= {month_end_week}
                    GROUP BY
                        p.modality,
                        {contrast_enhancement}        
                        d.weekday,
                        p.week
                    ORDER BY
                        p.week,
                        d.weekday
                ),
                plan_date_calc as (
                    SELECT *,
                        -- вычисляем дату от начала недели
                        date('{month_start.strftime('%Y-%m-%d')}', '-{start_weekday} days',
                        '+' || row_number || ' days') as plan_date
                    FROM plan_prepare
                )
                SELECT 
                    week, weekday, plan_date, row_number, modality, {contrast_enhancement} time_volume, 
                    cast(substr(plan_date, 6, 2) as int) as month,
                    --cast(substr(plan_date, 9, 2) as int) as day_number,
                    cast(substr(plan_date, 9, 2) as int) - 1 as day_index
                FROM plan_date_calc;
            """

        # print(f'query:\n{query}')
        db = DB()
        with db.get_cursor() as cursor:
            cursor.execute(query)
            day_plan = get_all(cursor)
        db.close()

        if DB_VERSION == 'SQLite':
            day_plan['plan_date'] = pd.to_datetime(day_plan['plan_date'])

        return day_plan

    def get_schedule(self, version, month_start, data_layer):

        if data_layer == 'day':
            if DB_VERSION == 'PG':
                query = f"""
                    -- расписание по врачам на месяц по дням
                    SELECT
                        d.id as doctor_id,
                        d.name,
                        date_trunc('day', da.day_start) as day,
                        extract(day from da.day_start) - 1 as day_index,
                        da.availability,
                        da.time_volume
                    FROM {self.schema_placeholder}doctor_availability as da
                    LEFT JOIN {self.schema_placeholder}doctor as d
                        ON d.uid = da.doctor
                    WHERE
                        da.version = '{version}' 
                        and date_trunc('month', da.day_start) = '{month_start.strftime('%Y-%m-%d')}'::date
                    ORDER BY
                        d.id,
                        extract(day from da.day_start)
                    ;"""
            else:
                query = f"""
                -- расписание по врачам на месяц по дням
                SELECT
                    d.id as doctor_id,
                    d.name,
                    date(da.day_start, 'start of day') as day,
                    cast(substr(da.day_start, 9, 2) as int) - 1 as day_index,
                    da.availability,
                    da.time_volume
                FROM {self.schema_placeholder}doctor_availability as da
                LEFT JOIN {self.schema_placeholder}doctor as d
                    ON d.uid = da.doctor
                WHERE
                    da.version = '{version}' 
                    and date(da.day_start, 'start of month') = '{month_start.strftime('%Y-%m-%d')}'
                ORDER BY
                    d.id,
                    cast(substr(da.day_start, 9, 2) as int);
            """
        elif data_layer == 'day_modality':
            if DB_VERSION == 'PG':
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
                    FROM {self.schema_placeholder}doctor_availability as da
                    LEFT JOIN {self.schema_placeholder}doctor as d
                        ON d.uid = da.doctor
                    LEFT JOIN {self.schema_placeholder}doctor_day_plan as ddp
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
                    ;"""
            else:
                query = f"""
                    -- расписание по врачам на месяц по дням в разрезе модальностей
                    SELECT
                        d.id as doctor_id,
                        d.name,
                        date(da.day_start, 'start of day') as day,
                        da.availability,
                        --da.time_volume
                        ddp.modality,
                        ddp.time_volume as time_volume
                    FROM {self.schema_placeholder}doctor_availability as da
                    LEFT JOIN {self.schema_placeholder}doctor as d
                        ON d.uid = da.doctor
                    LEFT JOIN {self.schema_placeholder}doctor_day_plan as ddp
                        ON ddp.doctor_availability = da.uid
                    WHERE
                        da.version = '{version}' 
                        and date(da.day_start, 'start of month') = '{month_start.strftime('%Y-%m-%d')}'
                    --GROUP BY
                    --    d.id,
                    --    d.name,
                    --    date(da.day_start, 'start of day'),
                    --    da.availability,
                    --    ddp.modality
                    ;"""
        elif data_layer == 'day_modality_ce':
            if DB_VERSION == 'PG':
                query = f"""
                    -- расписание по врачам на месяц по дням в разрезе модальностей и КУ
                    SELECT
                        d.name,
                        date_trunc('day', da.day_start) as day,
                        da.availability,
                        ddp.modality,
                        ddp.contrast_enhancement,
                        sum(ddp.time_volume)::time as time_volume
                    FROM {self.schema_placeholder}doctor_availability as da
                    LEFT JOIN {self.schema_placeholder}doctor as d
                        ON d.uid = da.doctor
                    LEFT JOIN {self.schema_placeholder}doctor_day_plan as ddp
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
                    ;"""
            else:
                raise NotImplementedError('Запрос для SQLite с data_layer == day_modality_ce не реализован.')
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

    # получение сводного плана
    # month_start = datetime(2024, 2, 1)
    # day_plan = get_day_plan('validation', month_start, with_ce=False)
    # print(day_plan[:20])
    # day_plan.to_excel(XLS_FILEPATH + 'day_plan_df.xlsx')
