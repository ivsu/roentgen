import os
import pandas as pd
import numpy as np
import uuid
import hashlib
import random
from datetime import time, datetime, timedelta

from common.db import DB, get_all
from common.timeutils import time_to_interval, get_time_chunck
from schedule.dataloader import DataLoader, get_month_layout, \
    DOCTOR_COLUMNS, DOCTOR_DAY_PLAN_COLUMNS, MODALITIES, MODALITIES_MAP, SCHEDULE_TYPES

DATALOADER = None
DB_SCHEMA = None
XLS_FILEPATH = '/Users/ivan/Documents/CIFROPRO/Проекты/Нейронки/Расписание рентген-центра/dataset/'


def get_uuid_from_columns(df_row, *args):
    """Генерирует UUID из значений заданных колонок строки датафрейма"""
    source = "/".join([str(df_row[column_name]) for column_name in args])
    row_hash = hashlib.md5(source.encode('utf-8'))
    return uuid.UUID(row_hash.hexdigest())


def load_summary(tablename, filename, sheetname, version=None):
    """"Загружает понедельный факт или план в разрезе модальностей из Excel"""

    df_src = pd.read_excel(XLS_FILEPATH + filename, sheetname)
    df = df_src.melt(
        id_vars=['year', 'week'], value_vars=MODALITIES,
        var_name='modality', value_name='amount'
    )
    df['contrast_enhancement'] = 'none'

    def enhance(value_var):
        extra_df = df_src.melt(
            id_vars=['year', 'week'], value_vars=[value_var],
            var_name='modality', value_name='amount'
        )
        extra_df[['modality', 'contrast_enhancement']] = extra_df['modality'].str.split('_', expand=True)
        return extra_df

    for ce in ['mrt_ce1', 'mrt_ce2', 'kt_ce1', 'kt_ce2']:
        df = pd.concat([df, enhance(ce)])

    df['amount'].fillna(0, inplace=True)
    print(df.head())

    if version:
        df['version'] = version

    # формируем уникальный uid из списка уникальности записи
    unique = ['year', 'week', 'modality', 'contrast_enhancement']
    df['uid'] = df.apply(get_uuid_from_columns, args=tuple(unique), axis=1)

    db = DB(DB_SCHEMA)
    db.upsert(df, tablename, unique)
    db.close()


def load_doctor(tablename, filename, sheetname):
    """Заполняет таблицу врачей"""
    df_src = pd.read_excel(XLS_FILEPATH + filename, sheetname)
    print(df_src.head())

    def is_empty(value):
        return pd.isna(value) or value is None or str(value).strip() == ''

    unknown_modalities = []

    def check_modality(mod_ru) -> bool:
        if mod_ru not in MODALITIES_MAP:
            if mod_ru not in unknown_modalities:
                print(f'Модальность "{mod_ru}" не найдена в словаре.')
                unknown_modalities.append(mod_ru)
            return False
        return True

    doctors = []
    for i, row in df_src.iterrows():
        modality_ru: str = row['modality']
        if is_empty(modality_ru):
            print('Не задана модальность в строке:', i)
            continue
        modality_ru = modality_ru.lower().strip()
        if not check_modality(modality_ru):
            continue
        main_modality = MODALITIES_MAP[modality_ru]

        # веса модальностей
        # w_modalities = [0 for _ in range(len(MODALITIES))]
        # modality_indices = {m: index for index, m in enumerate(MODALITIES)}

        # set_modality(modalities, modality_ru, unknown_modalities, True)

        # распаковка дополнительных модальностей
        extra_modalities = row['extra_modalities']
        modalities = []
        if not is_empty(extra_modalities):
            for em_ru in extra_modalities.split(','):
                em_ru = em_ru.lower().strip()
                if not check_modality(em_ru):
                    continue
                modality = MODALITIES_MAP[em_ru]
                # основную модальность в список дополнительных не добавляем
                if modality not in modalities and modality != main_modality:
                    modalities.append(modality)

        uid = uuid.uuid4()
        schedule_type = random.choice(SCHEDULE_TYPES)
        day_start_time = time(8, 0, 0)
        time_rate = row['time_rate']
        if schedule_type == '5/2':
            day_duration_sec = time_rate * 8 * 60 * 60
        elif schedule_type == '2/2':
            day_duration_sec = time_rate * 12 * 60 * 60
        else:
            raise RuntimeError('Неизвестный вид графика:', schedule_type)

        td = timedelta(seconds=day_duration_sec)
        rest_time = time(0, 30) if td <= timedelta(hours=8) else time(1, 0)
        day_end_time = datetime.min + time_to_interval(day_start_time) + td + time_to_interval(rest_time)

        is_active = True

        doctors.append([uid, row['name'], main_modality, modalities, schedule_type,
                        time_rate, is_active, day_start_time, day_end_time, rest_time])

    df = pd.DataFrame(doctors, columns=DOCTOR_COLUMNS)
    print(df.tail())
    print(df.iloc[0])
    unique = ['name']
    db = DB(DB_SCHEMA)
    db.upsert(df, tablename, unique)
    db.close()
    print('Сохранено записей:', len(df))


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
    month_start_weekday = month_layout['month_start_weekday']
    month_end_weekday = month_layout['month_end_weekday']
    num_days = month_layout['num_days']
    schedule_template = month_layout['schedule_template']
    working_days_matrices = month_layout['working_days_matrices']

    gen = np.random.default_rng()
    doctor_data = []

    time_table = []
    for i, doctor in doctors.iterrows():
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
                availability = -1 if gen.random() < 0.1 else 1
            else:
                availability = 0
            time_volume = timedelta(hours=month_schedule[day_number]) if availability > 0 else time(0)
            uid = uuid.uuid4()
            time_table.append([uid, doctor_uid, day_start, availability, time_volume])

    return pd.DataFrame(time_table, columns=columns), doctor_data


def load_doctor_availability(tablename, month_start, version):
    db = DB(DB_SCHEMA)

    q = (f"SELECT id, {', '.join(DOCTOR_COLUMNS)}"
         f" FROM {DB_SCHEMA}.doctor"
         f" WHERE is_active;"
         )
    with db.get_cursor() as cursor:
        cursor.execute(q)
        doctors: pd.DataFrame = get_all(cursor)

    month_layout = get_month_layout(month_start)
    # выход: [uid, doctor_uid, day_start, availability, time_volume]
    df, _ = generate_schedule(doctors, month_layout)
    df['version'] = version

    unique = ['version', 'doctor', 'day_start']
    try:
        db.upsert(df, tablename, unique)
        print('Сохранено записей:', len(df))
    except Exception as e:
        print(repr(e))
        df.to_excel(XLS_FILEPATH + 'doctor_availability_df.xlsx')
    db.close()


def load_doctor_day_plan(tablename, month_start, version):
    db = DB(DB_SCHEMA)

    q = (f"SELECT da.uid, da.time_volume, d.main_modality, d.modalities"
         f" FROM {DB_SCHEMA}.doctor_availability as da"
         f" LEFT JOIN {DB_SCHEMA}.doctor as d"
         f"   ON d.uid = da.doctor"
         f" WHERE da.version = '{version}' and da.availability = 1"
         f"     AND date_trunc('month', da.day_start) = '{month_start.strftime('%Y-%m-%d')}'"
         )
    with db.get_cursor() as cursor:
        cursor.execute(q)
        doctor_availability: pd.DataFrame = get_all(cursor)

    day_plan_version = 'example'

    gen = np.random.default_rng()
    doctor_day_plan = []
    for i, row in doctor_availability.iterrows():
        modalities = row['modalities']
        if len(modalities) == 0:
            doctor_day_plan.append([day_plan_version, row['uid'], row['main_modality'], 'none', row['time_volume']])
            continue

        gen.shuffle(modalities)
        modalities = np.hstack([row['main_modality'], modalities])
        modalities = modalities[0:gen.integers(1, len(modalities))]
        time_intervals = gen.dirichlet(np.ones(len(modalities)), size=1)[0]
        time_left = row['time_volume']
        for mi, m in enumerate(modalities):
            ce = 'none'
            if m in ['kt', 'mrt']:
                ce = gen.choice(['none', 'ce1', 'ce2'])
            if mi == len(modalities) - 1:
                time_value = time_left
            else:
                time_value = get_time_chunck(row['time_volume'], time_intervals[mi])
                time_left = (datetime.min + time_to_interval(time_left) - time_to_interval(time_value)).time()
            doctor_day_plan.append([day_plan_version, row['uid'], m, ce, time_value])

    df = pd.DataFrame(doctor_day_plan, columns=DOCTOR_DAY_PLAN_COLUMNS)
    unique = ['version', 'doctor_availability', 'modality', 'contrast_enhancement']

    try:
        db.upsert(df, tablename, unique)
        print('Сохранено записей:', len(df))
    except Exception as e:
        print(repr(e))
        df.to_excel(XLS_FILEPATH + 'doctor_day_plan_df.xlsx')
    db.close()


def create_day_plan(tablename, month_start, version):
    day_plan_df = DATALOADER.get_day_plan(version, month_start, with_ce=True)
    day_plan_df['uid'] = None
    day_plan_df['uid'] = day_plan_df['uid'].apply(lambda uid: uuid.uuid4())
    day_plan_df['version'] = version
    day_plan_df['time_volume'] = day_plan_df['time_volume'].apply(lambda t: timedelta(hours=t))
    day_plan_df = day_plan_df.rename(columns={'plan_date': 'day_start'}) \
        .drop(columns=['week', 'weekday', 'month', 'day_number', 'day_index'])
    print(day_plan_df.head())

    db = DB(DB_SCHEMA)
    unique = ['day_start', 'modality', 'contrast_enhancement']
    db.upsert(day_plan_df, tablename, unique)
    db.close()


def generate_test_work_plan_summary():
    plan_np = np.array([ # (mods, weeks)
        [56.],
        [75.]
    ])
    mods = np.arange(plan_np.shape[0]).repeat(plan_np.shape[1])
    weeks = np.tile(np.arange(plan_np.shape[1]), plan_np.shape[0])
    df = pd.DataFrame(plan_np.reshape(-1,), columns=['amount'])
    df['version'] = 'validation'
    df['year'] = 2024
    df['week'] = pd.Series(weeks) + 1
    df['modality'] = pd.Series(mods).apply(lambda m: MODALITIES[m])
    df['contrast_enhancement'] = 'none'
    # print(df)
    df['uid'] = None
    df['uid'] = df['uid'].apply(lambda uid: uuid.uuid4())

    db = DB(DB_SCHEMA)
    unique = ['version', 'year', 'week', 'modality', 'contrast_enhancement']
    db.upsert(df, 'work_plan_summary', unique)
    db.close()
    print('Записано строк:', len(df))


def generate_test_time_norm():
    time_norm = [
        ['kt', 'none', 1, 8],
        ['mrt', 'none', 1, 10],
    ]
    df = pd.DataFrame(time_norm, columns=['modality', 'contrast_enhancement', 'min_value', 'max_value'])
    # print(df)
    db = DB(DB_SCHEMA)
    unique = ['modality', 'contrast_enhancement']
    db.upsert(df, 'time_norm', unique)
    db.close()
    print('Записано строк:', len(df))


if __name__ == '__main__':
    os.chdir('..')

    mode = 'test'
    if mode == 'test':
        DB_SCHEMA = 'test'
    else:
        DB_SCHEMA = 'roentgen'

    DATALOADER = DataLoader(DB_SCHEMA)
    start_of_month = datetime(2024, 1, 1)

    if mode == 'test':
        # генерация таблицы доступности врачей
        # load_doctor_availability('doctor_availability', start_of_month, version='base')
        # генерация записей по работе врача в течение дня
        # load_doctor_day_plan('doctor_day_plan', start_of_month, version='base')
        # генерация плана работ
        generate_test_work_plan_summary()
        # генерация норм времени
        # generate_test_time_norm()
    else:
        # загрузка факта работ из Excel
        # load_summary('work_summary', 'work_summary.xlsx', 'Chart data')
        # загрузка плана работ из Excel
        # load_summary('work_plan_summary', 'work_plan.xlsx', 'Chart data', version='validation')

        # загрузка врачей из Excel
        # load_doctor('doctor', 'Пример табеля.xlsx', 'for_load')
        # генерация таблицы доступности врачей
        # load_doctor_availability('doctor_availability', start_of_month, version='base')
        # генерация записей по работе врача в течение дня
        # load_doctor_day_plan('doctor_day_plan', start_of_month, version='base')
        # генерация таблицы плана по дням
        # create_day_plan('day_plan', start_of_month, 'validation')
        pass
