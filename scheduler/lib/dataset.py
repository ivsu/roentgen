import os
import pandas as pd
import numpy as np
import uuid
import random
from datetime import time, datetime, timedelta

from lib.db import DB, get_all
from lib.timeutils import time_to_interval, get_time_chunck
from lib.dataloader import get_day_plan, get_month_layout, get_uuid_from_columns, \
    generate_schedule, \
    DOCTOR_COLUMNS, DOCTOR_DAY_PLAN_COLUMNS, DB_SCHEMA, MODALITIES, MODALITIES_MAP, SCHEDULE_TYPES


XLS_FILEPATH = '/Users/ivan/Documents/CIFROPRO/Проекты/Нейронки/Расписание рентген-центра/dataset/'


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

    db = DB()
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
    db = DB()
    db.upsert(df, tablename, unique)
    db.close()
    print('Сохранено записей:', len(df))


def load_doctor_availability(tablename, month_start, version):
    db = DB()

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
    db = DB()
    try:
        db.upsert(df, tablename, unique)
        print('Сохранено записей:', len(df))
    except Exception as e:
        print(repr(e))
        df.to_excel(XLS_FILEPATH + 'doctor_availability_df.xlsx')
    db.close()


def load_doctor_day_plan(tablename, month_start, version):
    db = DB()

    q = (f"SELECT da.uid, da.time_volume, d.main_modality, d.modalities"
         f" FROM {DB_SCHEMA}.doctor_availability as da"
         f" LEFT JOIN {DB_SCHEMA}.doctor as d"
         f"   ON d.uid = da.doctor"
         f" WHERE version = '{version}' and availability = 1"
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
    db = DB()
    try:
        db.upsert(df, tablename, unique)
        print('Сохранено записей:', len(df))
    except Exception as e:
        print(repr(e))
        df.to_excel(XLS_FILEPATH + 'doctor_day_plan_df.xlsx')
    db.close()


def create_day_plan(tablename, month_start, version):
    day_plan_df = get_day_plan(version, month_start, with_ce=True)
    day_plan_df['uid'] = None
    day_plan_df['uid'] = day_plan_df['uid'].apply(lambda uid: uuid.uuid4())
    day_plan_df['version'] = version
    day_plan_df['time_volume'] = day_plan_df['time_volume'].apply(lambda t: timedelta(hours=t))
    day_plan_df = day_plan_df.rename(columns={'plan_date': 'day_start'}) \
        .drop(columns=['week', 'weekday', 'month', 'day_number', 'day_index'])
    print(day_plan_df.head())

    db = DB()
    unique = ['day_start', 'modality', 'contrast_enhancement']
    db.upsert(day_plan_df, tablename, unique)
    db.close()


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
