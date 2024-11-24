import os
import pandas as pd
import numpy as np
import uuid
import hashlib
import random
from datetime import time, datetime, timedelta

from common.timeutils import time_to_interval, get_time_chunck
from common.logger import Logger
from common.showdata import expand_pandas_output
from schedule.dataloader import DataLoader, get_month_layout, \
    DOCTOR_COLUMNS, DOCTOR_DAY_PLAN_COLUMNS, MODALITIES, MODALITIES_MAP, SCHEDULE_TYPES
from settings import LOCAL_FOLDER, DB_VERSION, XLS_FILEPATH
if DB_VERSION == 'PG':
    from common.db import DB, get_all
else:
    from common.dblite import DB as SQLite, get_all

logger = Logger(__name__)

DATALOADER = None
DB_SCHEMA = None
DB_SCHEMA_PLACEHOLDER = None


def get_db():
    """
    Возвращает инстанс для взаимодействия с БД.
    Не для всех методов здесь настроено взаимодействие с SQLite.
    :return: инстанс DB (PostgreSQL, SQLite)
    """
    global DB_SCHEMA_PLACEHOLDER
    if DB_VERSION == 'PG':
        assert DB_SCHEMA is not None, "Не задана схема для базы PostgreSQL"
        DB_SCHEMA_PLACEHOLDER = DB_SCHEMA + '.'
        return DB(DB_SCHEMA)
    DB_SCHEMA_PLACEHOLDER = ''
    return SQLite()


def get_uuid_from_columns(df_row, *args):
    """Генерирует UUID из значений заданных колонок строки датафрейма"""
    source = "/".join([str(df_row[column_name]) for column_name in args])
    row_hash = hashlib.md5(source.encode('utf-8'))
    return uuid.UUID(row_hash.hexdigest())


def load_doctor(tablename, filename, sheetname):
    """Загружает из Excel и заполняет таблицу врачей в БД."""
    df_src = pd.read_excel(XLS_FILEPATH + filename, sheetname, header=1)
    logger.info(f'Считано строк из Excel: {len(df_src)} [{filename}]')
    logger.debug(f'Считано из Excel:\n{df_src.head()}')

    def is_empty(value) -> bool:
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
    doctors_for_avail = []
    for i, row in df_src.iterrows():
        modality_ru: str = row['modality']
        if is_empty(modality_ru):
            print('Не задана модальность в строке:', i)
            continue
        modality_ru = modality_ru.lower().strip()
        if not check_modality(modality_ru):
            continue
        main_modality = MODALITIES_MAP[modality_ru]

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
        # schedule_type = random.choice(SCHEDULE_TYPES)
        schedule_type = row['schedule']
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
        doctors_for_avail.append([uid, day_start_time, timedelta(seconds=day_duration_sec)])

    df = pd.DataFrame(doctors, columns=DOCTOR_COLUMNS)
    db = get_db()
    if DB_VERSION == 'SQLite':
        db.convert_str(df, ['uid', 'name', 'main_modality', 'schedule_type'])
        db.convert_list(df, ['modalities'])
        db.convert_time(df, ['day_start_time', 'day_end_time', 'rest_time'])
        db.convert_bool(df, ['is_active'])
    logger.debug(df.iloc[0])
    logger.debug(f'Приготовлено для записи:\n{df.tail()}')
    logger.debug(f'Отдельная запись:\n{df.iloc[0]}')

    unique = ['name']
    db.delete(tablename, "true")
    db.upsert(df, tablename, unique)
    db.close()
    logger.info(f'Данные о врачах записаны в БД. Всего записей: {len(df)}')

    # формируем датафрейм для таблицы доступности врачей
    df_avail = pd.DataFrame(doctors_for_avail, columns=['doctor', 'day_start_time', 'time_volume'])
    df_avail = pd.concat([df_avail, df_src.loc[:, 1:]], axis='columns')

    return df_avail


def load_doctor_availability(df_avail, tablename, month_start, version, msg='Сохранено записей'):
    """"Загружает данные о доступности врачей в течение месяца из полученного датафрейма,
     который, в свою очередь был считан из Excel."""

    columns = ['uid', 'doctor', 'day_start', 'availability', 'time_volume']  # + version

    # определяем количество дней в месяце
    num_days = df_avail.columns.tolist()[-1]

    time_table = []
    for i, doctor in df_avail.iterrows():
        doctor_uid = doctor['doctor']
        work_time = doctor['time_volume']

        for day_index in range(num_days):
            day_start = datetime.combine(month_start + timedelta(days=day_index), doctor['day_start_time'])
            # флаг доступности: -1 - недоступен для распределения, 1 - доступен
            availability = 1 if doctor[day_index + 1] == 1 else -1
            time_volume = work_time if availability == 1 else time(0)
            uid = uuid.uuid4()
            time_table.append([uid, doctor_uid, day_start, availability, time_volume])

    df = pd.DataFrame(time_table, columns=columns)
    df['version'] = version

    db = get_db()
    if DB_VERSION == 'SQLite':
        db.convert_str(df, ['uid', 'version', 'doctor'])
        db.convert_datetime(df, ['day_start'])
        db.convert_time(df, ['time_volume'])

    unique = ['version', 'doctor', 'day_start']
    db.delete(tablename, f"version = '{version}' and datetime(day_start, 'start of month') = "
                         f"'{month_start.strftime('%Y-%m-%d %H:%M:%S')}'")
    db.upsert(df, tablename, unique)
    db.close()
    logger.info(f'Данные о доступности врачей записаны в БД. Всего записей: {len(df)}')


def load_summary(tablename, filename, sheetname, version):
    """"Загружает понедельный факт или план в разрезе модальностей из Excel"""

    df_src = pd.read_excel(XLS_FILEPATH + filename, sheetname, header=1)
    logger.info(f'Считано строк из Excel: {len(df_src)} [{filename}]')
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

    # df['amount'].fillna(0, inplace=True)
    df.fillna({'amount': 0}, inplace=True)
    df['version'] = version
    # формируем уникальный uid из списка уникальности записи
    unique = ['year', 'week', 'modality', 'contrast_enhancement', 'version']
    df['uid'] = df.apply(get_uuid_from_columns, args=tuple(unique), axis=1)

    logger.debug(df.head())

    db = get_db()
    if DB_VERSION == 'SQLite':
        db.convert_str(df, ['uid', 'modality', 'contrast_enhancement', 'version'])

    db.delete(tablename, f"version = '{version}'")
    db.upsert(df, tablename, unique)
    db.close()
    logger.info(f'Данные о факте работ записаны в БД. Всего записей: {len(df)}')


def load_time_norm(tablename, filename, sheetname):
    """Загружает из Excel таблицу норм времени в БД."""
    df_src = pd.read_excel(XLS_FILEPATH + filename, sheetname, header=2)
    logger.info(f'Считано строк из Excel: {len(df_src)} [{filename}]')
    logger.debug(f'Нормы времени считаны из Excel:\n{df_src.head()}')

    columns = ['modality', 'contrast_enhancement', 'min_value', 'max_value']
    df = df_src[columns].copy()

    db = get_db()
    if DB_VERSION == 'SQLite':
        db.convert_str(df, ['modality', 'contrast_enhancement'])

    logger.debug(f'Нормы времени подготовлены для записи:\n{df.head()}')
    unique = ['modality', 'contrast_enhancement']
    db.delete(tablename, "true")
    db.upsert(df, tablename, unique)
    db.close()
    logger.info(f'Данные о нормах времени записаны в БД. Всего записей: {len(df)}')


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
    """
    Генерирует график доступности врачей (рабочие дни, выходные, дни отсутствия)
    на месяц на основе шаблона.

    :param doctors: датафрейм с данными о врачах
    :param month_layout: месячный шаблон, полученный методом get_month_layout
    :return: датафрейм с данными о доступности врачей
    """

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


# deprecated
def load_doctor_availability_gen(tablename, month_start, version, msg='Сохранено записей'):
    db = get_db()

    q = (f"SELECT id, {', '.join(DOCTOR_COLUMNS)}"
         f" FROM {DB_SCHEMA_PLACEHOLDER}doctor"
         f" WHERE is_active = 't';"
         )
    with db.get_cursor() as cursor:
        cursor.execute(q)
        doctors: pd.DataFrame = get_all(cursor)

    def convert_to_time(df, columns):
        for col in columns:
            df[col] = pd.to_datetime(df[col]).dt.time

    if DB_VERSION == 'SQLite':
        convert_to_time(doctors, ['day_start_time', 'day_end_time', 'rest_time'])
    # doctors['day_start_time'] = pd.to_datetime(doctors['day_start_time']).dt.time
    # doctors['day_end_time'] = pd.to_datetime(doctors['day_end_time']).dt.time
    # doctors['rest_time'] = pd.to_datetime(doctors['rest_time']).dt.time

    month_layout = get_month_layout(month_start)
    # выход: [uid, doctor_uid, day_start, availability, time_volume]
    df, _ = generate_schedule(doctors, month_layout)
    df['version'] = version

    if DB_VERSION == 'SQLite':
        db.convert_str(df, ['uid', 'version', 'doctor', 'day_start', 'time_volume'])

    unique = ['version', 'doctor', 'day_start']
    try:
        db.upsert(df, tablename, unique)
        print(f'{msg}: {len(df)}')
    except Exception as e:
        print(repr(e))
        df.to_excel(XLS_FILEPATH + 'doctor_availability_df.xlsx')
    db.close()


def load_doctor_day_plan(tablename, month_start, version):
    db = get_db()

    q = (f"SELECT da.uid, da.time_volume, d.main_modality, d.modalities"
         f" FROM {DB_SCHEMA_PLACEHOLDER}doctor_availability as da"
         f" LEFT JOIN {DB_SCHEMA_PLACEHOLDER}doctor as d"
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

    db = get_db()
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

    db = get_db()
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
    db = get_db()
    unique = ['modality', 'contrast_enhancement']
    db.upsert(df, 'time_norm', unique)
    db.close()
    print('Записано строк:', len(df))


def data_transfer():
    """Перенос данных из PG в SQLite"""
    tablename = 'work_summary'
    schema = 'roentgen'
    pd.set_option('display.expand_frame_repr', False)

    query = f"""
        select
        *
        from {schema}.{tablename}
        where version = 'train' 
    """
    db = DB(schema)
    with db.get_cursor() as cursor:
        cursor.execute(query)
        df = get_all(cursor)
    print(f'Считано из PG:\n{df.head(10)}')
    db.close()

    # колонки version нет в целевой базе
    df.drop(columns=['version'], inplace=True)

    def create_row(row):
        return f"('{row['uid']}', '{row['created_at']}', '{row['updated_at']}'," \
               f" {row['year']}, {row['week']}, '{row['modality']}', '{row['contrast_enhancement']}', {row['amount']})"

    df['string_values'] = df.apply(create_row, axis=1)
    with pd.option_context('display.max_colwidth', -1):
        values = df['string_values'].to_string(header=False, index=False)
    values = values.replace("\n", ",\n")
    # print(f'Строка:\n{values[:200]}')
    # for row in values.split('\n')[:10]:
    #     print(row)
    # print(values[:200])
    sql_columns = 'uid, created_at, updated_at, year, week, modality, contrast_enhancement, amount'
    unique_columns = 'year, week, modality, contrast_enhancement'

    query = f"""
        INSERT INTO {tablename}({sql_columns})
        VALUES {values}
        ON CONFLICT({unique_columns})
        DO UPDATE
            SET amount = excluded.amount;
    """
    sqlite = SQLite()
    try:
        with sqlite.get_cursor() as cursor:
            cursor.execute(query)
        sqlite.commit()
        print('Данные записаны.')
    except Exception as e:
        with open(LOCAL_FOLDER + 'sqlite_query.sql', 'w') as f:
            f.write(query)
            print('Ошибка при записи:', repr(e))
    sqlite.close()


def load_data():
    """Общий метод, который загружает из файлов:
        - таблицу врачей с их параметрами и доступностью на месяц;
        - таблицу факта работ понедельно в разрезе модальностей;
        - таблицу норм времени на выполнение работ по различным модальностям.
        """
    files = ['doctor_table.xlsx', 'work_fact_by_week.xlsx', 'time_norm.xlsx']
    not_found = []
    for f in files:
        if not os.path.exists(XLS_FILEPATH + f):
            not_found.append(XLS_FILEPATH + f)
    if len(not_found) > 0:
        raise FileNotFoundError("Не найдены файлы для загрузки:\n" + '\n'.join(not_found))

    assert 'ROENTGEN.SCHEDULE_START_DATE' in os.environ, 'В переменных среды не задана дата начала расчёта графика.'
    month_start = datetime.fromisoformat(os.environ['ROENTGEN.SCHEDULE_START_DATE'])

    # определяем имя листа из даты
    doctor_sheetname = month_start.strftime('%Y-%m')
    df_avail = load_doctor('doctor', files[0], doctor_sheetname)
    load_doctor_availability(df_avail, 'doctor_availability', month_start, version='final')
    load_summary('work_summary', files[1], 'Факт работ', version='train')
    load_time_norm('time_norm', files[2], 'Нормы времени')


if __name__ == '__main__':

    logger.setup(level=logger.INFO, layout='debug')
    expand_pandas_output()

    mode = 'main'
    if DB_VERSION == 'PG':
        DB_SCHEMA = 'test' if mode == 'test' else 'roentgen'

    dataloader = DataLoader(DB_SCHEMA)
    os.environ['ROENTGEN.SCHEDULE_START_DATE'] = '2024-05-01'

    if mode == 'test':
        # генерация таблицы доступности врачей
        # load_doctor_availability('doctor_availability', start_of_month, version='base')
        # генерация записей по работе врача в течение дня
        # load_doctor_day_plan('doctor_day_plan', start_of_month, version='base')
        # генерация плана работ
        # generate_test_work_plan_summary()
        # генерация норм времени
        # generate_test_time_norm()
        pass
    else:
        load_data()
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
