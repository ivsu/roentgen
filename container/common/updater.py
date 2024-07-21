import psycopg2 as pg
import psycopg2.extras as extras
from psycopg2.extensions import register_adapter, AsIs
import numpy as np
import pandas as pd
import logging
# import utils

# маппинг типов колонок на тип данных для создания временной таблицы
ARRAY_TYPE_MAP = {
    '_text': 'text[]',
    '_float4': 'real[]',
    '_int2': 'smallint[]',
    '_varchar': 'varchar[]',
}


def adapt_numpy_int64(numpy_int64):
    return AsIs(numpy_int64)


# check the value is not None and is not NaN and not empty string
def isempty(value):
    return value is None or value != value or value == ''


def _constructor(src_table, trg_table, columns, unique):
    """returns a dict with number of strings for query construction
    """

    # имена колонок с типами
    newtable = ',\n'.join('%s %s' % tup for tup in columns)
    # имена колонок без типов
    cols = ', '.join(f'{tup[0]}' for tup in columns)
    # присваивание полей для конструкции SET
    set_clause = ', '.join(f'{tup[0]} = {trg_table}.{tup[0]}'
                           for tup in columns)
    # условие соединения таблиц
    condition = ' AND '.join(f'{trg_table}.{fld} = {src_table}.{fld}'
                             for fld in unique)
    # имена колонок целевой таблицы
    tmpcol = ', '.join(f'{trg_table}.{tup[0]}' for tup in columns)
    # условие пустых значений исходной таблицы
    isnull = ' AND '.join(f'{src_table}.{fld} IS NULL' for fld in unique)

    # print(f'newtable:\n{newtable}')
    # print(f'newcol:\n{newcol}')
    # print(f'set_clause:\n{set_clause}')
    # print(f'condition:\n{condition}')
    # print(f'tmpcol:\n{tmpcol}')
    # print(f'isnull:\n{isnull}')

    return dict(
        newtable=newtable,
        cols=cols,
        set_clause=set_clause,
        condition=condition,
        tmpcol=tmpcol,
        isnull=isnull,
    )


def get_columns(cursor: pg.extensions.cursor, schema: str, tablename: str):
    """
    Возвращает сведения о колонках таблицы БД из information_schema.

    :param cursor: курсор базы данных
    :param schema: имя схемы базы данных
    :param tablename: имя таблицы в БД
    :returns: список колонок таблицы в виде кортежей:
              [(column_name, data_type), ...]
    """

    # получим имена колонок
    q = "SELECT column_name, data_type, udt_name" \
        "  FROM information_schema.columns" \
        f" WHERE table_schema = '{schema}'" \
        f" and table_name = '{tablename}'" \
        "  and column_name != 'id'"
    try:
        cursor.execute(q)
        columns = cursor.fetchall()
    except (Exception, pg.DatabaseError) as e:
        logging.error(f'Error by recieving table "{tablename}" columns: {e}\n'
                      f'Query: {q}\n', exc_info=True)
        raise

    # processes array data types
    rebuild = []
    for c in columns:
        dt = c[1]
        if dt.lower() == 'array':
            if c[2] not in ARRAY_TYPE_MAP:
                raise RuntimeError('Не задана обработка для типа массива:', c[2])
            dt = ARRAY_TYPE_MAP[c[2]]
        elif dt.lower() == 'user-defined':  # кастомные типы данных
            dt = c[2]

        rebuild.append(tuple([c[0], dt]))

    if len(rebuild) == 0:
        logging.error('Empty list recieved by table columns query.', )
        return None

    return rebuild


class Updater:
    """ Executes workflows update by client"""

    def __init__(self, schema: str):
        # self.db = db
        self.db_schema = schema

    def setup(self, settings):
        # self.settings = settings
        # self.client = settings.get_current('client')
        pass

    def upsert(self, cursor: pg.extensions.cursor, df: pd.DataFrame, tablename: str, unique: list[str]):
        """
        Updates data in table from DataFrame.

        :param cursor: DB cursor
        :param df: data to UPSERT
        :param tablename: table name in DB
        :param unique: list of unique fields
        """

        # for entirely updateble tables "index" column should be provided
        # print(f'unique: {unique}')
        if isempty(unique):
            if 'index' not in df.columns:
                logging.error('Для колонок, для которых не задан уникальный'
                              ' ключ unique, должна быть сформирована колонка'
                              ' ключа таблицы "index".')
                return
            unique = ['index']

        # for np.int64 to bigint correct converiion
        register_adapter(np.int64, adapt_numpy_int64)

        # делаем корректный кастинг UUID
        pg.extras.register_uuid()

        # get table columns from DB: (name, type)
        db_columns = get_columns(cursor, self.db_schema, tablename)

        if db_columns is None:
            raise RuntimeError(f'Failed to get columns for table "{tablename}"')

        # remove table columns that are not in dataframe (from API response)
        columns = [tup for tup in db_columns if tup[0] in list(df)]
        # column names list
        col_names = [tup[0] for tup in columns]

        tmp_table = 'newvals'
        s = _constructor(tablename, tmp_table, columns, unique)

        query = f"""
            BEGIN;

            -- сохраним записи для UPSERT во временной таблице
            CREATE TEMPORARY TABLE {tmp_table} ({s['newtable']});
            INSERT INTO {tmp_table} ({s['cols']}) VALUES %s;

            -- эксклюзивная блокировка целевой таблицы
            LOCK TABLE {self.db_schema}.{tablename} IN EXCLUSIVE MODE;

            -- обновление существующих записей
            UPDATE {self.db_schema}.{tablename}
            SET {s['set_clause']}
            FROM {tmp_table}
            WHERE {s['condition']};

            -- вставка новых записей
            INSERT INTO {self.db_schema}.{tablename} ({s['cols']})
            SELECT {s['tmpcol']}
            FROM {tmp_table}
            LEFT OUTER JOIN {self.db_schema}.{tablename} ON ({s['condition']})
            WHERE {s['isnull']};

            DROP TABLE {tmp_table};
            COMMIT;
            """

        # prepare data and arrange columns in accordance of DB,
        # columns missed in DB table will be removed from dataframe
        tuples = [tuple(x) for x in df[col_names].to_numpy()]
        # TODO: поиск дублей
        # uids = [str(t[0]) for t in tuples]
        # for i in range(len(uids)):
        #     for j in range(i + 1, len(uids)):
        #         if uids[i] == uids[j]:
        #             print(f'double: {uids[i]}')
        try:
            extras.execute_values(cursor, query, tuples)
        except (Exception, pg.DatabaseError) as e:
            logging.error(f'Ошибка при записи в БД: {e}\n'
                          f'query: {query}\n'
                          f'tuples[0] values: {tuples[0]}', exc_info=True)
            # TODO: вывод дублирующих записей
            # import re
            # m = re.search('Key \\(uid\\)=\\((.+?)\\) already exists.', repr(e))
            # if m:
            #     found = m.group(1)
            #     print(f'found: {found}: len = {len(found)}')
            #     print('tuples:')
            #     for t in tuples:
            #         if str(t[0]) == found:
            #             print(t)
            raise
