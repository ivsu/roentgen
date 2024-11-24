import os
import sqlite3
import pandas as pd
from datetime import datetime, time, timedelta

from common.logger import Logger
from settings import PROJECT_FOLDER

DB_PATH = 'db/roentgen.sqlite'
logger = Logger(__name__)


class CursorContextManager():

    def __init__(self, cursor: sqlite3.Cursor):
        self.cursor = cursor

    def __enter__(self):
        return self.cursor

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cursor.close()


def get_all(cursor: sqlite3.Cursor) -> pd.DataFrame:
    """
    Возвращает все записи из курсора.
    Данные датафрейма не копируются, он просто индексирует полученный список кортежей.
    """
    try:
        if cursor.description:
            result: list[tuple] = cursor.fetchall()
            return pd.DataFrame(result, columns=[desc[0] for desc in cursor.description])
        else:
            return pd.DataFrame([])
    except Exception as e:
        logger.error("Ошибка при получении записей и конвертации в DataFrame: " + repr(e))
        raise


class DB:
    def __init__(self, db_path=None):
        self.connection = None
        self.cursor = None
        self.db_path = db_path if db_path else DB_PATH

    def connect(self):
        """Creates DB session"""
        try:
            self.connection = sqlite3.connect(PROJECT_FOLDER + DB_PATH)
            logger.debug('Соединение с БД установлено.')

        except Exception as e:
            logger.error('Ошибка при установлении соединения с БД: ' + repr(e))
            raise

    def get_cursor(self) -> CursorContextManager:
        """Returns the DB cursor and optionally connect to DB"""
        if self.connection is None:
            self.connect()
        self.cursor = self.connection.cursor()
        return CursorContextManager(self.cursor)

    def commit(self):
        self.connection.commit()

    def close(self):
        """Полностью отключается от DB"""
        if self.cursor:
            self.cursor.close()
            self.cursor = None
        if self.connection:
            self.connection.close()
            self.connection = None
        logger.debug('Соединение закрыто.')

    def prepare_values(self):
        pass

    def upsert(self, df, tablename, unique):

        columns_list = df.columns.to_list()
        columns = ', '.join(columns_list)
        unique = ', '.join(unique)
        fields = ',\n'.join([f'\t{f} = excluded.{f}' for f in columns_list])
        values = '\n\t(' + '),\n\t('.join([', '.join([f'{v}' for v in row.values]) for _, row in df.iterrows()]) + ")"

        q = (
            f'INSERT INTO {tablename} ({columns})\n'
            f'VALUES {values}\n'
            f'ON CONFLICT ({unique})\n'
            'DO UPDATE SET\n'
            f'{fields};'
        )
        with self.get_cursor() as curr:
            curr.execute(q)
            self.connection.commit()

    def delete(self, tablename, where_condition):

        q = (
            f"DELETE FROM {tablename}\n"
            f"WHERE\n"
            f"{where_condition};"
        )
        with self.get_cursor() as curr:
            curr.execute(q)
            self.connection.commit()

    def convert_str(self, df, columns: list):
        for col in columns:
            df[col] = df[col].apply(lambda x: "'" + str(x) + "'")
            # df.loc[:, col] = df.loc[:, col].apply(lambda x: "'" + str(x) + "'")

    def convert_list(self, df, columns: list):
        for col in columns:
            df[col] = df[col].apply(lambda x: "'{" + ','.join(x) + "}'")

    def convert_bool(self, df, columns: list):
        for col in columns:
            df[col] = df[col].apply(lambda b: "'t'" if b else "'f'")


    def convert_datetime(self, df, columns: list):
        for col in columns:
            df[col] = df[col].apply(lambda dt: f"'{dt.strftime('%Y-%m-%d %H:%M:%S')}'")

    def convert_time(self, df, columns: list):
        """Конвертирует время из timedelta в строковое представление."""
        def f(t):
            if pd.isna(t):
                return "'00:00:00'"
            if isinstance(t, (time, datetime)):
                return f"'{t:%H:%M:%S}'"
            if isinstance(t, timedelta):
                return f"'{datetime.min + t:%H:%M:%S}'"
            raise TypeError (f'Неизвестный тип: {type(t)} - для значения: {t}')
            # ts = t.total_seconds()
            # hours, remainder = divmod(ts, 3600)
            # minutes, seconds = divmod(remainder, 60)
            # return f"'{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}'"

        for col in columns:
            df[col] = df[col].apply(f)


if __name__ == '__main__':
    os.chdir('..')
    logger.setup(level=logger.DEBUG, layout='debug')

    db = DB()
    # check connection
    with db.get_cursor() as cursor:
        cursor.execute("select 'dblite ready' as message;")
        result = get_all(cursor)

    # tablename = 'test_table'
    # unique = ['f1', 'f2']
    # df = pd.DataFrame({'f1': [10, 11, 12], 'f2': [100, 110, 120], 'f3': [200, 202, 209]})
    # db.upsert(df, tablename, unique)

    db.close()
    print(result)

    td = timedelta(seconds=100)
    v = datetime.min + td
    print(f"'{v:%H:%M:%S}'")

