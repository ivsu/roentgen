from typing import Iterable, Any

import os
import sqlite3
import pandas as pd
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


if __name__ == '__main__':
    os.chdir('..')
    logger.setup(level=logger.DEBUG, layout='debug')

    db = DB()
    # check connection
    with db.get_cursor() as cursor:
        cursor.execute("select 'dblite ready' as message;")
        result = get_all(cursor)
    db.close()
    print(result)

