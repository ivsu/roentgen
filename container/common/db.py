import psycopg2 as pg
from psycopg2 import extensions as pg_ext
from sshtunnel import SSHTunnelForwarder
import pandas as pd
import json
import os

from common.updater import Updater
from common.logger import Logger
from settings import CREDENTIALS_FILE

"""
TODO: что-то сделать с сообщением о входящих подключениях
"""

# В версии 37.*, 38.* модуля cryptography возникает CryptographyDeprecationWarning при установлении туннеля:
# /Users/ivan/opt/anaconda3/envs/trader/lib/python3.10/site-packages/paramiko/transport.py:219:
# CryptographyDeprecationWarning: Blowfish has been deprecated "class": algorithms.Blowfish
# workarounds:
# 1. Понизить версию модуля cryptography до 36.0.0.
# 2. import warnings
#    from cryptography.utils import CryptographyDeprecationWarning
#    warnings.filterwarnings("ignore", category=CryptographyDeprecationWarning)


logger = Logger(__name__)


def get_all(cursor: pg_ext.cursor) -> pd.DataFrame:
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


def paginator(cursor: pg_ext.cursor, limit: int = 100) -> pd.DataFrame:
    """
    Генератор заданного количества записей из курсора.
    Данные датафрейма не копируются, он просто индексирует полученный список кортежей.
    :param cursor: курсор базы данных
    :param limit: количество записей
    :return: список записей
    """
    while True:
        try:
            if cursor.description:
                page = cursor.fetchmany(limit)
                if not page:
                    break
                df = pd.DataFrame(page, columns=[desc[0] for desc in cursor.description])
                yield df
            else:
                break
        except Exception as e:
            logger.error("Ошибка при получении записей и конвертации в DataFrame: " + repr(e))
            raise


class DB:

    def __init__(self, db_schema=None):
        self.tunnel = None
        self.connection = None
        self.cursor = None
        with open(CREDENTIALS_FILE, 'r') as fp:
            self.credentials: dict = json.load(fp)

        if db_schema:
            self.schema = db_schema
        elif 'PG_SCHEMA' in self.credentials.keys():
            self.schema = self.credentials['PG_SCHEMA']
        else:
            self.schema = 'public'

        self.updater = Updater(self.schema)

    def set_tunnel(self):
        """Establishes SSH-tunnel"""
        cr = self.credentials

        try:
            logger.debug('Создание SSH-туннеля ...')
            self.tunnel = SSHTunnelForwarder(
                cr['SSH_HOST'],
                ssh_username=cr['SSH_USERNAME'],
                ssh_private_key=cr['SSH_PKEY'],
                ssh_private_key_password=cr['SSH_PKEY'],
                remote_bind_address=(cr['DB_HOST'], cr['PORT'])
            )
            self.tunnel.start()
            logger.debug('SSH-туннель установлен.')

        except Exception as e:
            logger.error('Ошибка при установлении SSH-туннеля: ' + repr(e))
            raise

    def connect(self):
        """Creates DB session"""
        cr = self.credentials
        if 'SSH_HOST' in cr:
            self.set_tunnel()
            host = cr['LOCALHOST']
            port = self.tunnel.local_bind_port
        else:
            host = cr['DB_HOST']
            port = cr['PORT']

        try:
            self.connection = pg.connect(
                host=host,
                port=port,
                user=cr['PG_USERNAME'],
                password=cr['PG_DB_PW'],
                database=cr['PG_DB_NAME']
            )
            logger.debug('Соединение с БД установлено.')

        except Exception as e:
            logger.error('Ошибка при установлении соединения с БД: ' + repr(e))
            self.tunnel.stop()
            raise

    def get_cursor(self) -> pg_ext.cursor:
        """Returns the DB cursor and optionally connect to DB"""
        if self.connection is None:
            self.connect()
        self.cursor = self.connection.cursor()
        return self.cursor

    def close(self):
        """Fully disconnects from DB"""
        if self.cursor:
            self.cursor.close()
            self.cursor = None
        if self.connection:
            self.connection.close()
            self.connection = None
        if self.tunnel:
            self.tunnel.stop()
            self.tunnel = None
        logger.debug('Соединение закрыто.')

    def upsert(self, df: pd.DataFrame, tablename: str, unique: list[str]):
        """
        Сохраняет датафрейм в базу данных методом UPSERT и подтверждает транзакцию.

        :param df: источник данных
        :param tablename: имя целевой таблицы
        :param unique: список имён полей, определяющих уникальность записи
        """
        try:
            with self.get_cursor() as cursor:
                self.updater.upsert(cursor, df, tablename, unique)
            self.connection.commit()
        except (Exception, pg.DatabaseError) as e:
            self.connection.rollback()
            logger.error("Ошибка при попытке записи в БД: " + repr(e))
            raise

    def upsert_with_cursor(self, cursor, df, tablename, unique):
        """
        Сохраняет датафрейм в базу данных методом UPSERT.
        Подтверждение транзакции должно быть сделано вовне.
        :param cursor: курсор БД
        :param df: источник данных
        :param tablename: имя целевой таблицы
        :param unique: список имён полей, определяющих уникальность записи
        """
        self.updater.upsert(cursor, df, tablename, unique)

    def get_count(self, tablename: str, where: str = 'true'):
        with self.get_cursor() as cursor:
            try:
                q = f"SELECT count(*) as amount" \
                    f" FROM {self.schema}.{tablename} as tablename" \
                    f" WHERE {where};"
                cursor.execute(q)
                pd = get_all(cursor)
                return pd.iloc[0]['amount']
            except Exception as e:
                logger.error(f"Ошибка при выполнении запроса: {repr(e)}\nQuery:\n{q}")
                raise


class Transaction:
    """
    Позволяет запомнить методы с аргументами и выполнить их в одной транзакции.
    Все методы первым аргументом должны иметь параметр cursor.
    """

    def __init__(self, db: DB):
        self.db = db
        self.methods = []
        self.method_args = []
        self.method_kwargs = []

    def set(self, method, *args, **kwargs):
        self.methods.append(method)
        self.method_args.append(args)
        self.method_kwargs.append(kwargs)

    def call(self):
        try:
            with self.db.get_cursor() as cursor:
                for i, method in enumerate(self.methods):
                    if not hasattr(self.db, method):
                        raise RuntimeError(f'Класс DB не содержит метода {method}')

                    # print(f'method_args: {self.method_args[i]}')
                    # print(f'method_kwargs: {self.method_kwargs[i]}')

                    getattr(self.db, method)(cursor, *self.method_args[i], **self.method_kwargs[i])
            self.db.connection.commit()

        except (Exception, pg.DatabaseError) as e:
            self.db.connection.rollback()
            logger.error("Ошибка при выполнении транзакции: " + repr(e))
            raise


if __name__ == '__main__':
    os.chdir('..')
    logger.setup(level=logger.DEBUG, layout='debug')

    db = DB()
    # check connection
    test_cursor = db.get_cursor()

    db.close()



