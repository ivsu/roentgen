import logging
import sys
import os
from inspect import currentframe, getframeinfo

PROJECT_ROOT = 'container'
FORMATS = {
    # формат для вывода в лог
    'log': '%(asctime)s %(levelname)s %(message)s [%(relativepath)s: %(funcName)s %(lineno)d]',
    # формат для вывода сообщений во время отладки
    'debug': '%(levelname)s: %(message)s',
}
DEFAULT_LEVEL = logging.WARNING


def logger_action(frame, throw):
    if frame:
        frameinfo = getframeinfo(frame)
        msg = f'FROM LOGGER: module: {frameinfo.filename}, line: {frameinfo.lineno}, function: {frameinfo.function}'
        if throw:
            raise RuntimeError(msg)
        elif frame:
            print(msg)
    if throw:
        raise RuntimeError("Исключение, вызванное logger.")


def get_custom_kwargs(kwargs):
    throw = kwargs.pop('throw') if 'throw' in kwargs else None
    frame = kwargs.pop('frame') if 'frame' in kwargs else None
    return throw, frame


class PackagePathFilter(logging.Filter):
    """Обрабатывает плейсхолдер relativepath в форматной строке"""

    def filter(self, record):
        pathname = record.pathname
        record.relativepath = None
        abs_sys_paths = map(os.path.abspath, sys.path)
        for path in sorted(abs_sys_paths, key=len, reverse=True):  # longer paths first
            if not path.endswith(os.sep):
                path += os.sep
            if pathname.startswith(path):
                # получаем путь от каталога проекта
                while not path.endswith(PROJECT_ROOT) and path != '/':
                    path = os.path.abspath(os.path.join(path, os.pardir))
                    # print(f'sys path: {path}')
                record.relativepath = os.path.relpath(pathname, path)
                break
        return True


class Logger(logging.Logger):
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL

    def __init__(self, name, level: int = DEFAULT_LEVEL):
        super().__init__(name, level)

        handler = logging.StreamHandler(stream=sys.stdout)
        handler.set_name('stdout')
        formatter = logging.Formatter(FORMATS['log'])
        handler.setFormatter(formatter)
        handler.addFilter(PackagePathFilter())
        self.addHandler(handler)

        # 'w' - режим перезаписи лога, 'a' - режим добавления
        # logging.basicConfig(level=logging.INFO, format=FORMAT,
        #                     filename="test.log", filemode="w")

    def setup(self, level: int = None, layout: str = None):

        for handler in self.handlers:
            if handler.get_name() == 'stdout':
                if level:
                    self.setLevel(level)
                    # handler.setLevel(level)
                if layout:
                    if layout not in FORMATS.keys():
                        raise RuntimeError(f'Неизвестный формат лога: {layout}. Допустимо: {list(FORMATS.keys())}')

                    formatter = logging.Formatter(FORMATS[layout])
                    handler.setFormatter(formatter)

    def debug(self, msg, *args, **kwargs):
        throw, frame = get_custom_kwargs(kwargs)
        super().debug(msg, *args, **kwargs)
        logger_action(frame, throw)

    def info(self, msg, *args, **kwargs):
        throw, frame = get_custom_kwargs(kwargs)
        super().info(msg, *args, **kwargs)
        logger_action(frame, throw)

    def warning(self, msg, *args, **kwargs):
        throw, frame = get_custom_kwargs(kwargs)
        super().warning(msg, *args, **kwargs)
        logger_action(frame, throw)

    def error(self, msg, *args, **kwargs):
        throw, frame = get_custom_kwargs(kwargs)
        super().error(msg, *args, **kwargs)
        logger_action(frame, throw)


if __name__ == '__main__':
    logger = Logger(__name__)
    logger.setup(level=logger.DEBUG, layout='debug')
    logger.warning('Test warning')
    logger.debug('First debug message')
    logger.debug('Second debug message', frame=currentframe())
    logger.debug('Third debug message with Exception', throw=True, frame=currentframe())
