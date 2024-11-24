import os

# используемая БД
DB_VERSION = ['PG', 'SQLite'][1]

DRIVE_FOLDER = '/content/drive/MyDrive/university/roentgen/'
LOCAL_FOLDER = '/Users/ivan/Documents/CIFROPRO/Проекты/Нейронки/Расписание рентген-центра/'

try:
    # выполнение в колабе
    import google.colab
    RUN_ENV = 'COLAB'
    PROJECT_FOLDER = '/content/roentgen/container/'
    # папка для загрузки данных из Excel
    XLS_FILEPATH = '/content/local/src_files/'
    # BOTS_FOLDER = DRIVE_FOLDER + 'bots/'
    BOTS_FOLDER = '/content/local/bots/'
    # if not os.path.exists('/content/drive/'):
    #     from google.colab import drive

except ModuleNotFoundError:
    RUN_ENV = 'IDE'
    PROJECT_FOLDER = '/Users/ivan/PycharmProjects/roentgen/container/'
    XLS_FILEPATH = '/Users/ivan/Documents/CIFROPRO/Проекты/Нейронки/' \
                   'Расписание рентген-центра/dataset/Таблицы для загрузки/'
    BOTS_FOLDER = LOCAL_FOLDER + 'bots/'

# настройки для подключения к БД (только для PG)
CREDENTIALS_FILE = PROJECT_FOLDER + 'assets/credentials.json'

os.environ['RUN_ENV'] = RUN_ENV

if not os.path.exists(BOTS_FOLDER):
    os.makedirs(BOTS_FOLDER)

if __name__ == '__main__':
    print('RUN_ENV:', RUN_ENV)

