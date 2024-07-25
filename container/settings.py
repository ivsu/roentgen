import os

# используемая БД
DB_VERSION = ['PG', 'SQLite'][1]

PROJECT_FOLDER = '/content/roentgen/container/'
BOTS_FOLDER = PROJECT_FOLDER + 'bots/'
DRIVE_FOLDER = '/content/drive/MyDrive/university/roentgen/'  # TODO: исключить использование
LOCAL_FOLDER = '/Users/ivan/Documents/CIFROPRO/Проекты/Нейронки/Расписание рентген-центра/'

try:
    # выполнение в колабе
    import google.colab
    RUN_ENV = 'COLAB'
    # if not os.path.exists('/content/drive/'):
    #     from google.colab import drive
    #     drive.mount('/content/drive/')

except:
    RUN_ENV = 'IDE'
    PROJECT_FOLDER = '/Users/ivan/PycharmProjects/roentgen/container/'
    BOTS_FOLDER = LOCAL_FOLDER + 'bots/'

# настройки для подключения к БД (только для PG)
CREDENTIALS_FILE = PROJECT_FOLDER + 'assets/credentials.json'

os.environ['RUN_ENV'] = RUN_ENV

if not os.path.exists(BOTS_FOLDER):
    os.mkdir(BOTS_FOLDER)

