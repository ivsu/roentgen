import os

CREDENTIALS_FILE = 'assets/credentials.json'
DRIVE_FOLDER = '/content/drive/MyDrive/university/roentgen/'
PROJECT_FOLDER = '/content/roentgen/container/'

try:
    import google.colab
    RUN_ENV = 'COLAB'
    CREDENTIALS_FILE = PROJECT_FOLDER + 'credentials.json'
except:
    RUN_ENV = 'IDE'
    PROJECT_FOLDER = '/Users/ivan/PycharmProjects/roentgen/container/'

os.environ['RUN_ENV'] = RUN_ENV
