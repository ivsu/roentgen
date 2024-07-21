import os

CREDENTIALS_FILE = 'assets/credentials.json'
PROJECT_FOLDER = 'drive/MyDrive/university/roentgen/'

try:
    import google.colab
    RUN_ENV = 'COLAB'
    CREDENTIALS_FILE = PROJECT_FOLDER + 'credentials.json'
except:
    RUN_ENV = 'IDE'

os.environ['RUN_ENV'] = RUN_ENV
