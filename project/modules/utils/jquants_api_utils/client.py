from dotenv import load_dotenv
from utils.paths import Paths
import jquantsapi
import os
from utils.singleton import SingletonMeta

class Cli(metaclass=SingletonMeta):
    def __init__(self):
        load_dotenv(f'{Paths.MODULES_FOLDER}/.env')
        self.cli = jquantsapi.Client(
            mail_address=os.getenv('JQUANTS_EMAIL'),
            password=os.getenv('JQUANTS_PASS')
        )

cli = Cli().cli

if __name__ == '__main__':
    print(cli)