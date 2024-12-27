from dotenv import load_dotenv
import paths
import jquantsapi
import os
from common_utils import SingletonMeta

class Cli(metaclass=SingletonMeta):
    def __init__(self):
        load_dotenv(f'{paths.MODULES_FOLDER}/.env')
        self.cli = jquantsapi.Client(
            mail_address=os.getenv('JQUANTS_EMAIL'),
            password=os.getenv('JQUANTS_PASS')
        )

cli = Cli().cli

if __name__ == '__main__':
    cli