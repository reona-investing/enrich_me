from dotenv import load_dotenv
import paths
import jquantsapi
import os

def get_client():
    load_dotenv(f'{paths.MODULES_FOLDER}/.env')
    #%% J-Quants APIクライアントの初期化
    cli = jquantsapi.Client(
        mail_address=os.getenv('JQUANTS_EMAIL'),
        password=os.getenv('JQUANTS_PASS')
    )
    return cli
