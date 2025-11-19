import os
os.environ['CURL_CA_BUNDLE'] = ''

from huggingface_hub import login

from src.evaluation import run
from src.config import Config


if __name__ == "__main__":

    config = Config()

    if config.read_model_from_huggingface:
        login(token=config.hugging_face_token)

    run(config=config)
