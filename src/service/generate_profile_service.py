import logging
import os

import src.generate_profile
from config import Config
from src.generate_profile import generate_profile, generate_and_store_profile

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
Config.init_configuration()

def load_classifier():
    src.generate_profile.load_predictor()


async def generate_profile_service(endpoint: str, sparql: bool = False) -> dict:
    if sparql:
        result = await generate_profile(endpoint=endpoint)
        return result
    else:
        try:
            result = await generate_profile(file=endpoint)
            os.remove(path=endpoint)
            return result
        except Exception as e:
            raise ValueError(f'Cannot process the given Knowledge Graph {e}')


async def generate_profile_service_store(endpoint: str, sparql=False):
    if sparql:
        result = await generate_and_store_profile(endpoint=endpoint, base_uri=Config.BASE_DOMAIN)
        return result
    else:
        try:
            result = await generate_and_store_profile(file=endpoint, base_uri=Config.BASE_DOMAIN)
            os.remove(path=endpoint)
            return result
        except Exception as e:
            raise ValueError(f'Cannot process the given Knowledge Graph {e}')
