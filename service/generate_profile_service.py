import logging
import os
from src.generate_profile import generate_profile, generate_and_store_profile

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def generate_profile_service(endpoint: str, sparql: bool = False) -> dict:
    if sparql:
        result = await generate_profile(endpoint=endpoint)
        return result
    else:
        try:
            result = await generate_profile(file=endpoint)
            os.remove(path= endpoint)
            return result
        except Exception as e:
            raise ValueError(f'Cannot process the given Knowledge Graph {e}')

async def generate_profile_service_store(endpoint: str, sparql = False):
    if sparql:
        result = await generate_and_store_profile(endpoint=endpoint)
        return result
    else:
        try:
            result = await generate_and_store_profile(file=endpoint)
            os.remove(path=endpoint)
            return result
        except Exception as e:
            raise ValueError(f'Cannot process the given Knowledge Graph {e}')