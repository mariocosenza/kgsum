import os

from src.predict_category import predict_category_remote, predict_category_local


async def generate_profile_service(endpoint: str, sparql: bool = False) -> object:
    if sparql:
        result = await predict_category_remote(endpoint)
        return result
    else:
        try:
            result = predict_category_local(endpoint)
            os.remove(path= endpoint)
            return result
        except Exception as e:
            print(e)
    return ''

def generate_local_profile(path):
    try:
        result = predict_category_local(path)
        os.remove(path=path)
        return result
    except Exception as e:
        print(e)
    return ''

def generate_profile_service_store(endpoint: str, sparql = False):
    return