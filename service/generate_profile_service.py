from src.predict_category import predict_category_remote, predict_category_local


async def generate_profile_service(endpoint: str, sparql = False):
    if sparql:
        result = await predict_category_remote(endpoint)
        return result
    else:
        await predict_category_local(endpoint)
    return

def generate_profile_service_store():
    return