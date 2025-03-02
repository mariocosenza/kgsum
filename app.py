from flasgger import Swagger
from flask import Flask, request

from service.generate_profile_service import generate_profile_service

app = Flask(__name__)
swagger = Swagger(app)



@app.route('/api/v1/sparql/profile', methods=['POST'])
async def sparql_profile():
    data =  request.get_json()
    result = await generate_profile_service(data['endpoint'], sparql=True)
    return {
        'profile': result
    }

@app.route('/api/v1/dump/profile')
def rdf_profile():
    return ''

@app.route('/api/v1/search')
def search():
    return ''

@app.route('/api/v1/info')
def info_from_db():
    return ''

if __name__ == '__main__':
    app.run()
