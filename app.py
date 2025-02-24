from flasgger import Swagger
from flask import Flask
app = Flask(__name__)
swagger = Swagger(app)



@app.route('/api/v1/sparql/category')
def sparql_category():
    return ''

@app.route('/api/v1/dump/category')
def rdf_category():
    return ''

@app.route('/api/v1/search')
def search():
    return ''

@app.route('/api/v1/info')
def info_from_db():
    return ''

if __name__ == '__main__':
    app.run()
