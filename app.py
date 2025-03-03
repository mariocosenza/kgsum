import os
from http.client import responses

from flasgger import Swagger
from flask import Flask, request, flash, redirect
from werkzeug.utils import secure_filename

from service.file_upload_service import allowed_file, UPLOAD_FOLDER
from service.generate_profile_service import generate_profile_service, generate_profile_service_store, \
    generate_local_profile

app = Flask(__name__)
swagger = Swagger(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/api/v1/sparql/profile', methods=['POST'])
async def sparql_profile():
    data =  request.get_json()
    if 'store' in request.args:
        store = request.args.get('store')
        if store:
            result = await generate_profile_service_store(data['endpoint'], sparql=True)
        else:
            result = await generate_profile_service(data['endpoint'], sparql=True)
        return {
            'profile': result
        }

@app.route('/api/v1/dump/profile', methods=['POST'])
async def rdf_profile():
    if 'file' not in request.files:
        flash('No file part')
        return {'error': 'No file part'} # Return error as JSON

    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return {'error': 'No selected file'} # Return error as JSON

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        filepath = os.path.join(UPLOAD_FOLDER, filename) # Use os.path.join for path construction

        if 'store' in request.args:
            store = request.args['store'].lower() # Ensure lowercase for boolean check
            if store in ('true', '1', 'yes'):
                result = await generate_profile_service_store(filepath, sparql=False)
            else:
                result = generate_local_profile(filepath)
        else:
            result = generate_local_profile(filepath)

        if result is None:
            return {'error': 'Profile generation failed unexpectedly.'}

        return {
            'profile': result
        }

    return {'error': 'Error Uploading File'}

@app.route('/api/v1/search')
def search():
    return ''

@app.route('/api/v1/info')
def info_from_db():
    return ''

if __name__ == '__main__':
    app.run()
