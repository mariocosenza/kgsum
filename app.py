import functools
import os

import psutil
from flasgger import Swagger
from flask import Flask, request, flash, jsonify
from werkzeug.utils import secure_filename

from service.file_upload_service import allowed_file, UPLOAD_FOLDER
from service.generate_profile_service import generate_profile_service_store, generate_profile_service

app = Flask(__name__)
app.secret_key = "secret"
swagger = Swagger(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

active_requests = 0


def check_system_load(func):
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        global active_requests
        cpu = psutil.cpu_percent(interval=0.1)
        ram = psutil.virtual_memory().percent
        if cpu > 90 or ram > 80:
            return jsonify({
                "error": "Server overloaded",
                "cpu_usage": cpu,
                "ram_usage": ram
            }), 500

        if active_requests >= 8:
            return jsonify({"active_requests": active_requests}), 429

        active_requests += 1
        try:
            return await func(*args, **kwargs)
        finally:
            active_requests -= 1

    return wrapper


@app.route('/api/v1/profile/sparql', methods=['POST'])
@check_system_load
async def sparql_profile():
    data = request.get_json()
    if 'store' in data:
        store = bool(data['store'])
        if store:
            result = await generate_profile_service_store(data['endpoint'], sparql=True)
        else:
            result = await generate_profile_service(data['endpoint'], sparql=True)
        return jsonify(result)
    # Fallback in case no 'store' param is provided
    result = await generate_profile_service(data['endpoint'], sparql=True)
    return jsonify(result)


@app.route('/api/v1/profile/file', methods=['POST'])
@check_system_load
async def rdf_profile():
    if 'file' not in request.files:
        flash('No file part')
        return {"error": "No file part"}  # Return error as JSON

    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return {"error": "No selected file"}  # Return error as JSON

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        filepath = os.path.join(UPLOAD_FOLDER, filename)

        if 'store' in request.args:
            store = request.args['store'].lower()
            if store in ('true', '1', 'yes'):
                result = generate_profile_service_store(filepath, sparql=False)
            else:
                result = await generate_profile_service(filepath, sparql=False)
        else:
            result = await generate_profile_service(filepath, sparql=False)

        if result is None:
            return {"error": "Profile generation failed unexpectedly."}

        return jsonify(result)

    return {"error": "Error Uploading File"}

if __name__ == '__main__':
    app.run()
