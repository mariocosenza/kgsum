import functools
import os
import hashlib
import jwt
import psutil
from flasgger import Swagger
from flask import Flask
from flask import request, jsonify, g
from werkzeug.utils import secure_filename

from service.generate_profile_service import load_classifier
from src.service.file_upload_service import UPLOAD_FOLDER, allowed_file
from src.service.generate_profile_service import generate_profile_service_store, generate_profile_service

AUTH = os.getenv("CLERK_MIDDLEWARE_ENABLED", "false").lower()
# PEM public key as string
PUBLIC_KEY = os.getenv("CLERK_MIDDLEWARE_PUBLIC_KEY", "false").lower()

app = Flask(__name__)

SECRET_KEY = os.environ.get("SECRET_KEY")
if not SECRET_KEY:
    raise ValueError("SECRET_KEY environment variable must be set.")

Swagger(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
active_requests = 0

load_classifier()


def clerk_jwt_required(fn):
    @functools.wraps(fn)
    async def wrapper(*args, **kwargs):
        if AUTH == "true":
            auth_header = request.headers.get("Authorization", None)
            if not auth_header or not auth_header.startswith("Bearer "):
                return jsonify({"error": "Missing or invalid Authorization header"}), 401
            token = auth_header.split(" ", 1)[1]
            try:
                payload = jwt.decode(
                    token,
                    PUBLIC_KEY,
                    algorithms=["RS256"],
                    options={"verify_aud": False}
                )
                g.current_user = payload
            except jwt.PyJWTError as e:
                app.logger.error(f"Token validation error: {e}")  # Only log internally
                return jsonify({"error": "Token validation error"}), 401

        return await fn(*args, **kwargs)

    return wrapper

def check_system_load(func):
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        global active_requests
        cpu = psutil.cpu_percent(interval=0.1)
        ram = psutil.virtual_memory().percent
        if cpu > 90 or ram > 90:
            return jsonify({
                "error": "Server overloaded",
                "cpu_usage": cpu,
                "ram_usage": ram
            }), 500

        if active_requests >= 8:
            return jsonify({"error": "Too many active requests", "active_requests": active_requests}), 429

        active_requests += 1
        try:
            return await func(*args, **kwargs)
        finally:
            active_requests -= 1

    return wrapper


# ----- Endpoints -----
@app.route('/api/v1/profile/sparql', methods=['POST'])
@check_system_load
@clerk_jwt_required
async def sparql_profile():
    """
    Generate a profile from a SPARQL endpoint.
    ---
    tags:
      - Profile
    consumes:
      - application/json
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          properties:
            endpoint:
              type: string
              description: SPARQL endpoint URL
            store:
              type: boolean
              description: Whether to store the profile result
          required:
            - endpoint
    responses:
      200:
        description: Successfully generated profile
        schema:
          type: object
          properties:
            status:
              type: string
            data:
              type: object
      400:
        description: Bad request (missing or invalid parameters)
      429:
        description: Too many active requests
      500:
        description: Server overloaded or internal error
    """
    data = request.get_json() or {}
    endpoint = data.get('endpoint')
    if not endpoint:
        return jsonify({"error": "Missing 'endpoint' parameter"}), 400

    store_value = data.get('store', False)
    if isinstance(store_value, bool):
        store_flag = store_value
    elif isinstance(store_value, str) and store_value.lower() in ['true', 'false']:
        store_flag = store_value.lower() == 'true'
    else:
        store_flag = False
    try:
        if store_flag:
            result = await generate_profile_service_store(endpoint, sparql=True)
        else:
            result = await generate_profile_service(endpoint, sparql=True)
    except Exception as e:
        app.logger.error(f"Profile generation failed: {e}")  # Only log internally
        return jsonify({"error": "Profile generation failed"}), 500

    return jsonify(result), 200


@app.route('/api/v1/profile/file', methods=['POST'])
@check_system_load
@clerk_jwt_required
async def rdf_profile():
    """
    Generate a profile from an uploaded RDF file.
    ---
    tags:
      - Profile
    consumes:
      - multipart/form-data
    parameters:
      - in: formData
        name: file
        type: file
        required: true
        description: The RDF file to upload
      - in: query
        name: store
        type: boolean
        required: false
        description: Whether to store the profile result
    responses:
      200:
        description: Successfully generated profile
        schema:
          type: object
          properties:
            status:
              type: string
            data:
              type: object
      400:
        description: Bad request (missing file or invalid parameters)
      429:
        description: Too many active requests
      500:
        description: Server overloaded or internal error
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "File type not allowed"}), 400

    filename = secure_filename(file.filename)
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], hashlib.sha256(str(filename).encode()).hexdigest())
    try:
        file.save(save_path)
    except Exception as e:
        app.logger.error(f"Error saving file: {e}")  # Only log internally
        return jsonify({"error": "Error saving file"}), 500

    store_param = request.args.get('store', 'false').lower()
    store_flag = store_param in ('true', '1', 'yes')

    try:
        if store_flag:
            result = await generate_profile_service_store(save_path, sparql=False)
        else:
            result = await generate_profile_service(save_path, sparql=False)
    except Exception as e:
        app.logger.error(f"Profile generation failed: {e}")  # Only log internally
        return jsonify({"error": "Profile generation failed"}), 500

    if result is None:
        return jsonify({"error": "Profile generation failed unexpectedly"}), 500

    return jsonify(result), 200


if __name__ == '__main__':
    app.run()