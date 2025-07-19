import os

ALLOWED_EXTENSIONS = {'xml', 'trig', 'ttl', 'nq', 'nt', 'rdf'}
UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', os.path.abspath('./uploads'))

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS