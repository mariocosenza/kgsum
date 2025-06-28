ALLOWED_EXTENSIONS = {'xml', 'trig', 'ttl', 'nq', 'nt', 'rdf'}
UPLOAD_FOLDER = './'

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS