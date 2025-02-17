from deep_translator import GoogleTranslator

def preprocess():
    translated = GoogleTranslator(source='auto', target='en').translate("keep it up, you are awesome")