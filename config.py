import os

class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "default_secret")

class ProductionConfig(Config):
    DEBUG = False

class DevelopmentConfig(Config):
    DEBUG = True