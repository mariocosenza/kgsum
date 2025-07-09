#!/bin/sh
exec gunicorn -b :5000 --timeout 1800 --access-logfile - --error-logfile - app:app