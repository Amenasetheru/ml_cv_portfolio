import os
from flask import Flask, render_template, request, flash, redirect, url_for
from dotenv import load_dotenv
from flask_wtf.csrf import CSRFProtect
from flask_wtf import RecaptchaField # Assurez-vous que cette ligne est présente
from wtforms import StringField, TextAreaField, SubmitField
from wtforms.validators import DataRequired, Email, Length
import requests
import json

# Load environment variables from the .env file
# This is for local development only. Render loads environment variables directly.
load_dotenv()

# Create the Flask application instance here, and expose it directly
# instead of returning it from a create_app function.
app = Flask(__name__)
app.config.from_mapping(
    SECRET_KEY=os.getenv('SECRET_KEY', 'super_secret_key_fallback'),
    # Note: Flask-WTF expects RECAPTCHA_PUBLIC_KEY and RECAPTCHA_PRIVATE_KEY
    # Ensure your environment variables on Render are named RECAPTCHA_SITE_KEY and RECAPTCHA_SECRET_KEY
    RECAPTCHA_PUBLIC_KEY=os.getenv('RECAPTCHA_SITE_KEY'),
    RECAPTCHA_PRIVATE_KEY=os.getenv('RECAPTCHA_SECRET_KEY')
)

# --- DEBUGGING RECAPTCHA KEYS (Crucial for Render logs) ---
# These print statements will appear in your Render logs when the app starts
print(f"DEBUG_FLASK: SECRET_KEY loaded (first 5 chars): {app.config['SECRET_KEY'][:5]}...")
print(f"DEBUG_FLASK: RECAPTCHA_PUBLIC_KEY loaded: {app.config['RECAPTCHA_PUBLIC_KEY']}")
# For security, only print the last 5 characters of the private key
print(f"DEBUG_FLASK: RECAPTCHA_PRIVATE_KEY loaded (last 5 chars): {app.config['RECAPTCHA_PRIVATE_KEY'][-5:] if app.config['RECAPTCHA_PRIVATE_KEY'] else 'None'}")
# --- END DEBUGGING ---

# Initialize CSRF protection
CSRFProtect(app)

# Import and register blueprints *after* the 'app' instance is created
# and its configurations and extensions are initialized
from . import routes # Imports the routes module
app.register_blueprint(routes.bp) # Registers the blueprint

# We no longer need the create_app() function because the 'app' instance
# is now directly available as an attribute of the module.
# Gunicorn will look for 'app' in the 'app' module (your 'app/__init__.py' file).

# No need for if __name__ == '__main__': block for Render deployment as Gunicorn handles it.
# If you run locally, ensure your .env is correctly loaded or set dummy keys for testing.
