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
load_dotenv()

# Create the Flask application instance here, and expose it directly
# instead of returning it from a create_app function.
app = Flask(__name__)
app.config.from_mapping(
    SECRET_KEY=os.getenv('SECRET_KEY', 'super_secret_key_fallback'),
    RECAPTCHA_PUBLIC_KEY=os.getenv('RECAPTCHA_SITE_KEY'), # Lire de RECAPTCHA_SITE_KEY
    RECAPTCHA_PRIVATE_KEY=os.getenv('RECAPTCHA_SECRET_KEY') # Lire de RECAPTCHA_SECRET_KEY
)

# --- DEBUGGING RECAPTCHA KEYS (Ajouté) ---
# These print statements will appear in your Render logs
print(f"DEBUG: RECAPTCHA_PUBLIC_KEY in app.config: {app.config['RECAPTCHA_PUBLIC_KEY']}")
# Pour des raisons de sécurité, ne pas afficher la clé privée en entier
print(f"DEBUG: RECAPTCHA_PRIVATE_KEY in app.config (last 5 chars): {app.config['RECAPTCHA_PRIVATE_KEY'][-5:] if app.config['RECAPTCHA_PRIVATE_KEY'] else 'None'}")
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

# NOTE: The 'if __name__ == '__main__':' block at the end of the file should be removed
# or commented out for Render deployments, as Gunicorn handles the app startup.
# If it's still there and defining 'app.run(debug=True)', it might interfere.
# Given your current setup, it seems you've already moved away from a create_app() function,
# so this note is more a general reminder.
