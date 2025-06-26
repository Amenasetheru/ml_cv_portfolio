import os
from flask import Flask
from dotenv import load_dotenv
from flask_wtf.csrf import CSRFProtect

# Load environment variables from the .env file
load_dotenv()

# Create the Flask application instance here, and expose it directly
# instead of returning it from a create_app function.
app = Flask(__name__)
app.config.from_mapping(
    SECRET_KEY=os.getenv('SECRET_KEY', 'super_secret_key_fallback'),
    RECAPTCHA_PUBLIC_KEY=os.getenv('RECAPTCHA_PUBLIC_KEY'),
    RECAPTCHA_PRIVATE_KEY=os.getenv('RECAPTCHA_PRIVATE_KEY')
    )

# Initialize CSRF protection
CSRFProtect(app)

# Import and register blueprints *after* the 'app' instance is created
# and its configurations and extensions are initialized
from . import routes # Imports the routes module
app.register_blueprint(routes.bp) # Registers the blueprint

# We no longer need the create_app() function because the 'app' instance
# is now directly available as an attribute of the module.
# Gunicorn will look for 'app' in the 'app' module (your 'app/__init__.py' file).
