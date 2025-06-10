import os
from flask import Flask
from dotenv import load_dotenv
from flask_wtf.csrf import CSRFProtect

# Load environment variables from .env file
load_dotenv()

def create_app():
    app = Flask(__name__)
    app.config.from_mapping(
        SECRET_KEY=os.getenv('SECRET_KEY', 'super_secret_key_fallback'),
        RECAPTCHA_PUBLIC_KEY=os.getenv('RECAPTCHA_PUBLIC_KEY'),
        RECAPTCHA_PRIVATE_KEY=os.getenv('RECAPTCHA_PRIVATE_KEY')
    )

    # Initialize CSRF protection
    CSRFProtect(app)

    # Import and register blueprints inside app context
    with app.app_context():
        from . import routes
        app.register_blueprint(routes.bp)

    return app

