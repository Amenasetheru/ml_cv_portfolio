import os
from flask import Flask
from dotenv import load_dotenv
from flask_wtf.csrf import CSRFProtect

# Charger les variables d'environnement depuis le fichier .env
load_dotenv()

# Créer l'instance de l'application Flask ici, et l'exposer directement
# au lieu de la retourner par une fonction create_app.
app = Flask(__name__)
app.config.from_mapping(
    SECRET_KEY=os.getenv('SECRET_KEY', 'super_secret_key_fallback'),
    RECAPTCHA_PUBLIC_KEY=os.getenv('RECAPTCHA_PUBLIC_KEY'),
    RECAPTCHA_PRIVATE_KEY=os.getenv('RECAPTCHA_PRIVATE_KEY')
    )

# Initialiser la protection CSRF
CSRFProtect(app)

# Importer et enregistrer les blueprints *après* la création de l'instance 'app'
# et l'initialisation de ses configurations et extensions
from . import routes # Importe le module routes
app.register_blueprint(routes.bp) # Enregistre le blueprint

# Nous n'avons plus besoin de la fonction create_app() car l'instance 'app'
# est maintenant directement disponible comme un attribut du module.
# Gunicorn cherchera 'app' dans le module 'app' (votre dossier 'app/__init__.py').
    
