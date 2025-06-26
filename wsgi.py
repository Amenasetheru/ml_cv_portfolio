# Importe l'instance de l'application 'app' directement depuis le package 'app'
# (qui est défini dans app/__init__.py)
from app import app

if __name__ == '__main__':
    # Cette partie est UNIQUEMENT pour le développement local
    # NE SERA PAS EXÉCUTÉE par Gunicorn sur Render
    app.run(debug=True) # Mettez debug=True pour le développement local, False pour la production
    
        