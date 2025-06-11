from app import create_app # Importe l'usine d'application depuis le package 'app'

# Crée l'instance de l'application Flask
# Cette variable 'app' sera celle que Gunicorn cherchera.
app = create_app()

if __name__ == '__main__':
    # Cette partie est UNIQUEMENT pour le développement local
    # NE SERA PAS EXÉCUTÉE par Gunicorn sur Render
    app.run(debug=False) # Important: Passez en debug=False pour la production
        

