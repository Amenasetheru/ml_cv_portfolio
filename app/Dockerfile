# Utilise une image Python 3.12 basée sur Debian Bookworm slim (minimale)
FROM python:3.12-slim-bookworm

# Installe les outils de compilation essentiels et d'autres outils nécessaires
# pour certaines dépendances Python qui pourraient en avoir besoin.
# 'apt-get clean' et 'rm -rf /var/lib/apt/lists/*' réduisent la taille de l'image.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Créer un utilisateur non-root (bonne pratique de sécurité)
RUN useradd -m -u 1000 user
USER user
    
# Ajouter le répertoire des exécutables locaux au PATH pour l'utilisateur 'user'
ENV PATH="/home/user/.local/bin:$PATH"

# Empêche la génération de fichiers .pyc et désactive le buffering de Python
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Installe uv en copiant le binaire précompilé (méthode officielle et rapide)
# Assurez-vous que la version de uv ici (0.6.14) est celle que vous souhaitez.
COPY --from=ghcr.io/astral-sh/uv:0.6.14 /uv /uvx /bin/

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Copier pyproject.toml et uv.lock en premier pour optimiser le cache Docker
# Ces fichiers sont nécessaires pour 'uv sync'
COPY --chown=user pyproject.toml uv.lock ./

# Installer toutes les dépendances définies dans uv.lock
# uv sync --locked utilise uv.lock pour installer toutes les dépendances du projet
RUN uv sync --locked

# Copier le reste des fichiers de l'application dans le répertoire de travail
COPY --chown=user . /app

# Exposer le port sur lequel l'application Flask écoutera
EXPOSE 7860

# Commande de démarrage de l'application Flask avec Gunicorn
# 'app:create_app()' signifie : Dans le module 'app' (votre dossier 'app/'),
# appeler la fonction 'create_app()' qui retourne l'instance de votre application Flask.
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "app:create_app()"]
