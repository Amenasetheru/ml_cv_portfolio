# Utilise une image Python 3.12 basée sur Debian Bookworm slim (minimale)
FROM python:3.12-slim-bookworm

# Installe les outils de compilation essentiels et d'autres outils nécessaires.
# Exécuté en tant que root (utilisateur par défaut de l'image FROM).
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    locales \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Configure les locales pour éviter les avertissements et assurer la compatibilité
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

# Installe Gunicorn globalement (en tant que root) pour s'assurer qu'il est dans le PATH système
# Ceci contourne tout problème potentiel d'installation ou de PATH pour l'utilisateur 'user'.
RUN pip install --no-cache-dir gunicorn

# Créer un utilisateur non-root pour exécuter l'application (bonne pratique de sécurité)
RUN useradd -m -u 1000 user
USER user
    
# Ajouter le répertoire des exécutables locaux de l'utilisateur au PATH
# Ceci est important pour les exécutables de paquets installés par 'uv sync' pour l'utilisateur 'user'.
ENV PATH="/home/user/.local/bin:$PATH"

# Empêche la génération de fichiers .pyc et désactive le buffering de Python
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Installe uv en copiant le binaire précompilé (méthode officielle et rapide)
# Notez le '/bin/uv' à la fin pour un chemin d'exécutable clair.
COPY --from=ghcr.io/astral-sh/uv:0.6.14 /uv /uvx /bin/uv

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Copier pyproject.toml et uv.lock en premier pour optimiser le cache Docker layer
# Ces fichiers sont nécessaires pour 'uv sync'.
COPY --chown=user pyproject.toml uv.lock ./

# Installer toutes les dépendances définies dans uv.lock pour l'utilisateur 'user'.
# Gunicorn est déjà installé globalement, donc cela installera les autres dépendances Flask, etc.
RUN uv sync --locked

# Copier le reste des fichiers de l'application dans le répertoire de travail
# Cela copiera aussi 'app.py' et le dossier 'app/'.
COPY --chown=user . /app

# Exposer le port sur lequel l'application Flask écoutera
EXPOSE 7860

# Commande de démarrage de l'application Flask avec Gunicorn
# 'app:app' fait référence à l'instance 'app' DANS le module 'app' (votre fichier 'app/__init__.py').
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "app:app"]
