# Utiliser une image de base Python
FROM python:3.8-slim

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Copier les fichiers de dépendances et les installer
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copier le reste des fichiers de l'application
COPY . .

# Exposer le port 5000 pour Flask
EXPOSE 8000

# Démarrer l'application Flask
CMD ["flask", "run", "--host=0.0.0.0"]
