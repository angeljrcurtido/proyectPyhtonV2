# Usa una imagen base de Python
FROM python:3.9-slim

# Establece el directorio de trabajo en /app
WORKDIR /app

# Copia el archivo requirements.txt al directorio de trabajo
COPY requirements.txt .

# Instala las dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Copia todo el código de la aplicación al directorio de trabajo
COPY . .

# Exponer el puerto en el que correrá la aplicación (Flask usa por defecto el 5000)
EXPOSE 5000

# Especificar la variable de entorno para producción
ENV FLASK_ENV=production

# Comando para ejecutar la aplicación

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "120", "app:app"]

