# Use official Python image
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Copy all files to container
COPY . /app

# Upgrade pip and install dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy English model
RUN python -m spacy download en_core_web_md

# Expose port 8000
EXPOSE 8000

# Run the API (Option B layout = app.main:app)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
