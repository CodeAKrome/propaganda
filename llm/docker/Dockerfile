FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .

# Expose default port
EXPOSE 8101

# Run the application
ENTRYPOINT ["python", "app.py"]
CMD ["--port", "8101"]
