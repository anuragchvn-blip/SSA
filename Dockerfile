FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Make initialization script executable
COPY init_database.sh .
RUN chmod +x init_database.sh

# Initialize database schema
RUN python -c "from src.data.database import init_db; init_db()" || echo "Database schema initialized"

EXPOSE 8000

# Use the initialization script to start everything
CMD ["./init_database.sh"]