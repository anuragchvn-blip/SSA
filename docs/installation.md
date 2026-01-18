# Installation Guide

## Prerequisites

- Python 3.10 or higher
- Node.js 18 or higher
- PostgreSQL 12 or higher (optional, for production)
- Git

## System Requirements

### Minimum Specifications
- CPU: Dual-core processor
- RAM: 4GB
- Storage: 2GB available space
- Network: Broadband connection for Space-Track API access

### Recommended Specifications
- CPU: Quad-core processor
- RAM: 8GB
- Storage: SSD with 10GB available space
- Network: Stable broadband connection

## Backend Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-org/ssa-engine.git
cd ssa-engine
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

If no requirements.txt exists, install the core dependencies:

```bash
pip install fastapi uvicorn sqlalchemy psycopg2-binary python-jose[cryptography] passlib[bcrypt] python-multipart sgp4 numpy pandas scikit-learn
```

### 4. Environment Configuration

Create a `.env` file in the project root:

```env
# Database Configuration
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=ssa_engine
POSTGRES_USER=ssa_user
POSTGRES_PASSWORD=your_secure_password

# Space-Track API Credentials
SPACE_TRACK_USERNAME=your_username
SPACE_TRACK_PASSWORD=your_password

# JWT Secret
JWT_SECRET_KEY=your_super_secret_jwt_key_here

# Application Settings
DEBUG=true
LOG_LEVEL=INFO
```

### 5. Database Setup

Initialize the database:

```bash
# If using the provided script
python -c "
from src.data.database import init_db
init_db()
"
```

Or run the database initialization script:

```bash
python db_init_current.py
```

### 6. Start the Backend Server

```bash
cd e:\SSA
python -c "import sys; sys.path.insert(0, '.'); from src.api.main import app; import uvicorn; uvicorn.run(app, host='0.0.0.0', port=8000)"
```

The backend will be available at `http://localhost:8000`.

## Frontend Installation

### 1. Navigate to Frontend Directory

```bash
cd frontend
```

### 2. Install Node Dependencies

```bash
npm install
```

### 3. Configure Environment Variables

Create a `.env.local` file in the frontend directory:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_APP_NAME=SSA Operations Console
NEXT_PUBLIC_ENVIRONMENT=development
```

### 4. Build the Application

```bash
npm run build
```

### 5. Start the Frontend Server

```bash
npm run start
```

The frontend will be available at `http://localhost:3000`.

## Docker Installation (Alternative)

If Docker is preferred for deployment:

### 1. Build Docker Images

```bash
docker-compose build
```

### 2. Start Services

```bash
docker-compose up -d
```

Both frontend and backend services will be available.

## Verification Steps

### 1. Check Backend Health

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "service": "SSA Conjunction Analysis Engine",
  "version": "1.0.0",
  "timestamp": "2026-01-17T14:30:00.000000Z"
}
```

### 2. Verify Database Connection

```bash
curl -H "Authorization: Bearer test-token" http://localhost:8000/status
```

### 3. Check Frontend Access

Navigate to `http://localhost:3000` in your browser.

## Troubleshooting

### Common Issues

#### Port Already in Use
If ports 8000 or 3000 are in use, modify the startup commands:
- Backend: Change `port=8000` to another port in the uvicorn command
- Frontend: Set PORT environment variable (`PORT=3001 npm start`)

#### Database Connection Issues
- Verify PostgreSQL is running
- Check credentials in `.env` file
- Ensure the database exists

#### Space-Track API Access
- Verify credentials in environment
- Check internet connectivity
- Confirm account status with Space-Track

### Service Startup Order
1. Database (if external)
2. Backend API
3. Frontend

## Post-Installation

### Initial Data Population

Run the TLE population script to load sample data:

```bash
python populate_tles.py
```

Run the conjunction event creation script:

```bash
python create_conjunctions.py
```

### Security Considerations

- Change default JWT secret in production
- Use HTTPS in production environments
- Implement proper firewall rules
- Regular security audits of dependencies