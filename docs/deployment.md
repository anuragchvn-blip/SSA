# Deployment Guide

## Production Architecture

### Recommended Infrastructure

For production deployment, the system should be deployed across multiple tiers:

```
┌─────────────────┐    ┌──────────────────────┐    ┌─────────────────┐
│   Load Balancer │───▶│   API Servers       │───▶│   Database      │
│   (NGINX/HAProxy)│   │   (Multiple Nodes)  │    │   (PostgreSQL)  │
└─────────────────┘    └──────────────────────┘    └─────────────────┘
         │                        │
         ▼                        ▼
┌─────────────────┐    ┌──────────────────────┐
│   CDN/Edge      │    │   Monitoring        │
│   (Cloudflare)  │    │   (Prometheus/Grafana)│
└─────────────────┘    └──────────────────────┘
```

## Environment Preparation

### Server Requirements

#### Backend Servers
- Minimum: 4 vCPUs, 8GB RAM
- Recommended: 8 vCPUs, 16GB RAM
- OS: Ubuntu 20.04 LTS or CentOS 8
- Disk: SSD with 50GB+ available space

#### Database Server
- Minimum: 4 vCPUs, 16GB RAM
- Recommended: 8 vCPUs, 32GB RAM
- Disk: SSD with 100GB+ available space for logs and backups
- PostgreSQL 12+ with extensions: pg_stat_statements, pg_cron

#### Load Balancer
- NGINX Plus or HAProxy Enterprise (for production)
- SSL termination capability
- Health check configuration

### Security Preparations

#### Firewall Configuration
```bash
# API server firewall rules
ufw allow ssh
ufw allow http
ufw allow https
ufw allow from DATABASE_SERVER_IP to any port 5432
ufw enable
```

#### SSL Certificate Setup
```bash
# Using Let's Encrypt with Certbot
sudo apt-get install certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

## Backend Deployment

### 1. Environment Configuration

Create a production `.env` file:

```env
# Database Configuration
POSTGRES_HOST=your-db-server.com
POSTGRES_PORT=5432
POSTGRES_DB=ssa_prod
POSTGRES_USER=ssa_prod_user
POSTGRES_PASSWORD=strong_production_password

# Space-Track API Credentials
SPACE_TRACK_USERNAME=your_prod_username
SPACE_TRACK_PASSWORD=your_prod_password

# JWT Secret (generate securely)
JWT_SECRET_KEY=your_very_long_and_random_secret_key_here

# Application Settings
DEBUG=false
LOG_LEVEL=WARNING
DATABASE_URL=postgresql://ssa_prod_user:password@your-db-server.com:5432/ssa_prod

# CORS Configuration
ALLOWED_ORIGINS=https://your-frontend-domain.com,https://your-backend-domain.com

# Rate Limiting
RATE_LIMIT_REQUESTS_PER_MINUTE=100
```

### 2. Process Management

Use systemd for process management. Create `/etc/systemd/system/ssa-api.service`:

```ini
[Unit]
Description=SSA Conjunction Analysis Engine API
After=network.target

[Service]
Type=simple
User=ssa-user
WorkingDirectory=/opt/ssa-engine
EnvironmentFile=/opt/ssa-engine/.env
ExecStart=/opt/ssa-engine/venv/bin/uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 4
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start the service:
```bash
sudo systemctl daemon-reload
sudo systemctl enable ssa-api
sudo systemctl start ssa-api
```

### 3. Reverse Proxy Configuration

Configure NGINX as a reverse proxy. Create `/etc/nginx/sites-available/ssa-api`:

```nginx
server {
    listen 443 ssl http2;
    server_name api.your-domain.com;

    ssl_certificate /path/to/certificate.crt;
    ssl_certificate_key /path/to/private.key;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Increase timeouts for long-running operations
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }

    # Health check endpoint
    location /health {
        access_log off;
        proxy_pass http://127.0.0.1:8000/health;
    }
}

server {
    listen 80;
    server_name api.your-domain.com;
    return 301 https://$server_name$request_uri;
}
```

## Frontend Deployment

### 1. Build for Production

```bash
cd frontend
npm run build
```

### 2. NGINX Configuration

Create `/etc/nginx/sites-available/ssa-frontend`:

```nginx
server {
    listen 443 ssl http2;
    server_name your-domain.com;

    ssl_certificate /path/to/certificate.crt;
    ssl_certificate_key /path/to/private.key;

    root /opt/ssa-frontend/.next;
    index index.html;

    location / {
        try_files $uri $uri/ /index.html;
    }

    # API proxy
    location /api/ {
        proxy_pass http://api.your-domain.com/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header Referrer-Policy "no-referrer-when-downgrade" always;
    add_header Content-Security-Policy "default-src 'self' http: https: data: blob: 'unsafe-inline'" always;
}

server {
    listen 80;
    server_name your-domain.com;
    return 301 https://$server_name$request_uri;
}
```

### 3. Enable Sites

```bash
sudo ln -s /etc/nginx/sites-available/ssa-frontend /etc/nginx/sites-enabled/
sudo ln -s /etc/nginx/sites-available/ssa-api /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

## Database Deployment

### 1. PostgreSQL Configuration

Optimize PostgreSQL for the workload. Add to `postgresql.conf`:

```
# Memory settings
shared_buffers = 4GB
effective_cache_size = 12GB
work_mem = 16MB
maintenance_work_mem = 512MB

# Connection settings
max_connections = 200
max_worker_processes = 8
max_parallel_workers_per_gather = 4
max_parallel_workers = 8

# Logging
log_statement = 'all'
log_min_duration_statement = 1000
log_line_prefix = '%t [%p]: [%l-1] user=%u,db=%d,app=%a,client=%h '
log_checkpoints = on
log_connections = on
log_disconnections = on

# Backup settings
wal_level = replica
archive_mode = on
archive_command = 'cp %p /path/to/archive/%f'
max_wal_senders = 3
```

### 2. Database Initialization

```sql
-- Create production database and user
CREATE DATABASE ssa_prod WITH OWNER ssa_prod_user;
GRANT ALL PRIVILEGES ON DATABASE ssa_prod TO ssa_prod_user;

-- Enable required extensions
\c ssa_prod
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;
```

## Monitoring and Observability

### 1. Application Logging

Configure structured logging in production:

```python
# In settings or config
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "json": {
            "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
            "format": "%(asctime)s %(name)s %(levelname)s %(filename)s %(lineno)d %(message)s"
        }
    },
    "handlers": {
        "json_file": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": "/var/log/ssa-engine/app.log",
            "formatter": "json",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5
        }
    },
    "root": {
        "level": "WARNING",
        "handlers": ["json_file"]
    }
}
```

### 2. Metrics Collection

Deploy Prometheus and Grafana for metrics collection:

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'ssa-api'
    static_configs:
      - targets: ['api-server-ip:8000']
    metrics_path: '/metrics'
    scrape_interval: 30s
```

### 3. Health Checks

Implement readiness and liveness probes for container orchestration:

- Liveness: `GET /health` - should return 200 within 5s
- Readiness: Check database connectivity and API responsiveness

## Backup and Recovery

### 1. Database Backups

Schedule automated backups:

```bash
# Daily backup script
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backup/ssa-db"
mkdir -p $BACKUP_DIR

pg_dump -h localhost -U ssa_prod_user -d ssa_prod > $BACKUP_DIR/ssa_prod_$DATE.sql
gzip $BACKUP_DIR/ssa_prod_$DATE.sql

# Keep only last 30 days of backups
find $BACKUP_DIR -name "*.sql.gz" -mtime +30 -delete
```

### 2. Configuration Backup

Backup environment files and SSL certificates regularly.

## Scaling Recommendations

### Horizontal Scaling
- Add API server instances behind load balancer
- Scale database connections proportionally
- Monitor load balancer metrics for capacity planning

### Vertical Scaling
- Increase instance sizes for processing-intensive operations
- Upgrade to faster storage for database
- Monitor memory and CPU utilization trends

### Database Scaling
- Implement read replicas for query distribution
- Consider sharding for large datasets
- Use connection pooling (PgBouncer)

## Security Hardening

### API Security
- Implement rate limiting (100 requests/minute per IP)
- Use OAuth2 with PKCE for authentication
- Validate and sanitize all inputs
- Implement proper CORS policies

### Network Security
- Deploy WAF (Web Application Firewall)
- Implement DDoS protection
- Use VPN for administrative access
- Encrypt data in transit and at rest

### Access Control
- Implement role-based access control (RBAC)
- Regular security audits of permissions
- Multi-factor authentication for admin access
- Regular credential rotation

## Rollback Procedures

### 1. Versioned Deployments

Maintain previous versions for quick rollback:

```bash
# Before deployment, tag current version
git tag deploy-$(date +%Y%m%d-%H%M%S)
git push origin deploy-$(date +%Y%m%d-%H%M%S)
```

### 2. Rollback Process

```bash
# Stop current service
sudo systemctl stop ssa-api

# Revert to previous version
git checkout previous-version-tag
pip install -r requirements.txt
# Restart service
sudo systemctl start ssa-api
```

This deployment guide provides a robust, production-ready setup for the SSA Conjunction Analysis Engine.