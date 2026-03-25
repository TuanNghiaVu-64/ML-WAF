# WAF Logs Location Guide

## Issue 1: `/var/log/modsecurity/` doesn't exist

The OWASP ModSecurity CRS Docker image **does not create** the `/var/log/modsecurity/` directory by default. Instead, logs are configured to go to **stdout/stderr**, which Docker captures.

## Where Logs Actually Are

### 1. **Docker Compose Logs (Primary Location)**
All ModSecurity logs go to stdout/stderr and are captured by Docker:

```bash
# Make sure you're in the ModSec_demo directory first!
cd C:\Users\ADMIN\Repos\ModSec_demo

# View logs
docker compose logs dvwa_modsec_crs

# Follow logs in real-time
docker compose logs -f dvwa_modsec_crs

# Last 50 lines
docker compose logs --tail=50 dvwa_modsec_crs

# Filter for ModSecurity messages
docker compose logs dvwa_modsec_crs | findstr /i "modsec security"
```

### 2. **Apache Access Logs**
```bash
docker exec dvwa-modsec-crs cat /var/log/apache2/access.log
docker exec dvwa-modsec-crs tail -20 /var/log/apache2/access.log
```

### 3. **Apache Error Logs (includes ModSecurity)**
```bash
docker exec dvwa-modsec-crs tail -20 /proc/self/fd/2
```

## Issue 2: "no configuration file provided: not found"

This error occurs when you run `docker compose` from the wrong directory. You must be in the directory containing `docker-compose.yml`.

**Solution:**
```bash
cd C:\Users\ADMIN\Repos\ModSec_demo
docker compose logs dvwa_modsec_crs
```

## Creating Persistent Log Directories

If you want to create the `/var/log/modsecurity/` directory and persist logs to your host:

1. **Add volume mounts to docker-compose.yml:**
```yaml
dvwa_modsec_crs:
  volumes:
    - ./logs/modsec_audit:/var/log/modsecurity/audit
    - ./logs/apache:/var/log/apache2
```

2. **Create directories:**
```bash
mkdir logs\modsec_audit
mkdir logs\apache
```

3. **Restart container:**
```bash
docker compose up -d dvwa_modsec_crs
```

## Quick Commands

```bash
# View all recent logs
docker compose logs dvwa_modsec_crs --tail=100

# View only ModSecurity blocks
docker compose logs dvwa_modsec_crs | findstr /i "blocked deny 403"

# View access log
docker exec dvwa-modsec-crs tail -20 /var/log/apache2/access.log

# Check if ModSecurity is logging
docker compose logs dvwa_modsec_crs | findstr /i "ModSecurity"
```

