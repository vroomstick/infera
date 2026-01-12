# Infera Daily Workflow

## Morning: Start Up

```bash
# 1. Open OrbStack (if not running)
open -a OrbStack

# 2. Start the database
cd ~/Documents/infera
docker compose up -d db

# 3. Verify (optional)
docker ps
```

## Work: Develop & Test

```bash
# Analyze a new company
make fetch TICKER=NVDA

# Start the API
make api

# Run tests
make test
```

### DBeaver Connection

| Field | Value |
|-------|-------|
| Host | `localhost` |
| Port | `5432` |
| Database | `infera` |
| Username | `infera` |
| Password | `infera_dev_password` |

## Evening: Shut Down (Optional)

```bash
# Stop containers (data persists)
docker compose down

# OR: Stop and delete all data
docker compose down -v
```

## Quick Reference

| Task | Command |
|------|---------|
| Start DB | `docker compose up -d db` |
| Stop DB | `docker compose down` |
| Fresh restart | `docker compose down -v && docker compose up -d db` |
| View logs | `docker compose logs -f db` |
| SQL shell | `docker exec -it infera-db psql -U infera -d infera` |

