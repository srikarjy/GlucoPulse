## PM2 Services

| Port | Name | Type |
|------|------|------|
| 8000 | glucopulse-api-8000 | FastAPI inference API (`serving.api`, `.venv`) |
| 8770 | glucopulse-mcp-8770 | Public MCP server, dev tokens (`agents.public_mcp.server`, `.venv-pa`) |

Docker services (TimescaleDB, Kafka, prior-auth stack) stay in docker-compose on OrbStack.
The MCP server needs `export GLUCOPULSE_MCP_DEV_SECRET=$(openssl rand -hex 32)` before its first start.

**Terminal Commands:**
```bash
pm2 start ecosystem.config.cjs   # First time
pm2 start all                    # After first time
pm2 stop all / pm2 restart all
pm2 start {name} / pm2 stop {name}
pm2 logs / pm2 status / pm2 monit
pm2 save                         # Save process list
pm2 resurrect                    # Restore saved list
```
