// PM2 process list for local development (macOS). Databases, Kafka and the prior-auth
// stack run in Docker (OrbStack) via docker-compose; PM2 only runs the Python app processes.
// Python runs from the project venvs directly (interpreter: 'none'), never a global Python.
//
// glucopulse-mcp-8770 needs GLUCOPULSE_MCP_DEV_SECRET (>= 32 chars) in the shell that starts it:
//   export GLUCOPULSE_MCP_DEV_SECRET=$(openssl rand -hex 32)
// The secret is read from the environment and never written to this file.
module.exports = {
  apps: [
    {
      name: 'glucopulse-api-8000',
      cwd: __dirname,
      script: '.venv/bin/python',
      args: '-m uvicorn serving.api:app --host 127.0.0.1 --port 8000',
      interpreter: 'none',
      autorestart: true,
      env: { PYTHONUNBUFFERED: '1' },
    },
    {
      name: 'glucopulse-mcp-8770',
      cwd: __dirname,
      script: '.venv-pa/bin/python',
      args: '-m agents.public_mcp.server',
      interpreter: 'none',
      autorestart: true,
      env: {
        PYTHONUNBUFFERED: '1',
        GLUCOPULSE_MCP_AUTH_MODE: 'dev',
        GLUCOPULSE_MCP_DEV_SECRET: process.env.GLUCOPULSE_MCP_DEV_SECRET || '',
      },
    },
  ],
};
