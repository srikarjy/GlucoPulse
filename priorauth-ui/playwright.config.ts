import { defineConfig } from "@playwright/test";
import { tmpdir } from "node:os";
import { join } from "node:path";

// E2E runs the REAL stack: the Python demo server (API + worker, deterministic agents, synthetic data) and the Next.js
// production build. Tokens are minted in the tests with the same shared secret.
export const SECRET = "e2e-only-secret-0123456789abcdef0123456789abcdef";
const API_PORT = 18082, WEB_PORT = 13000;
export default defineConfig({
  testDir: "./e2e", timeout: 60_000, retries: 0, workers: 1, fullyParallel: false, reporter: [["list"]],
  use: { baseURL: `http://127.0.0.1:${WEB_PORT}`, trace: "retain-on-failure" },
  webServer: [
    {
      command: `.venv-pa/bin/python -u -m priorauth.scripts.demo_server --port ${API_PORT} --db ${join(tmpdir(), `pa-e2e-${Date.now()}.db`)}`,
      cwd: "..", url: `http://127.0.0.1:${API_PORT}/healthz`, reuseExistingServer: false, timeout: 60_000,
      env: { PRIORAUTH_JWT_SECRET: SECRET },
    },
    {
      command: `npx next start -p ${WEB_PORT} -H 127.0.0.1`, url: `http://127.0.0.1:${WEB_PORT}/login`, reuseExistingServer: false, timeout: 60_000,
      env: { PRIORAUTH_API_URL: `http://127.0.0.1:${API_PORT}` },
    },
  ],
});
