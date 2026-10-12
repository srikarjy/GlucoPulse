import { createHmac } from "node:crypto";
import type { Page } from "@playwright/test";
import { SECRET } from "../playwright.config";

const b64 = (o: object | Buffer) => Buffer.from(o instanceof Buffer ? o : JSON.stringify(o)).toString("base64url");
/** Mint an API-audience HS256 token the same way priorauth.mcp_server.auth does. */
export function token(role: string, sub = `e2e-${role}`, ttl = 3600): string {
  const now = Math.floor(Date.now() / 1000);
  const body = `${b64({ alg: "HS256", typ: "JWT" })}.${b64({ sub, role, aud: "priorauth-api", iat: now, exp: now + ttl })}`;
  return `${body}.${createHmac("sha256", SECRET).update(body).digest("base64url")}`;
}

export async function signIn(page: Page, role: string, sub?: string) {
  await page.context().clearCookies();
  await page.goto("/login");
  await page.getByLabel("API token").fill(token(role, sub));
  await page.getByRole("button", { name: "Sign in" }).click();
  await page.waitForURL((u) => !u.pathname.startsWith("/login"));
}

/** Pick a scenario by profile and (optionally) a predicate on its label, then run it. Returns the request id. */
export async function runScenario(page: Page, profile: string, opts: { paraphrased?: boolean; injection?: boolean } = {}) {
  await page.goto("/playground");
  await page.getByLabel("Scenario type").selectOption(profile);
  const select = page.getByLabel("Scenario", { exact: true });
  await select.locator("option", { hasText: "·" }).first().waitFor({ state: "attached" });   // scenarios loaded and filtered
  const options = await select.locator("option").all();
  const wanted = [];
  for (const o of options) {
    const t = (await o.textContent()) ?? "";
    if (!t.includes("·")) continue;
    if (opts.paraphrased !== undefined && t.includes("paraphrased") !== opts.paraphrased) continue;
    wanted.push(await o.getAttribute("value"));
  }
  await select.selectOption(wanted[0]!);
  if (opts.injection) await page.getByRole("button", { name: /prompt-injection attempt/ }).click();
  await page.getByRole("button", { name: /Run through the agents/ }).click();
  await page.getByText(/^Request REQ-/).first().waitFor();
  const id = (await page.getByText(/^Request REQ-/).first().locator("code").textContent())!;
  await page.getByRole("heading", { name: "Draft recommendation" }).waitFor({ timeout: 30_000 });
  return id;
}
