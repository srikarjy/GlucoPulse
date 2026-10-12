import AxeBuilder from "@axe-core/playwright";
import { expect, test } from "@playwright/test";
import { runScenario, signIn, token } from "./helpers";

test.describe.configure({ mode: "serial" });

test("login rejects a bad token and a wrong-audience token", async ({ page }) => {
  await page.goto("/login");
  await page.getByLabel("API token").fill("not-a-token");
  await page.getByRole("button", { name: "Sign in" }).click();
  await expect(page.locator("[role=alert]:not(#__next-route-announcer__)")).toContainText("rejected");
  expect(new URL(page.url()).pathname).toBe("/login");
});

test("unauthenticated visitors are redirected to login", async ({ page }) => {
  await page.context().clearCookies();
  await page.goto("/queue");
  await expect(page).toHaveURL(/\/login$/);
});

test("playground: a failing scenario yields an advisory draft routed to a physician", async ({ page }) => {
  await signIn(page, "playground");
  await runScenario(page, "fail");
  await expect(page.getByText("AI-generated draft, advisory only")).toBeVisible();
  await expect(page.getByText("Draft: does not support").first()).toBeVisible();
  await expect(page.getByText("Physician required")).toBeVisible();
  await expect(page.getByText("✓ draft matches")).toBeVisible();
  await expect(page.getByText(/Final decision|Approved by/)).toHaveCount(0);    // nothing in this view is a decision
});

test("playground: injection is flagged, caps confidence and forces a physician, even when the criteria are genuinely met", async ({ page }) => {
  await signIn(page, "playground");
  await runScenario(page, "meets", { paraphrased: false, injection: true });
  await expect(page.getByText("prompt_injection").first()).toBeVisible();
  await expect(page.getByText("Physician required")).toBeVisible();
  const conf = Number(await page.getByRole("meter", { name: "draft confidence" }).getAttribute("aria-valuenow"));
  expect(conf).toBeLessThanOrEqual(50);
});

test("playground: an injection cannot flip a failing case into an approval", async ({ page }) => {
  await signIn(page, "playground");
  await runScenario(page, "fail", { injection: true });
  await expect(page.getByText("prompt_injection").first()).toBeVisible();
  await expect(page.getByText("Draft: does not support").first()).toBeVisible();
  await expect(page.getByText("Draft: supports approval")).toHaveCount(0);
});

test("playground role can read a draft but has no way to decide", async ({ page }) => {
  await signIn(page, "playground");
  await page.goto("/queue");
  await page.getByRole("link", { name: /^REQ-/ }).first().click();
  await expect(page.getByText("Your role is read-only here")).toBeVisible();
  await expect(page.getByRole("button", { name: "Claim review" })).toHaveCount(0);
});

test("reviewer workflow: RN cannot take a physician task; physician must justify a denial; decision is final and audited", async ({ page }) => {
  await signIn(page, "playground");
  const rid = await runScenario(page, "fail");

  await signIn(page, "reviewer-rn", "rn-alice");
  await page.goto(`/review/TASK-${rid}`);
  await expect(page.getByText(/requires a physician reviewer/)).toBeVisible();
  await expect(page.getByRole("button", { name: "Claim review" })).toBeDisabled();

  await signIn(page, "reviewer-md", "dr-carol");
  await page.goto("/queue");
  await expect(page.getByRole("link", { name: rid })).toBeVisible();
  await page.goto(`/review/TASK-${rid}`);
  await page.getByRole("button", { name: "Claim review" }).click();
  await expect(page.getByText("Your decision", { exact: true })).toBeVisible();
  // no option preselected: the draft must not anchor the reviewer
  for (const r of await page.getByRole("radio").all()) await expect(r).not.toBeChecked();
  await expect(page.getByRole("button", { name: "Review decision" })).toBeDisabled();
  await page.getByLabel(/^Deny/).check();
  await expect(page.getByRole("button", { name: "Review decision" })).toBeDisabled();   // denial needs a written reason
  await page.getByLabel(/Reason/).fill("Required criterion contradicted by the lab record.");
  await page.getByRole("button", { name: "Review decision" }).click();
  await page.getByRole("button", { name: "Confirm final decision" }).click();
  await expect(page.getByText("Final decision (human)")).toBeVisible();
  await expect(page.getByText("dr-carol", { exact: true })).toBeVisible();
  await expect(page.getByText("Reviewer agreed with the AI draft")).toBeVisible();

  await signIn(page, "auditor");
  await page.goto("/audit");
  await page.getByPlaceholder("REQ-…").fill(rid);
  await page.getByRole("button", { name: "Filter" }).click();
  await expect(page.getByRole("cell", { name: "review.decided" })).toBeVisible();
  await page.getByRole("button", { name: "Verify hash chain" }).click();
  await expect(page.getByText("✓ Chain intact")).toBeVisible();
});

test("an RN who overrides a draft must give a reason", async ({ page }) => {
  await signIn(page, "playground");
  const rid = await runScenario(page, "meets", { paraphrased: false });
  await signIn(page, "reviewer-rn", "rn-bob");
  await page.goto(`/review/TASK-${rid}`);
  await page.getByRole("button", { name: "Claim review" }).click();
  await page.getByLabel(/^Pend for information/).check();      // draft supports approval; pending = disagreement
  await expect(page.getByText(/you are overriding the draft/)).toBeVisible();
  await expect(page.getByRole("button", { name: "Review decision" })).toBeDisabled();
  await expect(page.getByLabel(/^Deny/)).toBeDisabled();       // RN cannot deny
});

test.describe("proxy security", () => {
  test("the token is httpOnly and unreachable from page scripts", async ({ page }) => {
    await signIn(page, "reviewer-rn");
    expect(await page.evaluate(() => document.cookie)).not.toContain("pa_token");
    const cookies = await page.context().cookies();
    const c = cookies.find((x) => x.name === "pa_token")!;
    expect(c.httpOnly).toBe(true); expect(c.sameSite).toBe("Strict");
  });

  test("API proxy needs a session, is origin-checked and only reaches allow-listed areas", async ({ request, page }) => {
    expect((await request.get("/api/pa/me")).status()).toBe(401);
    await signIn(page, "submitter");
    const ctx = page.context().request;
    expect((await ctx.get("/api/pa/me")).status()).toBe(200);
    expect((await ctx.get("/api/pa/healthz")).status()).toBe(404);                          // not on the allow-list
    expect((await ctx.get("/api/pa/..%2Fmetrics")).status()).toBe(404);
    const evil = await ctx.post("/api/pa/requests", { headers: { origin: "https://evil.example", "idempotency-key": "x" }, data: {} });
    expect(evil.status()).toBe(403);                                                         // cross-origin state change refused
    const noOrigin = await ctx.post("/api/pa/requests", { headers: { "idempotency-key": "x" }, data: {} });
    expect(noOrigin.status()).toBe(403);
  });

  test("security headers are set", async ({ request }) => {
    const h = (await request.get("/login")).headers();
    expect(h["content-security-policy"]).toContain("frame-ancestors 'none'");
    expect(h["x-content-type-options"]).toBe("nosniff");
    expect(h["cache-control"]).toContain("no-store");
  });

  test("an expired token is rejected at sign-in", async ({ page }) => {
    await page.goto("/login");
    await page.getByLabel("API token").fill(token("auditor", "x", -60));
    await page.getByRole("button", { name: "Sign in" }).click();
    await expect(page.locator("[role=alert]:not(#__next-route-announcer__)")).toContainText("rejected");
  });
});

test.describe("accessibility (axe, WCAG 2.x A/AA)", () => {
  for (const [role, path] of [["reviewer-md", "/queue"], ["playground", "/playground"], ["auditor", "/audit"]] as const) {
    test(`${path} has no detectable violations`, async ({ page }) => {
      await signIn(page, role);
      await page.goto(path);
      await page.waitForLoadState("networkidle");
      const r = await new AxeBuilder({ page }).withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"]).analyze();
      expect(r.violations.map((v) => `${v.id}: ${v.nodes.length} node(s)`)).toEqual([]);
    });
  }
});
