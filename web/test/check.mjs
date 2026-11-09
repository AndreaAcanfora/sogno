// Checks the WebGPU dreamer against MLX: serves web/, opens test/golden.html in Chromium and reads the
// verdict. Usage (from the repo root):
//
//   uv run sogno-export --model runs/sabbia --out web/model --golden web/test/golden.bin
//   node web/test/check.mjs
import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { extname, join, normalize } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";

const root = fileURLToPath(new URL("..", import.meta.url));
const types = { ".html": "text/html", ".css": "text/css", ".svg": "image/svg+xml", ".js": "text/javascript", ".json": "application/json", ".wasm": "application/wasm" };
const server = createServer(async (req, res) => {
  const path = normalize(join(root, decodeURIComponent(new URL(req.url, "http://x").pathname)));
  if (!path.startsWith(root)) return res.writeHead(403).end();
  try {
    const body = await readFile(path);
    res.writeHead(200, { "content-type": types[extname(path)] ?? "application/octet-stream" }).end(body);
  } catch {
    res.writeHead(404).end();
  }
}).listen(0);
const port = server.address().port;

// SOGNO_CHANNEL=chrome uses the installed Chrome instead of Playwright's Chromium.
const browser = await chromium.launch({
  channel: process.env.SOGNO_CHANNEL || undefined,
  args: ["--enable-unsafe-webgpu", "--enable-features=Vulkan", "--use-angle=metal", ...(process.env.SOGNO_GPU_ARGS ?? "").split(" ").filter(Boolean)],
});
let failed = true;
try {
  const page = await browser.newPage();
  page.on("console", (m) => console.log(`[page] ${m.text()}`));
  await page.goto(`http://localhost:${port}/test/golden.html?golden=golden.bin`);
  await page.waitForFunction(() => window.result, null, { timeout: 120_000 });
  const result = await page.evaluate(() => window.result);
  console.log(JSON.stringify(result, null, 2));
  if (result.error) throw new Error(result.error);
  // float16 weights on both sides; what is left is float32 summation order.
  const tolerance = 1e-3 * Math.max(1, result.scale);
  failed = result.worst > tolerance || result.argmaxAgreement < 0.999 || result.errors.length > 0;
  console.log(failed ? `FAIL: max |diff| ${result.worst} > ${tolerance}` : `ok: max |diff| ${result.worst.toExponential(2)}`);

  // The demo itself: it loads, dreams, and a brush stroke reaches both worlds.
  const demo = await browser.newPage({ viewport: { width: 1280, height: 640 } });
  const problems = [];
  demo.on("pageerror", (e) => problems.push(String(e)));
  demo.on("console", (m) => m.type() === "error" && !m.text().includes("404") && problems.push(m.text()));
  await demo.goto(`http://localhost:${port}/index.html?seed=7`);
  await demo.waitForFunction(() => window.sogno, null, { timeout: 60_000 });
  const box = await demo.locator("#real").boundingBox();
  await demo.mouse.move(box.x + box.width * 0.3, box.y + box.height * 0.1);
  await demo.mouse.down();
  await demo.mouse.move(box.x + box.width * 0.7, box.y + box.height * 0.1, { steps: 20 });
  await demo.waitForTimeout(500);
  await demo.mouse.up();
  await demo.waitForTimeout(3000);
  const state = await demo.evaluate(() => {
    const count = (f, id) => f.reduce((n, e) => n + (e === id), 0);
    return { stats: document.getElementById("stats").textContent, sandReal: count(window.sogno.real, 2), sandDream: count(window.sogno.dream, 2) };
  });
  console.log(JSON.stringify(state));
  if (process.env.SOGNO_SCREENSHOT) await demo.screenshot({ path: process.env.SOGNO_SCREENSHOT });
  if (problems.length || !state.sandReal || !state.sandDream) {
    console.log(`FAIL: demo ${JSON.stringify(problems)}`);
    failed = true;
  }
} finally {
  await browser.close();
  server.close();
}
process.exit(failed ? 1 : 0);
