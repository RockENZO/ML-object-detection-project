// Records a scripted walkthrough of the static viewer as WebM (see encode_viewer.sh for MP4/GIF).
//   node tools/demo/record_viewer.mjs [outputDir]
// Needs Playwright with a Chromium build (set PLAYWRIGHT_PATH if it is installed globally).
import { createRequire } from 'node:module';
import { createServer } from 'node:http';
import { readFile, stat, mkdir } from 'node:fs/promises';
import { createReadStream } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const require = createRequire(import.meta.url);
const { chromium } = require(process.env.PLAYWRIGHT_PATH || 'playwright');
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../web');
const out = path.resolve(process.argv[2] || 'demo-out');
await mkdir(out, { recursive: true });

const types = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.json': 'application/json', '.mp4': 'video/mp4', '.webm': 'video/webm' };
// Small static server with byte ranges: browsers need them to seek video.
const server = createServer(async (req, res) => {
  const file = path.join(root, decodeURIComponent(new URL(req.url, 'http://x').pathname).replace(/^\/$/, '/index.html'));
  if (!file.startsWith(root)) return res.writeHead(403).end();
  try {
    const { size } = await stat(file);
    const type = types[path.extname(file)] || 'application/octet-stream';
    const range = /bytes=(\d*)-(\d*)/.exec(req.headers.range || '');
    if (range) {
      const start = range[1] ? Number(range[1]) : 0;
      const end = range[2] ? Number(range[2]) : size - 1;
      res.writeHead(206, { 'Content-Type': type, 'Content-Range': `bytes ${start}-${end}/${size}`, 'Accept-Ranges': 'bytes', 'Content-Length': end - start + 1 });
      createReadStream(file, { start, end }).pipe(res);
    } else {
      res.writeHead(200, { 'Content-Type': type, 'Accept-Ranges': 'bytes', 'Content-Length': size });
      createReadStream(file).pipe(res);
    }
  } catch {
    res.writeHead(404).end();
  }
});
await new Promise((resolve) => server.listen(0, resolve));
const url = `http://localhost:${server.address().port}/index.html`;

const size = { width: 1360, height: 1560 };
const browser = await chromium.launch({ args: ['--autoplay-policy=no-user-gesture-required'] });
const context = await browser.newContext({ viewport: size, recordVideo: { dir: out, size } });
const page = await context.newPage();
page.on('pageerror', (error) => { throw error; });
await page.goto(url);
await page.waitForFunction(() => window.__viewer && document.getElementById('video').readyState >= 3);
const duration = await page.evaluate(() => window.__viewer.duration);

await page.waitForTimeout(1200);
await page.evaluate(() => document.getElementById('video').play());
const clock = () => page.evaluate(() => document.getElementById('video').currentTime);
const until = async (t) => { while ((await clock()) < t) await page.waitForTimeout(150); };

// Let play run, switch on space control, follow a player, jump between moments, then show a heatmap.
// Moments are reached with the "]" shortcut so the page does not scroll while recording.
await until(Math.min(3, duration * 0.1));
await page.check('#tSpace');
await until(Math.min(7, duration * 0.25));
if (await page.locator('#players tbody tr').count()) {
  await page.locator('#players tbody tr').nth(1).click({ force: true });
  await page.check('#tLabels');
  await page.waitForTimeout(3500);
  await page.locator('#clearSel').click({ force: true });
  await page.uncheck('#tLabels');
}
// The viewer ignores shortcuts while a form control has focus.
await page.evaluate(() => document.activeElement && document.activeElement.blur());
await page.keyboard.press(']');
await page.waitForTimeout(3200);
await page.keyboard.press(']');
await page.waitForTimeout(3200);
await page.uncheck('#tSpace');
await page.selectOption('#tHeat', 'A');
await page.waitForTimeout(2500);
await page.selectOption('#tHeat', '');
await page.evaluate(() => window.__viewer.seek(0));
await page.waitForTimeout(500);
await page.evaluate(() => document.getElementById('video').play());
await until(duration - 0.5);
await page.waitForTimeout(800);

const video = page.video();
await context.close();
await browser.close();
server.close();
console.log(await video.path());
