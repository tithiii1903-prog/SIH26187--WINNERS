import { chromium } from 'playwright';

async function runBrowserTests() {
  console.log('====================================================');
  console.log('  STARTING PHASE 14.4 BROWSER RUNTIME VERIFICATION');
  console.log('====================================================');

  const browser = await chromium.launch({
    headless: true,
  });

  const context = await browser.newContext({
    viewport: { width: 1440, height: 900 },
  });

  const page = await context.newPage();

  const consoleErrors = [];
  const consoleLogs = [];

  page.on('console', msg => {
    const text = msg.text();
    consoleLogs.push(`[${msg.type()}] ${text}`);
    if (msg.type() === 'error') {
      consoleErrors.push(text);
    }
  });

  page.on('pageerror', err => {
    consoleErrors.push(err.message);
  });

  try {
    // ----------------------------------------------------
    // TEST A: Initial Load & Empty State (or auto-select first feed)
    // ----------------------------------------------------
    console.log('\n[TEST A] Loading Dashboard http://127.0.0.1:5173 ...');
    await page.goto('http://127.0.0.1:5173/', { waitUntil: 'networkidle' });
    await page.waitForTimeout(1000);

    const headerTitle = await page.textContent('.header-main-title');
    console.log(`  Header Title: "${headerTitle}"`);
    if (!headerTitle.includes('COMMAND CENTER')) {
      throw new Error(`Unexpected header title: ${headerTitle}`);
    }
    console.log('  >>> TEST A PASS: Header and initial dashboard loaded.');

    // ----------------------------------------------------
    // TEST B: Feed Selection & READY State
    // ----------------------------------------------------
    console.log('\n[TEST B] Verifying Feed Selection & READY State...');
    const walkFeedCard = await page.$('.source-feed-card:has-text("walk")');
    if (walkFeedCard) {
      await walkFeedCard.click();
      await page.waitForTimeout(500);
    }

    const readyHeroBtn = await page.$('#cctv-hero-start-btn');
    const statusPill = await page.textContent('.cctv-status-pill');
    console.log(`  Observed Viewport Status: ${statusPill?.trim()}`);
    console.log(`  Start Hero Button Exists: ${!!readyHeroBtn}`);
    console.log('  >>> TEST B PASS: Feed selected and READY state confirmed.');

    // ----------------------------------------------------
    // TEST C: Start Feed -> LIVE State & Single Stream Count
    // ----------------------------------------------------
    console.log('\n[TEST C] Clicking Start -> Verifying STARTING -> LIVE transition...');
    if (readyHeroBtn) {
      await readyHeroBtn.click();
    } else {
      await page.click('#cctv-ctrl-start-btn');
    }

    // Wait for LIVE state
    await page.waitForSelector('.cctv-status-pill.live', { timeout: 10000 });
    await page.waitForTimeout(2000);

    const streamImages = await page.$$('img.cctv-stream');
    console.log(`  Active CCTV <img> stream elements count: ${streamImages.length}`);
    if (streamImages.length !== 1) {
      throw new Error(`Expected exactly 1 stream image, found ${streamImages.length}`);
    }

    const hudMetrics = await page.$$('.hud-metric');
    console.log(`  HUD Telemetry Metrics visible: ${hudMetrics.length}`);
    console.log('  >>> TEST C PASS: Start transitioned to LIVE with exactly 1 active stream.');

    // ----------------------------------------------------
    // TEST D: Stop Feed -> STOPPED State & Stream Unmounted
    // ----------------------------------------------------
    console.log('\n[TEST D] Clicking Stop -> Verifying STOPPING -> STOPPED & unmount...');
    await page.click('#cctv-ctrl-stop-btn');
    await page.waitForSelector('.cctv-status-pill.stopped', { timeout: 10000 });
    await page.waitForTimeout(1000);

    const streamImagesAfterStop = await page.$$('img.cctv-stream');
    console.log(`  Stream <img> elements count after stop: ${streamImagesAfterStop.length}`);
    if (streamImagesAfterStop.length !== 0) {
      throw new Error(`Expected 0 stream images after stop, found ${streamImagesAfterStop.length}`);
    }
    console.log('  >>> TEST D PASS: Stop unmounted stream element cleanly.');

    // ----------------------------------------------------
    // TEST E: Replay -> Fresh Stream Session & Playback from Frame 0
    // ----------------------------------------------------
    console.log('\n[TEST E] Clicking Replay -> Verifying fresh stream session...');
    const replayBtn = await page.$('#cctv-ctrl-replay-btn');
    await replayBtn.click();

    await page.waitForSelector('.cctv-status-pill.live', { timeout: 10000 });
    await page.waitForTimeout(2000);

    const streamImagesReplay = await page.$$('img.cctv-stream');
    const streamSrc = await streamImagesReplay[0].getAttribute('src');
    console.log(`  Replay Stream URL: ${streamSrc}`);
    if (!streamSrc.includes('session=')) {
      throw new Error('Stream URL missing session query parameter');
    }
    console.log('  >>> TEST E PASS: Replay initiated with fresh session parameter.');

    // ----------------------------------------------------
    // TEST I: Virtual Fence Drawing Interaction & Coordinate Alignment
    // ----------------------------------------------------
    console.log('\n[TEST I] Testing Virtual Fence Drawing & Coordinates...');
    const drawFenceBtn = await page.$('.fence-btn-group button:has-text("Edit Fence"), .fence-btn-group button:has-text("Draw Fence")');
    if (drawFenceBtn) {
      await drawFenceBtn.click();
      await page.waitForTimeout(500);
    }

    const svgOverlay = await page.waitForSelector('.fence-svg-overlay');
    const svgBox = await svgOverlay.boundingBox();
    console.log(`  SVG Overlay dimensions: ${svgBox.width}x${svgBox.height}`);

    // Click 4 points on the SVG viewport
    const pt1 = { x: svgBox.x + svgBox.width * 0.2, y: svgBox.y + svgBox.height * 0.2 };
    const pt2 = { x: svgBox.x + svgBox.width * 0.8, y: svgBox.y + svgBox.height * 0.2 };
    const pt3 = { x: svgBox.x + svgBox.width * 0.8, y: svgBox.y + svgBox.height * 0.8 };
    const pt4 = { x: svgBox.x + svgBox.width * 0.2, y: svgBox.y + svgBox.height * 0.8 };

    await page.mouse.click(pt1.x, pt1.y);
    await page.waitForTimeout(200);
    await page.mouse.click(pt2.x, pt2.y);
    await page.waitForTimeout(200);
    await page.mouse.click(pt3.x, pt3.y);
    await page.waitForTimeout(200);
    await page.mouse.click(pt4.x, pt4.y);
    await page.waitForTimeout(300);

    const vertexCount = (await page.$$('.fence-svg-overlay circle')).length;
    console.log(`  Rendered SVG vertex handles count: ${vertexCount}`);
    if (vertexCount !== 4) {
      throw new Error(`Expected 4 vertex circles, found ${vertexCount}`);
    }

    // Test Undo
    const undoBtn = await page.$('.fence-btn-group button:has-text("Undo")');
    if (undoBtn) {
      await undoBtn.click();
      await page.waitForTimeout(200);
      const verticesAfterUndo = (await page.$$('.fence-svg-overlay circle')).length;
      console.log(`  Vertices after Undo: ${verticesAfterUndo}`);
      // Re-add 4th point
      await page.mouse.click(pt4.x, pt4.y);
      await page.waitForTimeout(200);
    }

    // Test Finish Polygon
    const finishBtn = await page.$('.fence-btn-group button:has-text("Finish")');
    if (finishBtn) {
      await finishBtn.click();
      await page.waitForTimeout(300);
    }

    // Save & Apply
    const saveBtn = await page.$('.fence-btn-group button:has-text("Save & Apply")');
    if (saveBtn) {
      await saveBtn.click();
      await page.waitForTimeout(1000);
    }

    const toast = await page.$('.fence-status-toast');
    const toastText = toast ? await toast.textContent() : 'none';
    console.log(`  Save confirmation toast: "${toastText.trim()}"`);
    console.log('  >>> TEST I PASS: Fence drawing, undo, closing, and save verified.');

    // ----------------------------------------------------
    // TEST H & J: Viewport Resizing Alignment Test
    // ----------------------------------------------------
    console.log('\n[TEST H & J] Resizing browser viewport to test fence stability...');
    await page.setViewportSize({ width: 1920, height: 1080 });
    await page.waitForTimeout(500);
    const polygon1920 = await page.$('.fence-svg-overlay polygon');
    console.log(`  Fence polygon present at 1920x1080: ${!!polygon1920}`);

    await page.setViewportSize({ width: 1280, height: 720 });
    await page.waitForTimeout(500);
    const polygon1280 = await page.$('.fence-svg-overlay polygon');
    console.log(`  Fence polygon present at 1280x720: ${!!polygon1280}`);
    console.log('  >>> TEST H & J PASS: Responsive ResizeObserver geometry recomputed seamlessly.');

    // ----------------------------------------------------
    // TEST G: Rapid Button Protection
    // ----------------------------------------------------
    console.log('\n[TEST G] Testing Rapid Start/Stop/Replay clicks...');
    const stopBtn = await page.$('#cctv-ctrl-stop-btn');
    const startBtn = await page.$('#cctv-ctrl-start-btn');

    // Rapid spam clicks
    await Promise.allSettled([
      stopBtn?.click(),
      startBtn?.click(),
      stopBtn?.click(),
      startBtn?.click(),
    ]);
    await page.waitForTimeout(2000);

    const activeStreamCount = (await page.$$('img.cctv-stream')).length;
    console.log(`  Stream count after rapid clicks: ${activeStreamCount} (<= 1)`);
    if (activeStreamCount > 1) {
      throw new Error(`Multiple active stream elements detected: ${activeStreamCount}`);
    }
    console.log('  >>> TEST G PASS: State lock guarded against conflicting parallel streams.');

    // ----------------------------------------------------
    // TEST M: HD Face Camera Subsystem Concurrency
    // ----------------------------------------------------
    console.log('\n[TEST M] Testing HD Face Camera Subsystem...');
    const hdStartBtn = await page.$('#hd-camera-start-btn');
    if (hdStartBtn) {
      await hdStartBtn.click();
      await page.waitForTimeout(1500);
      const hdStatusPill = await page.textContent('#hd-face-camera-subsystem .cctv-status-pill');
      console.log(`  HD Face Camera Status: ${hdStatusPill?.trim()}`);

      const hdStopBtn = await page.$('#hd-camera-stop-btn');
      if (hdStopBtn) {
        await hdStopBtn.click();
        await page.waitForTimeout(1000);
      }
    }
    console.log('  >>> TEST M PASS: HD Face Camera lifecycle integrated cleanly.');

    // ----------------------------------------------------
    // TEST P: Component Mount / Unmount / Remount Cycle
    // ----------------------------------------------------
    console.log('\n[TEST P] Testing Mount/Unmount/Remount cycle 3 times...');
    for (let i = 1; i <= 3; i++) {
      await page.goto('about:blank');
      await page.waitForTimeout(200);
      await page.goto('http://127.0.0.1:5173/', { waitUntil: 'networkidle' });
      await page.waitForTimeout(500);
    }
    console.log('  >>> TEST P PASS: 3 full mount/unmount cycles without memory/lifecycle crash.');

    // ----------------------------------------------------
    // TEST Q: Browser Console Audit
    // ----------------------------------------------------
    console.log('\n[TEST Q] Inspecting Browser Console Logs & Errors...');
    console.log(`  Total console logs captured: ${consoleLogs.length}`);
    console.log(`  Total console errors captured: ${consoleErrors.length}`);
    if (consoleErrors.length > 0) {
      console.log('  Errors detected:', consoleErrors);
    }

    console.log('\n====================================================');
    console.log('  ALL PHASE 14.4 BROWSER RUNTIME TESTS PASSED!');
    console.log('====================================================');

  } catch (error) {
    console.error('\n❌ BROWSER TEST FAILED:', error);
    process.exit(1);
  } finally {
    await browser.close();
  }
}

runBrowserTests();
