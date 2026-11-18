// SPDX-License-Identifier: AGPL-3.0-only
// Shared helpers for the image demos (camera.html, photos.html, firehose.html): the stage fit, one /v1/systemone call with
// images, a rolling rate meter, and the sample photo library (photos/manifest.json, fetched by photos/fetch.py).
const Vision = (() => {
  const MODEL = "sparkscore-d3";

  function fitStage() {
    const fit = () => document.documentElement.style.setProperty("--s", Math.min(innerWidth / 1920, innerHeight / 1080));
    addEventListener("resize", fit); fit();
  }

  // A browser holds six HTTP/1.1 connections to a host, so a page with more calls in flight would run six at
  // a time (and the server would batch six). On loopback the calls take turns over localhost and two
  // *.localhost names, which the browser resolves to loopback itself; serve.py answers them with CORS.
  const LOOPBACK = /^(localhost|127\.0\.0\.1|\[::1\])$/.test(location.hostname);
  const HOSTS = LOOPBACK ? ["", `http://s1.localhost:${location.port}`, `http://s2.localhost:${location.port}`] : [""];
  let turn = 0;

  // -> {answers, backendMs, totalMs, pack (the image pack: "built" when read now)}; throws Error (with .status) with the server's detail on a refusal. Busy (529: every
  // slot taken, as when a new sort starts while the last one's calls still run) is retried: a slot frees
  // within one engine pass.
  async function ask(body) {
    const t0 = performance.now();
    const json = JSON.stringify({ model: MODEL, ...body });
    let r;
    for (let tries = 0; ; tries++) {
      const host = HOSTS[turn++ % HOSTS.length];
      r = await fetch(`${host}/v1/systemone`, { method: "POST", headers: { "Content-Type": "application/json" }, body: json });
      if (r.status !== 529 || tries >= 50) break;
      await sleep(200);
    }
    const data = await r.json().catch(() => ({}));
    if (!r.ok) {
      const error = new Error(typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail || r.status));
      error.status = r.status;
      throw error;
    }
    // Server-Timing: "backend;dur=12.3, pack;desc=built" -> {backend: {dur: "12.3"}, pack: {desc: "built"}}.
    const timing = Object.fromEntries((r.headers.get("Server-Timing") || "").split(",").map(entry => {
      const [name, ...fields] = entry.trim().split(";");
      return [name, Object.fromEntries(fields.map(f => f.split("=")))];
    }));
    return { answers: data.answers, backendMs: parseFloat(timing.backend?.dur) || 0, totalMs: performance.now() - t0,
             pack: timing.pack?.desc ?? null };
  }

  // Events per second and mean engine ms over the last few seconds.
  class Rate {
    constructor(windowMs = 4000) { this.windowMs = windowMs; this.events = []; }
    tick(ms) { const now = performance.now(); this.events.push([now, ms]); this.trim(now); }
    trim(now) { while (this.events.length && now - this.events[0][0] > this.windowMs) this.events.shift(); }
    perSecond() { this.trim(performance.now()); return this.events.length < 2 ? 0 : (this.events.length - 1) / ((this.events.at(-1)[0] - this.events[0][0]) / 1000); }
    ms() { return this.events.length ? this.events.reduce((a, e) => a + e[1], 0) / this.events.length : 0; }
  }

  const sleep = ms => new Promise(r => setTimeout(r, ms));
  const escape = s => s.replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  async function library() {
    const r = await fetch("photos/manifest.json");
    if (!r.ok) throw new Error("no photo library: run python3 demo/photos/fetch.py");
    return (await r.json()).map(p => ({ ...p, url: `photos/${p.file}` }));
  }

  // A data URL for an image URL or File, downscaled so its long side is at most `maxSide` (JPEG) when larger.
  async function dataURL(src, maxSide = 1280) {
    const blob = src instanceof Blob ? src : await (await fetch(src)).blob();
    const bitmap = await createImageBitmap(blob);
    const scale = Math.min(1, maxSide / Math.max(bitmap.width, bitmap.height));
    if (scale === 1 && /^image\/(jpeg|png|webp)$/.test(blob.type)) {
      return await new Promise(res => { const fr = new FileReader(); fr.onload = () => res(fr.result); fr.readAsDataURL(blob); });
    }
    const c = document.createElement("canvas");
    c.width = Math.round(bitmap.width * scale); c.height = Math.round(bitmap.height * scale);
    c.getContext("2d").drawImage(bitmap, 0, 0, c.width, c.height);
    return c.toDataURL("image/jpeg", 0.9);
  }

  return { MODEL, fitStage, ask, Rate, sleep, escape, library, dataURL };
})();
