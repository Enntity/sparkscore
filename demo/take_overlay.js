// SPDX-License-Identifier: AGPL-3.0-only
// Video overlays for demo/record.py takes, drawn over a page's #stage while it is recorded (no page loads this
// itself): a slow-motion title, a full-speed transition with a live badge, and narrative captions. The take
// drives them through window.__ov.
(() => {
  const css = `
#ov { position: absolute; inset: 0; pointer-events: none; z-index: 20; }
#ov .tint { position: absolute; inset: 0; opacity: 0; transition: opacity 0.8s;
  background: repeating-linear-gradient(0deg, rgba(255, 255, 255, 0.025) 0 2px, transparent 2px 5px),
              radial-gradient(ellipse at 50% 50%, transparent 45%, rgba(40, 24, 0, 0.55) 100%); }
#ov .tint.on { opacity: 1; }

#ov .slow { position: absolute; left: 832px; top: 250px; width: 1040px; height: 662px; display: flex; flex-direction: column;
  align-items: center; justify-content: center; gap: 10px; opacity: 0; transform: scale(0.94); transition: opacity 0.6s, transform 0.6s; }
#ov .slow.on { opacity: 1; transform: none; }
#ov .slow .icon { font: 700 64px var(--head); color: #ffc23d; letter-spacing: -8px; animation: ov-pulse 1.6s ease-in-out infinite;
  text-shadow: 0 0 30px rgba(255, 194, 61, 0.8); }
#ov .slow .title { font: 700 112px var(--head); letter-spacing: 8px; color: transparent;
  -webkit-text-stroke: 2px rgba(255, 194, 61, 0.95); filter: drop-shadow(0 0 22px rgba(255, 194, 61, 0.45)); }
#ov .slow .sub { font: 500 26px var(--mono); color: rgba(255, 194, 61, 0.8); letter-spacing: 4px; text-transform: uppercase; }
@keyframes ov-pulse { 50% { opacity: 0.35; } }

#ov .fast { position: absolute; inset: 0; opacity: 0; overflow: hidden; }
#ov .fast.go { animation: ov-dim 1.9s ease-out forwards; }
@keyframes ov-dim { 0% { opacity: 0; background: rgba(3, 4, 10, 0); } 10% { opacity: 1; background: rgba(3, 4, 10, 0.6); }
  70% { opacity: 1; background: rgba(3, 4, 10, 0.45); } 100% { opacity: 0; background: rgba(3, 4, 10, 0); } }
#ov .streak { position: absolute; left: 0; height: 3px; border-radius: 3px;
  background: linear-gradient(90deg, transparent, rgba(62, 232, 255, 0.9), #fff); filter: drop-shadow(0 0 8px rgba(62, 232, 255, 0.9)); }
#ov .fast.go .streak { animation: ov-streak var(--d) linear var(--delay) infinite; }
@keyframes ov-streak { from { transform: translateX(-100%); } to { transform: translateX(2400px); } }
#ov .fast .title { position: absolute; left: 0; right: 0; top: 380px; text-align: center; font: italic 700 176px var(--head);
  letter-spacing: 2px; color: #fff; text-shadow: 0 0 40px rgba(62, 232, 255, 0.9), 0 0 120px rgba(62, 232, 255, 0.5); }
#ov .fast .title span { color: var(--cyan); }
#ov .fast .sub { position: absolute; left: 0; right: 0; top: 590px; text-align: center; font: 500 30px var(--mono);
  color: var(--cyan); letter-spacing: 6px; text-transform: uppercase; }
#ov .fast.go .title { animation: ov-slam 1.9s cubic-bezier(.2, .8, .2, 1) forwards; }
#ov .fast.go .sub { animation: ov-slam 1.9s cubic-bezier(.2, .8, .2, 1) 0.08s backwards; }
@keyframes ov-slam { 0% { transform: scale(1.7) skewX(-14deg); opacity: 0; filter: blur(12px); }
  14% { transform: scale(1) skewX(-14deg); opacity: 1; filter: none; } 72% { transform: translateX(40px) skewX(-14deg); opacity: 1; filter: none; }
  100% { transform: translateX(900px) skewX(-14deg); opacity: 0; filter: blur(14px); } }

#ov .live { position: absolute; right: 70px; top: 266px; display: flex; align-items: center; gap: 12px;
  font: 700 19px var(--mono); letter-spacing: 2px; color: var(--green); padding: 9px 18px; border-radius: 999px;
  background: rgba(3, 10, 8, 0.85); border: 1px solid rgba(61, 255, 168, 0.45); opacity: 0; transform: translateY(-8px);
  transition: opacity 0.4s, transform 0.4s; }
#ov .live.on { opacity: 1; transform: none; }
#ov .live i { width: 12px; height: 12px; border-radius: 50%; background: var(--green); box-shadow: 0 0 12px var(--green);
  animation: ov-pulse 1s ease-in-out infinite; }

#ov .cap { position: absolute; left: 50%; bottom: 152px; max-width: 1300px; padding: 18px 32px 20px; border-radius: 18px;
  background: rgba(5, 8, 20, 0.9); border: 1px solid rgba(62, 232, 255, 0.35); box-shadow: 0 18px 50px rgba(0, 0, 0, 0.6);
  opacity: 0; transform: translate(-50%, 24px); transition: opacity 0.45s, transform 0.45s cubic-bezier(.2, .8, .2, 1); }
#ov .cap.on { opacity: 1; transform: translate(-50%, 0); }
#ov .cap .k { font: 600 16px var(--mono); color: var(--cyan); letter-spacing: 4px; text-transform: uppercase; margin-bottom: 6px; }
#ov .cap.slow-k .k { color: #ffc23d; }
#ov .cap .t { font: 600 34px var(--head); color: var(--text); white-space: nowrap; }
`;
  const style = document.createElement("style");
  style.textContent = css;
  document.head.appendChild(style);

  const ov = document.createElement("div");
  ov.id = "ov";
  ov.innerHTML = `
    <div class="tint"></div>
    <div class="slow"><div class="icon">◀◀</div><div class="title">SLOW MOTION</div><div class="sub">one photo at a time</div></div>
    <div class="fast"><div class="title">FULL SPEED <span>▶▶</span></div><div class="sub">live · real time · nothing sped up</div></div>
    <div class="live"><i></i>FULL SPEED · REAL TIME</div>
    <div class="cap"><div class="k"></div><div class="t"></div></div>`;
  document.getElementById("stage").appendChild(ov);
  const q = s => ov.querySelector(s);
  const fast = q(".fast");
  for (let k = 0; k < 34; k++) {
    const s = document.createElement("div");
    s.className = "streak";
    s.style.top = `${Math.round(30 + Math.random() * 1020)}px`;
    s.style.width = `${Math.round(180 + Math.random() * 520)}px`;
    s.style.setProperty("--d", `${(0.35 + Math.random() * 0.45).toFixed(2)}s`);
    s.style.setProperty("--delay", `${(-Math.random()).toFixed(2)}s`);
    fast.appendChild(s);
  }

  let capTimer = 0;
  window.__ov = {
    // A caption (kicker, text); a showing one fades out first. `slow` colours the kicker amber.
    caption(kicker, text, slow = false) {
      const cap = q(".cap");
      clearTimeout(capTimer);
      const show = () => {
        cap.querySelector(".k").textContent = kicker;
        cap.querySelector(".t").textContent = text;
        cap.classList.toggle("slow-k", slow);
        cap.classList.add("on");
      };
      if (cap.classList.contains("on")) { cap.classList.remove("on"); capTimer = setTimeout(show, 420); } else show();
    },
    hide() { clearTimeout(capTimer); q(".cap").classList.remove("on"); },
    slow(on) { q(".slow").classList.toggle("on", on); q(".tint").classList.toggle("on", on); },
    // The full-speed transition (1.9 s), then the live badge.
    fast() {
      this.slow(false);
      fast.classList.remove("go"); void fast.offsetWidth; fast.classList.add("go");
      setTimeout(() => q(".live").classList.add("on"), 1500);
    },
    live(on) { q(".live").classList.toggle("on", on); },
  };
})();
