// The home-page story: one real generated incident, from the outage to the
// postmortem, played from story.json (scripts/make_story.py). No key, no cost.
(() => {
  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g,
    (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const still = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const t = (s) => Date.parse(s);
  const hm = (ms) => new Date(ms).toISOString().slice(11, 16);
  const hms = (ms) => new Date(ms).toISOString().slice(11, 19);
  const day = (ms) => new Date(ms).toLocaleDateString("en-GB", { weekday: "long", timeZone: "UTC" });
  const tag = (id) => `<a class="tag" href="#p-${esc(id)}">${esc(id)}</a>`;
  const once = (el, fn) => {
    if (still || !("IntersectionObserver" in window)) return fn();
    const io = new IntersectionObserver(([x]) => { if (x.isIntersecting) { io.disconnect(); fn(); } }, { threshold: 0.3 });
    io.observe(el);
  };
  let S;

  // In-page links scroll instead of changing the hash, which the app uses for routing.
  $("story").addEventListener("click", (e) => {
    const a = e.target.closest('a[href^="#"]');
    const target = a && !a.getAttribute("href").startsWith("#/") && document.querySelector(a.getAttribute("href"));
    if (!target) return;
    e.preventDefault();
    target.scrollIntoView({ behavior: still ? "auto" : "smooth", block: "start" });
  });

  fetch("/static/copilot/story.json")
    .then((r) => (r.ok ? r.json() : Promise.reject(r.status)))
    .then((data) => { S = data; wall(); game(); proof(); morning(); clock(); })
    .catch(() => { $("story").hidden = true; });

  // Scene 1: the status wall goes red in the order the services failed.
  function wall() {
    const start = t(S.window.start);
    $("wall-when").innerHTML = `${esc(day(start))} · ${hm(start)} UTC · <b>production</b>`;
    const onset = Object.fromEntries(S.onset.map((o) => [o.service, t(o.first_error)]));
    const span = Math.max(start, ...Object.values(onset)) - start + 30000;
    const failsAt = (s) => onset[s.name] ?? (s.status === "healthy" ? null : start + span);
    $("wall").innerHTML = S.services.map((s) => `
      <div class="tile" tabindex="0" aria-label="${esc(`${s.name}: ${s.status}, ${s.error_rate}% errors`)}">
        <p class="t-name">${esc(s.name)}</p><p class="t-state">healthy</p>
        <p class="t-rate">0.0<small>% errors</small></p>
        <p class="t-deps">${s.deps.length ? `calls ${esc(s.deps.join(", "))}` : "calls nothing"}</p></div>`).join("");
    const tiles = [...$("wall").children];
    const frame = (now) => {
      $("wall-clock").textContent = `${hms(now)} UTC`;
      tiles.forEach((el, i) => {
        const s = S.services[i], at = failsAt(s), failing = at != null && now >= at;
        const rate = at == null ? s.error_rate : failing ? s.error_rate * Math.min(1, (now - at) / 20000) : 0.2;
        el.className = `tile${failing ? ` ${s.status}` : ""}`;
        el.querySelector(".t-state").textContent = failing ? s.status : "healthy";
        el.querySelector(".t-rate").firstChild.textContent = rate.toFixed(1);
      });
    };
    let raf;
    const play = () => {
      cancelAnimationFrame(raf);
      if (still) return frame(start + span);
      const t0 = performance.now(), dur = 7000;
      const step = (ts) => {
        const p = Math.min(1, (ts - t0) / dur);
        frame(start + p * span);
        if (p < 1) raf = requestAnimationFrame(step);
      };
      raf = requestAnimationFrame(step);
    };
    $("wall-replay").onclick = play;
    frame(start);
    once($("wall"), play);
  }

  // Scene 2: find the culprit on the tape before Hindsight does.
  function game() {
    const C = S.changes, cause = new Set(S.cause);
    const start = t(S.window.start), end = t(S.window.end);
    const from = Math.min(t(C[0].when), start);
    // Same scale as the app's tape: the last two hours get 60% of the width.
    const split = Math.max(from, start - 2 * 36e5), cut = split > from ? 40 : 0;
    const pos = (ms) => (ms < split ? ((ms - from) / (split - from)) * cut
      : cut + ((ms - split) / (end - split)) * (100 - cut));
    const ago = (c) => {
      const m = Math.round((start - t(c.when)) / 6e4);
      return m < 120 ? `${m} min before the outage` : `${Math.round(m / 60)} h before the outage`;
    };
    $("game-kicker").textContent = `${C.length} changes in the ${Math.round((start - from) / 36e5)} hours before ${hm(start)}`;
    $("g-tape").innerHTML = `<div class="g-window" style="left:${pos(start)}%"></div>`
      + C.map((c, i) => `<button type="button" tabindex="-1" class="g-tick ${esc(c.kind)}" data-i="${i}"
          style="left:${pos(t(c.when))}%" aria-label="${esc(`${c.id}, ${c.service}`)}"></button>`).join("")
      + S.onset.map((o) => `<span class="g-onset" style="left:${pos(t(o.first_error))}%"
          title="${esc(`${o.service}: first error ${hms(t(o.first_error))}`)}"></span>`).join("");
    $("g-axis").innerHTML = `<span>${hm(from)}</span>`
      + (cut ? `<span class="g-mid" style="left:${cut}%">${hm(split)} · last 2 h</span>` : "")
      + `<span>outage ${hm(start)}</span>`;

    const ticks = [...$("g-tape").querySelectorAll(".g-tick")];
    const first = S.onset[0];
    const out = new Set();
    let sel = C.length - 1, guesses = 0, solved = false;

    const verdict = (kind, strong, rest) => {
      $("g-verdict").className = `g-verdict ${kind}`;
      $("g-verdict").innerHTML = `<strong>${esc(strong)}</strong> ${esc(rest)}`;
    };
    const show = (i) => {
      sel = (i + C.length) % C.length;
      const c = C[sel];
      ticks.forEach((b, j) => b.classList.toggle("sel", j === sel));
      $("g-card").className = `g-card${solved && cause.has(c.id) ? " hit" : ""}`;
      $("g-card").innerHTML = `<p class="g-meta"><b>${esc(c.id)}</b><br>${esc(c.kind)} · ${esc(c.service)}<br>
        ${hm(t(c.when))} UTC, ${ago(c)}</p><p class="g-what">${esc(c.what)}</p>`;
      $("g-accuse").disabled = solved || out.has(sel);
    };
    const reveal = (found) => {
      solved = true;
      const i = C.findIndex((c) => cause.has(c.id));
      ticks[i].classList.add("hit");
      show(i);
      $("g-give").hidden = true;
      verdict(found ? "hit" : "",
        found ? (guesses === 1 ? "First try. That's the one." : `That's the one, after ${guesses} guesses.`)
          : `It was ${C[i].id}.`,
        `A complete investigation names it in ${S.evidence.length} lookups and cites evidence for every step. Here it is.`);
    };
    const accuse = () => {
      if (solved || out.has(sel)) return;
      guesses += 1;
      if (cause.has(C[sel].id)) return reveal(true);
      out.add(sel);
      ticks[sel].classList.add("out");
      const card = $("g-card");
      card.classList.remove("miss");
      void card.offsetWidth;   // restart the shake
      card.classList.add("miss");
      $("g-accuse").disabled = true;
      verdict("miss", "Not this one.", guesses >= 4
        ? `Hint: ${first.service} failed first, so look at what changed on ${first.service}.`
        : guesses >= 2 ? `Hint: the red dots mark each service's first error. ${first.service} went first.`
          : `${C.length - cause.size - out.size} innocent changes left.`);
    };

    $("g-tape").onclick = (e) => { const b = e.target.closest(".g-tick"); if (b) show(Number(b.dataset.i)); };
    $("g-prev").onclick = () => show(sel - 1);
    $("g-next").onclick = () => show(sel + 1);
    $("g-accuse").onclick = accuse;
    $("g-give").onclick = () => reveal(false);
    document.querySelector(".game").addEventListener("keydown", (e) => {
      if (e.key === "ArrowLeft" || e.key === "ArrowRight") {
        e.preventDefault();
        show(sel + (e.key === "ArrowLeft" ? -1 : 1));
      }
    });
    verdict("", "Pick a change on the timeline, or step through them.", `${C.length - cause.size} of them are innocent.`);
    show(sel);
  }

  // Scene 3: the chain on the left, the exhibits that prove it on the right.
  function colour(e) {
    return e.result.split("\n").map((line) => {
      const l = esc(line);
      if (/\b(ERROR|CRITICAL|FATAL)\b/.test(line)) return `<span class="err">${l}</span>`;
      if (e.tool === "get_commit" && /^\+(?!\+\+)/.test(line)) return `<span class="add">${l}</span>`;
      if (e.tool === "get_commit" && /^-(?!--)/.test(line)) return `<span class="del">${l}</span>`;
      return l;
    }).join("\n");
  }

  function proof() {
    const ev = S.evidence;
    const bySvc = {};
    ev.forEach((e) => { if (e.args.service) (bySvc[e.args.service] ||= []).push(e.id); });
    const causeEv = ev.filter((e) => S.cause.some((id) => Object.values(e.args).includes(id))).map((e) => e.id);
    const trace = ev.filter((e) => e.tool === "get_trace").map((e) => e.id);
    $("proof-kicker").textContent = `Reference investigation · ${ev.length} lookups · every step cited`;
    $("p-chain").innerHTML = S.chain.map((h, i) => `<li>
      <span class="p-eff">${esc(h.effect.replace(/_/g, " "))}</span>
      <span class="p-svc">${esc(h.service)}</span>
      ${[...new Set([...(i === 0 ? causeEv : []), ...(bySvc[h.service] || []), ...trace])].map(tag).join(" ")}
      ${i === 0 ? `<span class="p-cause">started by ${esc(S.cause.join(" + "))}</span>` : ""}</li>`).join("");
    $("p-exhibits").innerHTML = ev.map((e) => `<li id="p-${esc(e.id)}">
      <header><span class="tag">${esc(e.id)}</span>
        <code>${esc(e.tool)}(${esc(Object.entries(e.args).map(([k, v]) => `${k}=${v}`).join(", "))})</code></header>
      <pre>${colour(e)}</pre></li>`).join("");
    [...$("p-exhibits").children].forEach((li) => once(li, () => li.classList.add("in")));
  }

  // Scene 4: the postmortem, built from the same facts.
  function morning() {
    const start = t(S.window.start), end = t(S.window.end);
    const c = S.changes.find((x) => S.cause.includes(x.id));
    const decoy = S.changes.filter((x) => !S.cause.includes(x.id) && t(x.when) < start).at(-1);
    const ev = S.evidence;
    const causeEv = ev.filter((e) => S.cause.some((id) => Object.values(e.args).includes(id))).map((e) => e.id);
    const logsFor = (svc) => ev.filter((e) => e.args.service === svc).map((e) => e.id);
    const symptom = S.chain.at(-1).service;
    $("m-doc").innerHTML = `
      <h3>Postmortem · ${esc(S.incident_id)}</h3>
      <p>For ${Math.round((end - start) / 6e4)} minutes from ${hm(start)} UTC on ${esc(day(start))}, ${esc(symptom)} returned errors.
        The cause was <code>${esc(c.id)}</code>, a ${esc(c.kind)} to ${esc(c.service)}: “${esc(c.what)}” ${causeEv.map(tag).join(" ")}.</p>
      <h4>How it spread</h4>
      <ol>${S.chain.map((h) => `<li>${esc(h.service)}: ${esc(h.effect.replace(/_/g, " "))} ${logsFor(h.service).map(tag).join(" ")}</li>`).join("")}</ol>
      ${decoy ? `<h4>Ruled out</h4><p>${S.changes.length - S.cause.length} other changes, including <code>${esc(decoy.id)}</code>
        (“${esc(decoy.what)}”), which landed later and closer to the outage.</p>` : ""}
      <h4>Scored against the known answer</h4><p>${Number(S.score).toFixed(2)} out of 1.00. Every tag opens the evidence it came from.</p>`;
  }

  // The clock in the corner moves from the outage to the next morning.
  function clock() {
    const start = t(S.window.start), end = t(S.window.end);
    const scenes = document.querySelectorAll(".scene[data-clock]");
    const labels = [`${hm(start)} UTC · outage begins`, `${hm(start)} UTC · what changed?`,
      `${hm(end)} UTC · outage over`, "next morning · 09:30"];
    scenes.forEach((s, i) => { s.dataset.clock = labels[i] || s.dataset.clock; });
    $("story-clock").textContent = labels[0];
    if (!("IntersectionObserver" in window)) return;
    const io = new IntersectionObserver((es) => es.forEach((x) => {
      if (x.isIntersecting) $("story-clock").textContent = x.target.dataset.clock;
    }), { rootMargin: "-45% 0px -50% 0px" });
    scenes.forEach((s) => io.observe(s));
  }
})();
