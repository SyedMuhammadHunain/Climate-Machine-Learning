// Dry run of the demo click path: runs the real predict.html JavaScript against a live server
// (with a stub DOM, no browser) and checks that the numbers on screen equal the API's numbers.
//
// Usage:  uvicorn app.main:app --port 8000   (in another terminal)
//         node scripts/dryrun_demo.js http://127.0.0.1:8000
const fs = require("fs"), vm = require("vm"), path = require("path");
const BASE = process.argv[2] || "http://127.0.0.1:8000";
const html = fs.readFileSync(path.join(__dirname, "..", "frontend", "climate-resilient-frontend", "predict.html"), "utf8");
const js = [...html.matchAll(/<script(?![^>]*src)[^>]*>([\s\S]*?)<\/script>/g)].map(m => m[1]).join("\n");

const els = {};
const mk = (init = "") => { let v = init; return {
    get value() { return v; }, set value(x) { v = String(x); },
    style: {}, innerHTML: "", textContent: "", dataset: {}, placeholder: "", onclick: null,
    children: [], replaceChildren(...kids) { this.children = kids; },
    classList: { add() {}, remove() {}, toggle() {} }, addEventListener() {}, scrollIntoView() {} }; };
const el = id => els[id] || (els[id] = mk());
el("field-province").value = "Punjab";   // the page's default
el("field-crop").value = "ALL";
const document = { createElement: () => mk(), getElementById: el, querySelector: () => el("heading"), querySelectorAll: () => [], addEventListener() {} };
const ctx = { document, fetch: (u, o) => fetch(BASE + String(u).replace(/^https?:\/\/[^/]+/, ""), o),
    console: { error() {}, log() {} }, performance, AbortSignal, AbortController, URLSearchParams, setTimeout, clearTimeout };
ctx.window = ctx; ctx.window.API_BASE = ""; ctx.location = { origin: "http://x" };
vm.createContext(ctx); vm.runInContext(js, ctx);

const strip = h => h.replace(/<[^>]+>/g, " ").replace(/&#39;/g, "'").replace(/&quot;/g, '"').replace(/&lt;/g, "<").replace(/&gt;/g, ">").replace(/&amp;/g, "&").replace(/\s+/g, " ").trim();
const sleep = ms => new Promise(r => setTimeout(r, ms));
const post = async (p, body) => (await fetch(BASE + p, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) })).json();
let failures = 0;
const check = (ok, msg) => { console.log((ok ? "  ok   " : "  FAIL ") + msg); if (!ok) failures++; };
const run = async () => { await ctx.submitPredictionForm(); await sleep(700); };

(async () => {
    await sleep(1500);
    const meta = await (await fetch(BASE + "/api/v1/metadata")).json();
    const dry = meta.scenarios.find(s => s.province === "Sindh" && s.kind === "driest");
    const pills = el("presets-pills").children;
    check(pills.length === 3 && pills.every(b => b.className === "preset-pill" && b.textContent && !b.textContent.includes("<")),
        `3 preset buttons built as DOM nodes: ${pills.map(b => b.textContent).join(" | ")}`);
    console.log(`1. Sindh, Driest real year (${dry.year}), all crops`);
    el("field-province").value = "Sindh";
    ctx.applyPreset("driest");
    check(el("presets-pills").children.map(b => b.textContent).join(" ").includes(`Driest real year (${dry.year})`), "preset pill label shows the real year");
    check(Number(el("field-precipitation").value) === dry.inputs.precipitation_sum, "rain field filled from the real year");
    await run();
    let api = await post("/api/v1/recommend", { province: "Sindh", ...dry.inputs });
    let text = strip(el("team-items-1").innerHTML);
    const yl = x => x.toFixed(2);
    api.groups.forEach(g => {
        check(text.includes(`Rank ${g.rank} \u00b7 ${g.pct_of_normal.toFixed(1)}% of a typical year`), `group rank ${g.rank}: ${g.crops.length} crop(s) at ${g.pct_of_normal.toFixed(1)}%`);
        if (g.crops.length > 1) check(text.includes(`${g.crops.length} crops tied`), `rank ${g.rank} is marked ${g.crops.length} crops tied`);
    });
    api.ranked_crops.forEach(i => check(text.includes(`${i.crop} ${i.barely_grown ? "barely grown here " : ""}`) && text.includes(`Yield ${yl(i.predicted_yield)}`), `${i.crop} yield ${yl(i.predicted_yield)}${i.barely_grown ? " (barely grown)" : ""}`));
    check(!/Rank #/.test(text) && !text.includes("tied with") && !text.includes("FAO") && !text.includes("training rows"), "rows carry no per-crop rank inside a tie, no FAO, no training rows");
    check(!text.includes("dataset yield units") && strip(el("team-title-1").textContent || "").includes(api.yield_unit), "the yield unit is stated once, in the results heading");
    check(api.groups.some(g => g.crops.length > 1), "the demo ranking has ties, so the page must group them");
    const adv = strip(el("advisory-summary").innerHTML);
    check(adv.includes(api.headline) && adv.includes(api.ranking_basis) && adv.includes(api.model_caveat) && adv.includes(api.absolute_yield_note), "advisory shows the headline, the ranking rule and both caveats from the API, unchanged");
    check(api.scenario_facts.every(f => adv.includes(f.label) && adv.includes(f.value)), "advisory shows every scenario fact");
    check(api.ranked_crops.some(i => i.barely_grown) && text.includes("barely grown here"), "barely grown crops are tagged");

    console.log("2. Typical (median) year, all crops");
    ctx.applyPreset("median");
    await run();
    const med = meta.reference.Sindh.median;
    api = await post("/api/v1/recommend", { province: "Sindh", ...med });
    text = strip(el("team-items-1").innerHTML);
    api.groups.forEach(g => check(text.includes(`${g.pct_of_normal.toFixed(1)}% of a typical year`), `typical year: group rank ${g.rank} at ${g.pct_of_normal.toFixed(1)}%`));
    check(api.groups.length === 1 && text.includes(`${api.ranked_crops.length} crops tied`), "typical year: every crop ties and the page says so");

    console.log("2b. Typical (median) year sends the exact province median: every crop keeps 100.0%, in every province");
    for (const province of ["Punjab", "Sindh", "Balochistan"]) {
        el("field-province").value = province;
        el("field-crop").value = "ALL";
        ctx.applyPreset("median");
        const exact = meta.reference[province].median;
        const optionalIds = { "field-temp-max": "temperature_2m_max", "field-temp-min": "temperature_2m_min", "field-et0": "fao_evapotranspiration", "field-radiation": "shortwave_radiation",
            "field-precip-hours": "precipitation_hours", "field-soil-mean": "soil_wetness_mean", "field-soil-max": "soil_wetness_max" };
        check(Number(el("field-precipitation").value) === exact.precipitation_sum && Number(el("field-temperature").value) === exact.temperature_2m_mean
              && Object.entries(optionalIds).every(([id, key]) => Number(el(id).value) === exact[key]), `${province}: all nine fields hold the exact median`);
        await run();
        const typical = await post("/api/v1/recommend", { province, ...exact });
        const shown = strip(el("team-items-1").innerHTML);
        check(typical.groups.length === 1 && typical.groups[0].pct_of_normal === 100.0 && shown.includes(`100.0% of a typical year`) && shown.includes(`${typical.ranked_crops.length} crops tied`), `${province}: all ${typical.ranked_crops.length} crops at 100.0% on the page`);
    }
    el("field-province").value = "Sindh";

    console.log("3. Rice, Driest real year");
    el("field-crop").value = "rice";
    ctx.applyPreset("driest");
    await run();
    const one = await post("/api/v1/predict", { province: "Sindh", crop: "rice", ...dry.inputs });
    text = strip(el("single-crop-items").innerHTML);
    check(text.includes(`Predicted yield: ${one.predicted_yield.toFixed(2)}`), `predicted yield ${one.predicted_yield.toFixed(2)}`);
    check(text.includes(`Typical year in Sindh: ${one.normal_yield.toFixed(2)}`), `typical yield ${one.normal_yield.toFixed(2)}`);
    check(text.includes(`${one.pct_of_normal.toFixed(1)}% of typical`), `${one.pct_of_normal.toFixed(1)}% of typical`);

    console.log("4. Rain 5 mm (outside the trained range)");
    el("field-precipitation").value = "5";
    el("validation-alert").style.display = "none";
    await run();
    const bad = await post("/api/v1/predict", { province: "Sindh", crop: "rice", ...dry.inputs, precipitation_sum: 5 });
    check(el("validation-alert").style.display === "flex", "alert is shown");
    check(el("validation-msg").textContent === bad.message, "alert text equals the API message");

    console.log(failures ? `\nFAILED: ${failures} check(s)` : "\nPASSED: demo click path matches the API.");
    process.exit(failures ? 1 : 0);
})();
