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

    console.log("5. Crop pages: the real crop.html JavaScript, 13 crops, against the API");
    const cropHtml = fs.readFileSync(path.join(__dirname, "..", "frontend", "climate-resilient-frontend", "crop.html"), "utf8");
    const cropJs = [...cropHtml.matchAll(/<script(?![^>]*src)[^>]*>([\s\S]*?)<\/script>/g)].map(m => m[1]).join("\n");
    const loadCrop = async search => {
        const store = {};
        const get = id => store[id] || (store[id] = mk());
        const doc = { getElementById: get, createElement: () => mk(), body: { dataset: {} }, title: "", querySelector: () => mk(), querySelectorAll: () => [] };
        const c = { document: doc, fetch: (u, o) => fetch(BASE + String(u).replace(/^https?:\/\/[^/]+/, ""), o), console: { error() {}, log() {} },
            AbortSignal, URLSearchParams, setTimeout, clearTimeout };
        c.window = c; c.window.API_BASE = ""; c.location = { origin: "http://x", search };
        vm.createContext(c); vm.runInContext(cropJs, c);
        for (let i = 0; i < 100 && doc.body.dataset.ready !== "true"; i++) await sleep(100);
        return { get, doc };
    };
    const scenario = meta.scenarios.find(s => s.province === "Sindh" && s.kind === "driest");
    const wet = meta.scenarios.find(s => s.province === "Sindh" && s.kind === "wettest");
    const rec = inputs => post("/api/v1/recommend", { province: "Sindh", ...inputs });
    const base = await rec(scenario.inputs);
    const years = [["Driest real year (" + scenario.year + ")", await rec(scenario.inputs)], ["Typical (median) year", await rec(meta.reference.Sindh.median)],
                   ["Wettest real year (" + wet.year + ")", await rec(wet.inputs)]];
    const order = base.ranked_crops.map(i => i.crop);
    const groupRank = (r, crop) => r.groups.find(g => g.crops.includes(crop));
    for (const [at, crop] of order.entries()) {
        const { get, doc } = await loadCrop("?" + new URLSearchParams({ crop, province: "Sindh", ...scenario.inputs }).toString());
        const item = base.ranked_crops.find(i => i.crop === crop), text = strip(get("crop-grid").innerHTML);
        const g = groupRank(base, crop), others = g.crops.length - 1;
        const rank = others === 0 ? `Rank ${g.rank}` : `Rank ${g.rank}, tied with ${others} other crop${others > 1 ? "s" : ""} at ${g.pct_of_normal.toFixed(1)}%`;
        const rows = years.map(([label, r]) => { const it = r.ranked_crops.find(i => i.crop === crop); return `${label} ${groupRank(r, crop).rank} ${it.pct_of_normal.toFixed(1)}% ${it.predicted_yield.toFixed(2)}`; });
        const fao = item.fao_water_need;
        const water = fao ? `${fao.mm_min}-${fao.mm_max} mm per growing period, drought sensitivity ${fao.drought_sensitivity}` : "not listed in the FAO table";
        const prev = order[(at + 12) % 13], next = order[(at + 1) % 13];
        const ok = get("crop-title").textContent === crop && get("crop-province").textContent === "Sindh"
            && text.includes(`Predicted yield ${item.predicted_yield.toFixed(2)}`) && text.includes(`Typical year yield ${item.normal_yield.toFixed(2)}`)
            && text.includes(`Share of a typical year ${item.pct_of_normal.toFixed(1)}%`) && text.includes(rank)
            && rows.every(r => text.includes(r)) && text.includes(water)
            && text.includes(`Training rows: ${item.training_rows_for_crop} of ${item.training_rows_possible}`)
            && text.includes(item.barely_grown ? "Barely grown here" : "Training rows") && (item.barely_grown === text.includes("Barely grown here"))
            && text.includes(base.model_caveat) && text.includes(base.absolute_yield_note) && (text.match(/dataset yield units/g) || []).length === 1
            && get("crop-prev").href.includes(`crop=${prev}&`) && get("crop-prev").textContent === `Previous: ${prev}`
            && get("crop-next").href.includes(`crop=${next}&`) && get("crop-next").textContent === `Next: ${next}`
            && get("crop-content").hidden === false && get("crop-error").hidden === true;
        check(ok, `${crop}: rank, yields, share, real-year rows, water need, training rows, caveats and prev/next equal the API`);
    }
    const badCases = [["?crop=quinoa&province=Sindh", "quinoa"], ["?crop=rice&province=Atlantis", "Atlantis"], ["?province=Sindh", "No crop"],
        ["?" + new URLSearchParams({ crop: "rice", province: "Sindh", ...scenario.inputs, precipitation_sum: "5" }).toString(),
         (await post("/api/v1/recommend", { province: "Sindh", ...scenario.inputs, precipitation_sum: 5 })).message]];
    for (const [search, expected] of badCases) {
        const { get } = await loadCrop(search);
        const shown = strip(get("crop-error").innerHTML);
        check(shown.includes(expected) && get("crop-error").hidden === false && get("crop-content").hidden === true && get("crop-grid").innerHTML === "",
            `bad address ${search.slice(0, 40)}: the message is shown, no numbers`);
    }

    console.log(failures ? `\nFAILED: ${failures} check(s)` : "\nPASSED: demo click path matches the API.");
    process.exit(failures ? 1 : 0);
})();
