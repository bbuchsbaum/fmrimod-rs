// lean-browser-playwright: fresh installed Chromium; no user/system browser profile.
const {chromium} = require('playwright');
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
(async () => {
  const [base, output] = process.argv.slice(2);
  fs.mkdirSync(output, {recursive: true});
  fs.rmSync(path.join(output, "browser.json"), {force: true});
  const browser = await chromium.launch({headless: true});
  const errors = [];
  let context;
  try {
    context = await browser.newContext({viewport: {width: 1280, height: 900}});
    const page = await context.newPage();
    page.on('pageerror', error => errors.push(error.message));
    await page.goto(base, {waitUntil: 'networkidle'});
    await page.getByRole('heading', {name: 'Model fMRI data in Rust', exact: true}).waitFor();
    await page.screenshot({path: path.join(output, 'guide-desktop.png'), fullPage: true});
    await page.locator("#mdbook-search-toggle").click();
    await page.locator("#mdbook-searchbar").pressSequentially("AR noise");
    await page.locator("#mdbook-searchresults a").first().waitFor({state: "visible"});
    await page.locator("#mdbook-searchbar").fill("");
    await page.locator("#mdbook-search-toggle").click();
    await page.locator('a[href="fit.html"]').first().click();
    await page.getByRole('heading', {name: 'Fit subjects with AR noise', exact: true}).waitFor();
    assert((await page.locator('pre').innerText()).includes('NoisePolicy::EstimateAr'));
    await page.goto(new URL('group.html', base).href);
    await page.getByRole('heading', {name: 'Estimate a population effect', exact: true}).waitFor();
    assert((await page.locator('main').innerText()).includes('conditional first-level uncertainty'));
    await page.goto(new URL('api/fmrimod/index.html', base).href, {waitUntil: 'networkidle'});
    assert((await page.locator('h1').innerText()).includes('fmrimod'));
    await page.screenshot({path: path.join(output, 'api-desktop.png'), fullPage: true});
    await page.goto(new URL("api/fmrimod/index.html?search=RepeatedCohort", base).href);
    await page.locator('a[href*="struct.RepeatedCohort.html"]').first().waitFor({state: "visible", timeout: 15000});
    await page.locator('a[href*="struct.RepeatedCohort.html"]').first().click();
    assert((await page.locator("h1").innerText()).includes("RepeatedCohort"));
    await page.goto(new URL('api/fmri_splines/types/spline_traits/trait.SplineBuilder.html', base).href, {waitUntil: 'networkidle'});
    assert((await page.locator('h1').innerText()).includes('SplineBuilder'));
    await page.setViewportSize({width: 390, height: 844});
    await page.goto(new URL('outputs.html', base).href, {waitUntil: 'networkidle'});
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1));
    await page.screenshot({path: path.join(output, 'guide-mobile.png'), fullPage: true});
    assert.deepEqual(errors, []);
    fs.writeFileSync(path.join(output, 'browser.json'), JSON.stringify({base, browser: await browser.version(),
      passed: true, desktop: true, mobile_width: 390, navigation: true, guide_search: true, api_search: true, api: true, page_errors: errors}, null, 2));
  } finally {
    if (context) await context.close();
    await browser.close();
  }
})().catch(error => {console.error(error); process.exit(1)});
