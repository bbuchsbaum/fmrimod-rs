// lean-browser-playwright: fresh installed Chromium, no system/user profile.
const {chromium}=require('playwright');
const fs=require('node:fs');const path=require('node:path');const assert=require('node:assert/strict');
(async()=>{
 const [infoFile,outDir]=process.argv.slice(2);fs.mkdirSync(outDir,{recursive:true});
 const info=JSON.parse(fs.readFileSync(infoFile,'utf8'));
 const browser=await chromium.launch({headless:true});
 let context;
 try {
  context=await browser.newContext({viewport:{width:1200,height:1000},deviceScaleFactor:1});
  const page=await context.newPage();const errors=[];page.on('pageerror',e=>errors.push(e.message));
  const reports=[];
  await page.goto(`http://127.0.0.1:${info.port}/`,{waitUntil:'load'});
  await page.getByRole('heading',{name:'Design → group contrast'}).waitFor();
  async function run(name,change,expected){
   if(change.phase!==undefined)await page.locator('#phase').fill(String(change.phase));
   for(const key of ['contrast','noise','mode'])if(change[key])await page.locator('#'+key).selectOption(change[key]);
   if(change.variance!==undefined)await page.locator('#variance').setChecked(change.variance);
   const previous=await page.evaluate(()=>window.renderSequence);
   await page.getByRole('button',{name:'Run analysis',exact:true}).click();
   await page.waitForFunction(n=>window.renderSequence>n||document.querySelector('#error').textContent.length>0,previous,{timeout:120000});
   assert.equal(await page.locator('#error').textContent(),'');
   const state=await page.evaluate(()=>({timing:window.lastRender,reads:window.lastResult.source_reads,prepares:window.lastResult.model_preparations,builds:window.lastResult.retained_fit_builds,plan:window.lastResult.plan,features:window.lastResult.effects.length,finite:window.lastResult.effects.every(Number.isFinite),variance:window.lastResult.variance,painted:Array.from(document.querySelector('#map').getContext('2d').getImageData(0,0,48,48).data).filter((v,i)=>i%4===0&&v!==6).length}));
   assert.equal(state.reads,expected.reads);assert.equal(state.prepares,expected.prepares);assert.equal(state.builds,expected.builds);assert(state.finite);assert(state.painted>100);assert.equal(state.plan.subjects.length,25);
   const screenshot=path.join(outDir,name+'.png');await page.screenshot({path:screenshot,fullPage:true});
   state.timing.screenshot_confirmed_upper_bound_ms=await page.evaluate(()=>performance.now()-window.requestStart);
   reports.push({name,...state,screenshot:path.basename(screenshot)});console.log(name,JSON.stringify(state.timing));
  }
  await run('cold-estimated',{}, {reads:25,prepares:25,builds:25});
  await page.getByRole('button',{name:'Inspect plan',exact:true}).click();
  await page.waitForFunction(()=>window.lastInspection?.inspection_only===true);
  assert((await page.evaluate(()=>window.lastInspection.plan.subjects)).every(s=>s.model==='Reuse'&&s.fit==='UseRetainedFit'));
  await run('contrast-edit',{contrast:'difference'},{reads:0,prepares:0,builds:0});
  await run('design-edit',{phase:.18},{reads:0,prepares:25,builds:25});
  await run('repeat',{}, {reads:0,prepares:0,builds:0});
  await run('estimates-only',{variance:false},{reads:0,prepares:0,builds:0});
  assert(reports.at(-1).plan.subjects.every(s=>s.uncertainty==='None'));
  await run('file-batches',{mode:'files',variance:true},{reads:25,prepares:25,builds:0});
  await run('file-contrast',{contrast:'task'},{reads:25,prepares:0,builds:0});
  await run('fixed-noise',{noise:'fixed'},{reads:25,prepares:25,builds:0});
  const before=await page.evaluate(()=>window.drawCount);
  for(const z of [0,31]){
   await page.locator('#slice').focus();await page.locator('#slice').press(z===0?'Home':'End');
   assert.equal(await page.locator('#sliceValue').textContent(),String(z));
  }
  assert((await page.evaluate(()=>window.drawCount))>=before+2);
  assert.deepEqual(errors,[]);
  fs.writeFileSync(path.join(outDir,'browser.json'),JSON.stringify({backend:'installed Playwright Chromium',headless:true,playwright:require('playwright/package.json').version,browser:browser.version(),owner_node_pid:process.pid,profile:'fresh ephemeral context',cases:reports,slider_endpoints:[0,31],page_errors:errors,timing_definition:'Browser performance clock: request submission through canvas update and two animation frames; screenshot completion is a render-confirmed upper bound, not physical display latency.'},null,2)+'\n');
 } finally {if(context)await context.close();await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
