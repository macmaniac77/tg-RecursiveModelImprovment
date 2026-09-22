// Optional DOM smoke test: NODE_PATH=<jsdom installation>/node_modules node tests/viewer_catalog_smoke.cjs artifacts/tinygrad-viewer.html
const fs = require('node:fs');
const assert = require('node:assert/strict');
const {JSDOM, VirtualConsole} = require('jsdom');
const errors = [];
const virtualConsole = new VirtualConsole();
virtualConsole.on('jsdomError', error => errors.push(error.message));
const dom = new JSDOM(fs.readFileSync(process.argv[2], 'utf8'), {runScripts: 'dangerously', virtualConsole});
const doc = dom.window.document;
function button(text, root=doc) {
  const node = [...root.querySelectorAll('button')].find(b=>b.textContent===text);
  assert.ok(node, `Missing button ${text}`);return node;
}
button('Tinygrad examples').click();
assert.match(doc.querySelector('#content').textContent, /86 Python files/);
const search = doc.querySelector('[aria-label="Search Tinygrad examples"]');
search.value='yolov8.py';search.dispatchEvent(new dom.window.Event('input'));
const file = [...doc.querySelectorAll('#content details')].find(d=>d.firstChild.textContent.startsWith('examples/yolov8.py'));
assert.ok(file);file.open=true;file.dispatchEvent(new dom.window.Event('toggle'));
button('C2f', file).click();
assert.match(doc.querySelector('#detail').textContent, /Inference only/);
button('YOLOV8_C2F', doc.querySelector('#detail')).click();
assert.match(doc.querySelector('#detail').textContent, /Primitive coverage complete/);
button('Tinygrad examples').click();
const llmSearch=doc.querySelector('[aria-label="Search Tinygrad examples"]');
llmSearch.value='tinygrad/llm/model.py';llmSearch.dispatchEvent(new dom.window.Event('input'));
const llmFile=[...doc.querySelectorAll('#content details')].find(d=>d.firstChild.textContent.startsWith('tinygrad/llm/model.py'));
assert.ok(llmFile);llmFile.open=true;llmFile.dispatchEvent(new dom.window.Event('toggle'));
button('GatedDeltaNetBlock._attention',llmFile).click();
button('GATED_DELTA_SCAN',doc.querySelector('#detail')).click();
assert.match(doc.querySelector('#detail').textContent,/Primitive coverage complete/);
button('AOI & primitive audit').click();
const filter = doc.querySelector('[aria-label="Filter AOIs"]');
filter.value='swiglu';filter.dispatchEvent(new dom.window.Event('input'));
button('SWIGLU').click();assert.match(doc.querySelector('#detail').textContent,/gate.weight/);
assert.deepEqual(errors, []);
console.log('DOM smoke passed: example search, source selection, library navigation, AOI filtering.');
dom.window.close();
