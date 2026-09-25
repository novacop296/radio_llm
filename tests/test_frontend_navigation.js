/**
 * Automated Frontend Navigation & Tab Synchronization Test Suite
 * Tests all 9 top-level tabs and their corresponding workspace containers.
 */

const fs = require('fs');
const path = require('path');

// 1. Build a mini DOM environment
class ClassList {
  constructor(el) {
    this.el = el;
    this.classes = new Set();
  }
  add(cls) { this.classes.add(cls); }
  remove(cls) { this.classes.delete(cls); }
  contains(cls) { return this.classes.has(cls); }
  toggle(cls, force) {
    if (force !== undefined) {
      if (force) this.add(cls);
      else this.remove(cls);
    } else {
      if (this.contains(cls)) this.remove(cls);
      else this.add(cls);
    }
  }
}

class DOMElement {
  constructor(id, tagName = 'div', classes = []) {
    this.id = id;
    this.tagName = tagName.toUpperCase();
    this.style = {};
    this.classList = new ClassList(this);
    classes.forEach(c => this.classList.add(c));
    this.listeners = {};
    this.dataset = {};
    this.children = [];
    this.innerHTML = '';
    this.textContent = '';
    this.value = '';
    this.disabled = false;
  }

  addEventListener(event, handler) {
    if (!this.listeners[event]) this.listeners[event] = [];
    this.listeners[event].push(handler);
  }

  click() {
    const handlers = this.listeners['click'] || [];
    const event = {
      preventDefault: () => {},
      stopPropagation: () => {},
      target: this,
      currentTarget: this
    };
    handlers.forEach(h => h(event));
    if (typeof this.onclick === 'function') {
      this.onclick(event);
    }
  }

  querySelector(sel) {
    return this.querySelectorAll(sel)[0] || null;
  }

  querySelectorAll(sel) {
    const results = [];
    if (sel.startsWith('.')) {
      const cls = sel.substring(1);
      if (this.classList.contains(cls)) results.push(this);
    }
    this.children.forEach(child => {
      results.push(...child.querySelectorAll(sel));
    });
    return results;
  }

  appendChild(child) {
    this.children.push(child);
  }

  removeChild(child) {
    const idx = this.children.indexOf(child);
    if (idx !== -1) this.children.splice(idx, 1);
  }

  getContext() {
    return {
      clearRect: () => {},
      fillRect: () => {},
      strokeRect: () => {},
      drawImage: () => {},
      save: () => {},
      restore: () => {},
      scale: () => {},
      translate: () => {}
    };
  }

  setAttribute(k, v) { this[k] = v; }
  getAttribute(k) { return this[k] || null; }
  getBoundingClientRect() { return { left: 0, top: 0, width: 512, height: 512 }; }
}

class MockDocument {
  constructor() {
    this.elements = new Map();
    this.body = new DOMElement('body', 'body');
    this.listeners = {};
  }

  createElement(tag) {
    return new DOMElement('', tag);
  }

  getElementById(id) {
    if (!id) return null;
    if (!this.elements.has(id)) {
      this.elements.set(id, new DOMElement(id));
    }
    return this.elements.get(id);
  }

  querySelectorAll(sel) {
    const results = [];
    for (const el of this.elements.values()) {
      if (sel.includes('.main-tab-btn') && el.classList.contains('main-tab-btn')) {
        results.push(el);
      } else if (sel.startsWith('.') && el.classList.contains(sel.substring(1))) {
        results.push(el);
      }
    }
    return results;
  }

  querySelector(sel) {
    return this.querySelectorAll(sel)[0] || null;
  }

  addEventListener(event, handler) {
    if (!this.listeners[event]) this.listeners[event] = [];
    this.listeners[event].push(handler);
  }
}

// 2. Setup environment and load app.js
const mockDoc = new MockDocument();
const mockWindow = {
  addEventListener: () => {},
  open: () => {},
  fetch: async () => ({
    ok: true,
    status: 200,
    json: async () => ({}),
    text: async () => ''
  }),
  URL: {
    createObjectURL: () => 'blob:mock',
    revokeObjectURL: () => {}
  }
};

global.document = mockDoc;
global.window = mockWindow;
global.fetch = mockWindow.fetch;
global.Blob = class {};
global.Image = class { constructor() { setTimeout(() => { if (this.onload) this.onload(); }, 10); } };
global.alert = (msg) => {};
global.confirm = (msg) => true;

// Pre-register all 9 tab buttons and containers
const TAB_DEFS = [
  { key: 'dataset', btnId: 'tabDatasetQueue', containerId: 'datasetQueueViewContainer', name: 'Dataset & Review Queue' },
  { key: 'individual', btnId: 'tabIndividualReview', containerId: 'individualViewContainer', name: 'Individual Human Review' },
  { key: 'consensus', btnId: 'tabConsensusDashboard', containerId: 'consensusDashboardContainer', name: 'Multi-Reviewer Consensus & Adjudication' },
  { key: 'analytics', btnId: 'tabEvaluationAnalytics', containerId: 'evaluationAnalyticsContainer', name: 'Evaluation & Workflow Analytics' },
  { key: 'research_eval', btnId: 'tabResearchEvaluation', containerId: 'researchEvaluationViewContainer', name: 'Research Evaluation' },
  { key: 'registry', btnId: 'tabExperimentRegistry', containerId: 'experimentRegistryViewContainer', name: 'Experiment Registry' },
  { key: 'benchmarking', btnId: 'tabExternalBenchmarking', containerId: 'externalBenchmarkingViewContainer', name: 'External Benchmarking' },
  { key: 'experiments', btnId: 'tabResearchExperiments', containerId: 'researchExperimentsContainer', name: 'Research Experiments' },
  { key: 'counterfactual', btnId: 'tabCounterfactualExplainability', containerId: 'counterfactualViewContainer', name: 'Counterfactual Explainability' }
];

TAB_DEFS.forEach(t => {
  const btn = mockDoc.getElementById(t.btnId);
  btn.classList.add('main-tab-btn');
  const container = mockDoc.getElementById(t.containerId);
  container.style.display = (t.key === 'individual') ? 'block' : 'none';
});
mockDoc.getElementById('individualProgressBar').style.display = 'block';

// Parse and run app.js
const appJsPath = path.join(__dirname, '..', 'frontend', 'app.js');
const appJsCode = fs.readFileSync(appJsPath, 'utf8');

try {
  eval(appJsCode);
} catch (e) {
  console.error('FATAL: Failed to evaluate app.js:', e);
  process.exit(1);
}

// Trigger DOMContentLoaded
if (mockDoc.listeners['DOMContentLoaded']) {
  mockDoc.listeners['DOMContentLoaded'].forEach(h => h());
}

// 3. Test Navigation across all 9 tabs
let passed = 0;
let total = 0;

function assert(condition, message) {
  total++;
  if (condition) {
    passed++;
    console.log(`  [PASS] ${message}`);
  } else {
    console.error(`  [FAIL] ${message}`);
    process.exitCode = 1;
  }
}

console.log('================================================================');
console.log('RUNNING AUTOMATED 9-TAB NAVIGATION TEST SUITE');
console.log('================================================================\n');

TAB_DEFS.forEach(target => {
  console.log(`Testing Navigation to Tab: ${target.name} (${target.btnId})`);
  const btn = mockDoc.getElementById(target.btnId);
  btn.click();

  // Check button active state
  assert(btn.classList.contains('active'), `Button #${target.btnId} has 'active' class`);

  // Check other buttons inactive
  TAB_DEFS.filter(o => o.key !== target.key).forEach(other => {
    const otherBtn = mockDoc.getElementById(other.btnId);
    assert(!otherBtn.classList.contains('active'), `Other button #${other.btnId} is NOT active`);
  });

  // Check target container visible
  const container = mockDoc.getElementById(target.containerId);
  assert(container.style.display === 'block', `Container #${target.containerId} display is 'block'`);

  // Check other containers hidden
  TAB_DEFS.filter(o => o.key !== target.key).forEach(other => {
    const otherContainer = mockDoc.getElementById(other.containerId);
    assert(otherContainer.style.display === 'none', `Other container #${other.containerId} display is 'none'`);
  });

  console.log('');
});

// Test Round-Trip / Repeat Navigation
console.log('Testing Repeated / Switching Navigation Cycle...');
const testCycle = ['dataset', 'counterfactual', 'benchmarking', 'consensus', 'experiments', 'individual'];
testCycle.forEach(k => {
  const def = TAB_DEFS.find(d => d.key === k);
  const btn = mockDoc.getElementById(def.btnId);
  btn.click();
  const cont = mockDoc.getElementById(def.containerId);
  assert(cont.style.display === 'block' && btn.classList.contains('active'), `Successfully navigated in cycle to: ${def.name}`);
});

console.log('\n================================================================');
console.log(`NAVIGATION TEST SUMMARY: ${passed} / ${total} CHECKS PASSED`);
console.log('================================================================');

if (passed === total) {
  process.exit(0);
} else {
  process.exit(1);
}
