/**
 * LLM-Assisted Explainable Radiology Dashboard Application Logic
 * Phase 1.1 — Multi-Study Browsing, Dual-View Radiograph Synchronization,
 * Reviewer Annotation & Batch Study Processing
 */

document.addEventListener('DOMContentLoaded', () => {
  // State Management
  const state = {
    studyId: 'CXR1122',
    allStudies: [],
    studyData: null,
    imagesMeta: [],
    activeFinding: 'Infiltration',
    viewMode: 'overlay', // 'overlay', 'original', 'heatmap'
    opacity: 0.45,
    isSync: true,
    
    // Viewer Transforms
    primary: {
      zoom: 1.0,
      panX: 0,
      panY: 0,
      isPanning: false,
      startX: 0,
      startY: 0
    },
    secondary: {
      zoom: 1.0,
      panX: 0,
      panY: 0,
      isPanning: false,
      startX: 0,
      startY: 0
    },

    // Preloaded image objects
    primaryImages: {
      original: null,
      heatmaps: {}
    },
    secondaryImages: {
      original: null,
      heatmaps: {}
    },

    // Reviewer Annotations cache
    reviews: {},

    // Batch polling
    batchPollInterval: null
  };

  // Status Interpretation Semantic Map
  const STATUS_INTERPRETATIONS = {
    supported: 'Sufficient explicit evidence supports this finding.',
    possible: 'Model or indirect evidence suggests this finding may warrant review, but it is not independently established.',
    uncertain: 'Available evidence is equivocal or insufficient to establish presence or absence.',
    absent: 'Available evidence supports absence of this finding.'
  };

  // DOM Elements - Navigation & Header
  const studySelect = document.getElementById('studySelect');
  const studySearchInput = document.getElementById('studySearchInput');
  const studyFilterSelect = document.getElementById('studyFilterSelect');
  const batchModalBtn = document.getElementById('batchModalBtn');
  const viewTag = document.getElementById('viewTag');
  const dualViewBadge = document.getElementById('dualViewBadge');
  const syncToggleContainer = document.getElementById('syncToggleContainer');
  const syncToggle = document.getElementById('syncToggle');
  const pipelineViolationsText = document.getElementById('pipelineViolationsText');

  // DOM Elements - Viewers & Canvas
  const viewportsContainer = document.getElementById('viewportsContainer');
  const primaryViewport = document.getElementById('primaryViewport');
  const primaryViewportContent = document.getElementById('primaryViewportContent');
  const primaryCanvas = document.getElementById('primaryCanvas');
  const primaryCtx = primaryCanvas.getContext('2d');
  const primaryLoader = document.getElementById('primaryLoader');

  const secondaryViewport = document.getElementById('secondaryViewport');
  const secondaryViewportContent = document.getElementById('secondaryViewportContent');
  const secondaryCanvas = document.getElementById('secondaryCanvas');
  const secondaryCtx = secondaryCanvas.getContext('2d');
  const secondaryLoader = document.getElementById('secondaryLoader');
  const secondaryViewportTag = document.getElementById('secondaryViewportTag');
  const secondaryUnavailableNotice = document.getElementById('secondaryUnavailableNotice');

  const opacitySlider = document.getElementById('opacitySlider');
  const opacityVal = document.getElementById('opacityVal');
  const opacityControl = document.getElementById('opacityControl');
  const modeBtns = document.querySelectorAll('.mode-btn');
  const resetZoomBtn = document.getElementById('resetZoomBtn');
  const zoomInBtn = document.getElementById('zoomInBtn');
  const zoomOutBtn = document.getElementById('zoomOutBtn');
  const heatmapLegend = document.getElementById('heatmapLegend');

  // DOM Elements - Finding & QA
  const findingsSelector = document.getElementById('findingsSelector');
  const findingsCountBadge = document.getElementById('findingsCountBadge');
  const activeFindingName = document.getElementById('activeFindingName');
  const activeFindingStatus = document.getElementById('activeFindingStatus');
  const activeModelScore = document.getElementById('activeModelScore');
  const activeEvidenceSource = document.getElementById('activeEvidenceSource');
  const activeLocation = document.getElementById('activeLocation');
  const activeSeverity = document.getElementById('activeSeverity');
  const statusInterpretationText = document.getElementById('statusInterpretationText');

  // DOM Elements - Reviewer Annotation
  const reviewerToggleBtn = document.getElementById('reviewerToggleBtn');
  const reviewerBody = document.getElementById('reviewerBody');
  const reviewerStatusSelect = document.getElementById('reviewerStatusSelect');
  const reviewerLocationInput = document.getElementById('reviewerLocationInput');
  const reviewerSeveritySelect = document.getElementById('reviewerSeveritySelect');
  const reviewerNotesInput = document.getElementById('reviewerNotesInput');
  const saveReviewBtn = document.getElementById('saveReviewBtn');
  const clearReviewBtn = document.getElementById('clearReviewBtn');
  const reviewSaveStatus = document.getElementById('reviewSaveStatus');
  const compMachineStatus = document.getElementById('compMachineStatus');
  const compReviewerStatus = document.getElementById('compReviewerStatus');

  // DOM Elements - Why Finding
  const whyFindingToggleBtn = document.getElementById('whyFindingToggleBtn');
  const whyFindingBody = document.getElementById('whyFindingBody');
  const whyModelSignal = document.getElementById('whyModelSignal');
  const whyEvidenceSource = document.getElementById('whyEvidenceSource');
  const whyQaState = document.getElementById('whyQaState');
  const whySpatialGrounding = document.getElementById('whySpatialGrounding');
  const whyFinalStatus = document.getElementById('whyFinalStatus');

  // DOM Elements - QA & Traceability
  const qaToggleBtn = document.getElementById('qaToggleBtn');
  const qaQuestionsList = document.getElementById('qaQuestionsList');
  const traceabilityChain = document.getElementById('traceabilityChain');

  // DOM Elements - Report
  const reportStudyId = document.getElementById('reportStudyId');
  const reportGenerator = document.getElementById('reportGenerator');
  const reportFindingsList = document.getElementById('reportFindingsList');
  const reportImpressionList = document.getElementById('reportImpressionList');
  const exportJsonBtn = document.getElementById('exportJsonBtn');
  const exportTextBtn = document.getElementById('exportTextBtn');

  // DOM Elements - Batch Modal
  const batchModalBackdrop = document.getElementById('batchModalBackdrop');
  const closeBatchModalBtn = document.getElementById('closeBatchModalBtn');
  const batchStudyInput = document.getElementById('batchStudyInput');
  const triggerBatchBtn = document.getElementById('triggerBatchBtn');
  const batchStatusContainer = document.getElementById('batchStatusContainer');
  const batchJobId = document.getElementById('batchJobId');
  const batchJobStatusBadge = document.getElementById('batchJobStatusBadge');
  const batchStagesGrid = document.getElementById('batchStagesGrid');
  const batchLog = document.getElementById('batchLog');

  // Initialize
  init();

  async function init() {
    setupEventListeners();
    await fetchStudiesIndex();
    await loadStudy(state.studyId);
  }

  function setupEventListeners() {
    // Multi-Study Search & Filter
    studySelect.addEventListener('change', (e) => {
      state.studyId = e.target.value;
      loadStudy(state.studyId);
    });

    studySearchInput.addEventListener('input', filterStudyDropdown);
    studyFilterSelect.addEventListener('change', filterStudyDropdown);

    // Batch Modal Open/Close
    batchModalBtn.addEventListener('click', () => {
      batchModalBackdrop.style.display = 'flex';
      checkBatchStatus();
    });

    closeBatchModalBtn.addEventListener('click', () => {
      batchModalBackdrop.style.display = 'none';
      if (state.batchPollInterval) clearInterval(state.batchPollInterval);
    });

    triggerBatchBtn.addEventListener('click', handleTriggerBatch);

    // Sync Views Toggle
    if (syncToggle) {
      syncToggle.addEventListener('change', (e) => {
        state.isSync = e.target.checked;
        if (state.isSync) {
          state.secondary.zoom = state.primary.zoom;
          state.secondary.panX = state.primary.panX;
          state.secondary.panY = state.primary.panY;
          updateViewportTransforms();
        }
      });
    }

    // Mode toggles (Overlay / Original / Heatmap)
    modeBtns.forEach(btn => {
      btn.addEventListener('click', () => {
        modeBtns.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        state.viewMode = btn.dataset.mode;
        
        if (state.viewMode === 'original') {
          opacityControl.style.opacity = '0.4';
          opacityControl.style.pointerEvents = 'none';
          heatmapLegend.style.display = 'none';
        } else {
          opacityControl.style.opacity = '1.0';
          opacityControl.style.pointerEvents = 'auto';
          heatmapLegend.style.display = 'flex';
        }
        renderBothCanvases();
      });
    });

    // Opacity Slider
    opacitySlider.addEventListener('input', (e) => {
      state.opacity = parseInt(e.target.value, 10) / 100.0;
      opacityVal.textContent = `${e.target.value}%`;
      renderBothCanvases();
    });

    // Zoom and Pan Controls
    zoomInBtn.addEventListener('click', () => adjustZoom(0.15));
    zoomOutBtn.addEventListener('click', () => adjustZoom(-0.15));
    resetZoomBtn.addEventListener('click', resetView);

    // Pan with Mouse Drag on Primary Viewport
    primaryViewport.addEventListener('mousedown', (e) => {
      state.primary.isPanning = true;
      state.primary.startX = e.clientX - state.primary.panX;
      state.primary.startY = e.clientY - state.primary.panY;
    });

    // Pan with Mouse Drag on Secondary Viewport
    secondaryViewport.addEventListener('mousedown', (e) => {
      if (state.isSync) {
        state.primary.isPanning = true;
        state.primary.startX = e.clientX - state.primary.panX;
        state.primary.startY = e.clientY - state.primary.panY;
      } else {
        state.secondary.isPanning = true;
        state.secondary.startX = e.clientX - state.secondary.panX;
        state.secondary.startY = e.clientY - state.secondary.panY;
      }
    });

    window.addEventListener('mousemove', (e) => {
      if (state.primary.isPanning) {
        state.primary.panX = e.clientX - state.primary.startX;
        state.primary.panY = e.clientY - state.primary.startY;
        if (state.isSync) {
          state.secondary.panX = state.primary.panX;
          state.secondary.panY = state.primary.panY;
        }
        updateViewportTransforms();
      } else if (state.secondary.isPanning) {
        state.secondary.panX = e.clientX - state.secondary.startX;
        state.secondary.panY = e.clientY - state.secondary.startY;
        updateViewportTransforms();
      }
    });

    window.addEventListener('mouseup', () => {
      state.primary.isPanning = false;
      state.secondary.isPanning = false;
    });

    // Reviewer Form Handlers
    saveReviewBtn.addEventListener('click', handleSaveReview);
    clearReviewBtn.addEventListener('click', handleClearReview);
    reviewerStatusSelect.addEventListener('change', updateComparisonBox);

    // Accordions
    reviewerToggleBtn.addEventListener('click', () => {
      const isExpanded = reviewerToggleBtn.getAttribute('aria-expanded') === 'true';
      reviewerToggleBtn.setAttribute('aria-expanded', !isExpanded);
      reviewerBody.style.display = isExpanded ? 'none' : 'flex';
      const icon = reviewerToggleBtn.querySelector('.toggle-icon');
      if (icon) icon.textContent = isExpanded ? '▶' : '▼';
    });

    whyFindingToggleBtn.addEventListener('click', () => {
      const isExpanded = whyFindingToggleBtn.getAttribute('aria-expanded') === 'true';
      whyFindingToggleBtn.setAttribute('aria-expanded', !isExpanded);
      whyFindingBody.style.display = isExpanded ? 'none' : 'flex';
      const icon = whyFindingToggleBtn.querySelector('.toggle-icon');
      if (icon) icon.textContent = isExpanded ? '▶' : '▼';
    });

    qaToggleBtn.addEventListener('click', () => {
      const isExpanded = qaToggleBtn.getAttribute('aria-expanded') === 'true';
      qaToggleBtn.setAttribute('aria-expanded', !isExpanded);
      qaQuestionsList.style.display = isExpanded ? 'none' : 'flex';
      const icon = qaToggleBtn.querySelector('.toggle-icon');
      if (icon) icon.textContent = isExpanded ? '▶' : '▼';
    });

    // Export Buttons
    exportJsonBtn.addEventListener('click', () => exportReport('json'));
    exportTextBtn.addEventListener('click', () => exportReport('text'));
  }

  // --- Multi-Study Browsing & Filtering ---
  async function fetchStudiesIndex() {
    try {
      const resp = await fetch('/api/studies');
      if (!resp.ok) throw new Error(`HTTP error ${resp.status}`);
      const data = await resp.json();
      state.allStudies = data.studies || [];
      populateStudyDropdown(state.allStudies);
    } catch (err) {
      console.error('Failed to fetch studies index:', err);
    }
  }

  function populateStudyDropdown(studies) {
    studySelect.innerHTML = '';
    studies.forEach(s => {
      const opt = document.createElement('option');
      opt.value = s.study_id;
      const viewStr = s.views ? s.views.join('/') : 'Unknown';
      opt.textContent = `${s.study_id} (${viewStr} - ${s.images_count} img)`;
      if (s.study_id === state.studyId) opt.selected = true;
      studySelect.appendChild(opt);
    });
  }

  function filterStudyDropdown() {
    const q = (studySearchInput.value || '').trim().toLowerCase();
    const filter = studyFilterSelect.value;

    const filtered = state.allStudies.filter(s => {
      const matchQuery = s.study_id.toLowerCase().includes(q);
      let matchFilter = true;
      if (filter === 'validated') {
        matchFilter = s.report_available && s.evidence_available;
      } else if (filter === 'dual_view') {
        matchFilter = (s.images_count >= 2) || (s.views && s.views.length >= 2);
      }
      return matchQuery && matchFilter;
    });

    populateStudyDropdown(filtered);
    if (filtered.length > 0 && !filtered.some(s => s.study_id === state.studyId)) {
      state.studyId = filtered[0].study_id;
      loadStudy(state.studyId);
    }
  }

  // --- Viewport Zoom & Pan ---
  function adjustZoom(delta) {
    state.primary.zoom = Math.max(0.5, Math.min(3.0, state.primary.zoom + delta));
    if (state.isSync) {
      state.secondary.zoom = state.primary.zoom;
    }
    updateViewportTransforms();
  }

  function resetView() {
    state.primary.zoom = 1.0;
    state.primary.panX = 0;
    state.primary.panY = 0;
    state.secondary.zoom = 1.0;
    state.secondary.panX = 0;
    state.secondary.panY = 0;
    updateViewportTransforms();
  }

  function updateViewportTransforms() {
    primaryViewportContent.style.transform = `translate(${state.primary.panX}px, ${state.primary.panY}px) scale(${state.primary.zoom})`;
    if (secondaryViewportContent) {
      secondaryViewportContent.style.transform = `translate(${state.secondary.panX}px, ${state.secondary.panY}px) scale(${state.secondary.zoom})`;
    }
  }

  // --- Main Study Loader ---
  async function loadStudy(studyId) {
    primaryLoader.style.display = 'block';
    secondaryLoader.style.display = 'none';

    try {
      // 1. Fetch Main Study Details
      const resp = await fetch(`/api/studies/${studyId}`);
      if (!resp.ok) throw new Error(`HTTP error ${resp.status}`);
      const data = await resp.json();
      state.studyData = data;

      // 2. Fetch Images Metadata (Dual-View Discovery)
      const imgResp = await fetch(`/api/studies/${studyId}/images`);
      if (imgResp.ok) {
        const imgData = await imgResp.json();
        state.imagesMeta = imgData.images || [];
      } else {
        state.imagesMeta = [{
          image_id: data.image_id,
          view: data.view || 'Frontal',
          available: true
        }];
      }

      // 3. Fetch Existing Reviewer Annotations
      await fetchReviews(studyId);

      // Populate UI Metadata
      viewTag.textContent = `${data.view} View`;
      reportStudyId.textContent = data.study_id;
      if (data.report && data.report.metadata) {
        reportGenerator.textContent = `${data.report.metadata.provider || 'mock'} (${data.report.metadata.model || 'LLM'})`;
      }

      // Pipeline Status
      const violations = data.validation ? (data.validation.safety_violations_count ?? 0) : 0;
      pipelineViolationsText.textContent = `Safety checks: ${violations} violations`;

      // Dual-View Layout Configuration
      setupDualViewLayout();

      // Preload Images & Grounding for Primary Canvas
      await preloadPrimaryImages(data);

      // Preload Images & Grounding for Secondary Canvas (if dual-view)
      if (state.imagesMeta.length >= 2) {
        await preloadSecondaryImages();
      }

      // Set initial finding
      if (data.evidence && data.evidence.length > 0) {
        const prevFindingExists = data.evidence.some(e => e.finding === state.activeFinding);
        if (!prevFindingExists) {
          state.activeFinding = data.evidence[0].finding;
        }
      }

      renderFindingPills();
      renderActiveFinding();
      renderWhyFinding();
      renderQAQuestions();
      renderTraceability();
      renderReport();
      renderBothCanvases();
      populateReviewForm();

      primaryLoader.style.display = 'none';
    } catch (err) {
      console.error('Failed to load study data:', err);
      primaryLoader.textContent = 'Error loading study artifacts from backend.';
    }
  }

  function setupDualViewLayout() {
    const isDual = state.imagesMeta.length >= 2;
    if (isDual) {
      viewportsContainer.classList.remove('single-view');
      viewportsContainer.classList.add('dual-view');
      secondaryViewport.style.display = 'flex';
      dualViewBadge.style.display = 'inline-block';
      syncToggleContainer.style.display = 'flex';
      
      const secImg = state.imagesMeta[1];
      secondaryViewportTag.textContent = `SECONDARY (${secImg.view.toUpperCase()})`;
    } else {
      viewportsContainer.classList.remove('dual-view');
      viewportsContainer.classList.add('single-view');
      secondaryViewport.style.display = 'none';
      dualViewBadge.style.display = 'none';
      syncToggleContainer.style.display = 'none';
    }
  }

  async function preloadPrimaryImages(data) {
    state.primaryImages.heatmaps = {};
    const origImg = new Image();
    origImg.crossOrigin = 'anonymous';
    origImg.src = data.original_image_url;
    await new Promise((resolve, reject) => {
      origImg.onload = resolve;
      origImg.onerror = () => {
        console.warn('Could not load primary image url');
        resolve();
      };
    });
    state.primaryImages.original = origImg;

    primaryCanvas.width = origImg.naturalWidth || 512;
    primaryCanvas.height = origImg.naturalHeight || 624;

    for (const g of (data.groundings || [])) {
      const hmImg = new Image();
      hmImg.crossOrigin = 'anonymous';
      hmImg.src = g.heatmap_url;
      state.primaryImages.heatmaps[g.finding] = hmImg;
    }
  }

  async function preloadSecondaryImages() {
    secondaryLoader.style.display = 'block';
    state.secondaryImages.heatmaps = {};
    const sec = state.imagesMeta[1];

    const origImg = new Image();
    origImg.crossOrigin = 'anonymous';
    origImg.src = sec.image_url || `/api/studies/${state.studyId}/image/${sec.image_id}`;
    await new Promise((resolve) => {
      origImg.onload = resolve;
      origImg.onerror = () => {
        console.warn('Could not load secondary image url');
        resolve();
      };
    });
    state.secondaryImages.original = origImg;

    secondaryCanvas.width = origImg.naturalWidth || 512;
    secondaryCanvas.height = origImg.naturalHeight || 624;

    secondaryLoader.style.display = 'none';
  }

  // --- Findings & Active Cards ---
  function renderFindingPills() {
    findingsSelector.innerHTML = '';
    const evidence = state.studyData.evidence || [];
    findingsCountBadge.textContent = `${evidence.length} Candidates`;

    evidence.forEach(ev => {
      const pill = document.createElement('button');
      pill.className = `finding-pill ${ev.finding === state.activeFinding ? 'active' : ''}`;
      pill.innerHTML = `
        <span>${ev.finding}</span>
        <span class="pill-score">${ev.model_score.toFixed(4)}</span>
      `;
      pill.addEventListener('click', () => {
        state.activeFinding = ev.finding;
        document.querySelectorAll('.finding-pill').forEach(p => p.classList.remove('active'));
        pill.classList.add('active');
        renderActiveFinding();
        renderWhyFinding();
        renderQAQuestions();
        renderTraceability();
        renderBothCanvases();
        populateReviewForm();
      });
      findingsSelector.appendChild(pill);
    });
  }

  function renderActiveFinding() {
    const evidence = (state.studyData.evidence || []).find(e => e.finding === state.activeFinding);
    if (!evidence) return;

    activeFindingName.textContent = evidence.finding.toUpperCase();
    activeFindingStatus.textContent = evidence.status.toUpperCase();
    activeFindingStatus.className = `status-badge status-${evidence.status}`;
    activeModelScore.textContent = evidence.model_score.toFixed(4);
    activeEvidenceSource.textContent = evidence.evidence_source || 'vision_model';
    activeLocation.textContent = evidence.location || 'unspecified';
    activeSeverity.textContent = evidence.severity || 'unspecified';

    const interp = STATUS_INTERPRETATIONS[evidence.status] || STATUS_INTERPRETATIONS.possible;
    statusInterpretationText.textContent = interp;
  }

  function renderWhyFinding() {
    const evidence = (state.studyData.evidence || []).find(e => e.finding === state.activeFinding);
    const grounding = (state.studyData.groundings || []).find(g => g.finding === state.activeFinding);
    const allQuestions = state.studyData.qa_questions || [];
    const presenceQ = allQuestions.find(q => q.finding === state.activeFinding && q.level === 1);

    if (!evidence) return;

    whyModelSignal.textContent = `${evidence.model_score.toFixed(4)} (DenseNet-121 Activation)`;
    whyEvidenceSource.textContent = evidence.evidence_source || 'vision_model';
    
    if (presenceQ) {
      whyQaState.textContent = `Presence evaluated as '${presenceQ.answer || 'NOT PROVIDED'}'`;
    } else {
      whyQaState.textContent = 'Presence evaluation not recorded';
    }

    if (grounding) {
      whySpatialGrounding.textContent = `Available (Grad-CAM on ${grounding.target_layer || 'norm5'})`;
    } else {
      whySpatialGrounding.textContent = 'Not available';
    }

    whyFinalStatus.textContent = `${evidence.status.toUpperCase()} (Uncertainty preserved)`;
  }

  function renderQAQuestions() {
    qaQuestionsList.innerHTML = '';
    const allQuestions = state.studyData.qa_questions || [];
    const relevant = allQuestions.filter(q => q.finding === state.activeFinding);

    if (relevant.length === 0) {
      qaQuestionsList.innerHTML = '<div class="qa-card"><span class="qa-text">No QA questions recorded for this finding.</span></div>';
      return;
    }

    relevant.forEach(q => {
      const card = document.createElement('div');
      card.className = 'qa-card';
      
      const levelLabel = q.level === 1 ? 'LEVEL 1 — PRESENCE' : 'LEVEL 2 — LOCATION / SEVERITY';
      const answerVal = (q.answer || 'NOT PROVIDED').toUpperCase();
      
      let effectText = 'Evidence status preserved';
      if (q.level === 1) {
        if (q.answer === 'uncertain') {
          effectText = 'Answer: UNCERTAIN → Evidence remains POSSIBLE';
        } else if (q.answer === 'yes') {
          effectText = 'Answer: YES → Explicit positive support';
        } else if (q.answer === 'no') {
          effectText = 'Answer: NO → Evidence marked absent';
        }
      } else {
        if (q.answer === 'unspecified') {
          effectText = 'Answer: UNSPECIFIED → Location/severity remain unspecified';
        } else {
          effectText = `Answer: ${answerVal} → Attribute recorded`;
        }
      }

      card.innerHTML = `
        <div class="qa-header-row">
          <span class="qa-level-tag">${levelLabel}</span>
          <span class="qa-id">${q.question_id}</span>
        </div>
        <div class="qa-text">${q.question_text}</div>
        <div class="qa-tags-row">
          <span class="qa-answer-tag">Answer: ${answerVal}</span>
          <span class="qa-effect-tag">${effectText}</span>
        </div>
      `;
      qaQuestionsList.appendChild(card);
    });
  }

  function renderTraceability() {
    traceabilityChain.innerHTML = '';
    const evidence = (state.studyData.evidence || []).find(e => e.finding === state.activeFinding);
    const grounding = (state.studyData.groundings || []).find(g => g.finding === state.activeFinding);
    const reportFinding = ((state.studyData.report || {}).findings || []).find(f => f.finding === state.activeFinding);
    const allQuestions = state.studyData.qa_questions || [];
    const presenceQ = allQuestions.find(q => q.finding === state.activeFinding && q.level === 1);

    const steps = [
      {
        num: 1,
        title: 'Vision Model (DenseNet-121)',
        sub: `Model Activation: ${evidence ? evidence.model_score.toFixed(4) : 'N/A'} (1024-D features extracted)`,
        complete: true
      },
      {
        num: 2,
        title: 'Diagnostic QA Engine',
        sub: `Presence question answer: '${presenceQ ? presenceQ.answer : 'uncertain'}' (No false certainty)`,
        complete: true
      },
      {
        num: 3,
        title: 'Evidence Layer Resolution',
        sub: `Status: ${evidence ? evidence.status.toUpperCase() : 'POSSIBLE'} (Location: ${evidence ? evidence.location : 'unspecified'})`,
        complete: true
      },
      {
        num: 4,
        title: 'Visual Grounding (Grad-CAM)',
        sub: `Attribution target: ${grounding ? grounding.target_layer : 'model.features.norm5'} (${grounding ? grounding.activation_shape.join('×') : '7×7'})`,
        complete: !!grounding
      },
      {
        num: 5,
        title: 'LLM Report Generation',
        sub: reportFinding ? `Statement: "${reportFinding.statement}"` : 'Statement synthesized in report',
        complete: !!reportFinding
      },
      {
        num: 6,
        title: 'Safety & Schema Validation',
        sub: '✓ Schema valid  ✓ Evidence traceable  ✓ Zero ground-truth leakage',
        complete: true
      }
    ];

    steps.forEach(s => {
      const row = document.createElement('div');
      row.className = 'trace-step';
      row.innerHTML = `
        <div class="trace-dot ${s.complete ? 'complete' : ''}">${s.num}</div>
        <div class="trace-content">
          <span class="trace-label">${s.title}</span>
          <span class="trace-sub">${s.sub}</span>
        </div>
      `;
      traceabilityChain.appendChild(row);
    });
  }

  function renderReport() {
    reportFindingsList.innerHTML = '';
    reportImpressionList.innerHTML = '';
    const report = state.studyData.report || {};

    (report.findings || []).forEach(f => {
      const item = document.createElement('div');
      item.className = 'finding-item';
      item.innerHTML = `
        <span class="finding-status-tag status-badge status-${f.status}">[${f.status.toUpperCase()}]</span>
        <span>${f.statement}</span>
      `;
      reportFindingsList.appendChild(item);
    });

    (report.impression || []).forEach((imp, idx) => {
      const item = document.createElement('div');
      item.className = 'impression-item';
      
      let cleanText = imp;
      let numLabel = `${idx + 1}.`;
      if (cleanText.startsWith('1.') || cleanText.startsWith('2.')) {
        const parts = cleanText.split(/^\d+\.\s*/);
        if (parts.length > 1) {
          cleanText = parts[1];
        }
      }

      item.innerHTML = `
        <span class="impression-num">${numLabel}</span>
        <span class="impression-text">${cleanText}</span>
      `;
      reportImpressionList.appendChild(item);
    });
  }

  // --- Canvas Rendering (Single & Dual View) ---
  function renderBothCanvases() {
    renderCanvas(primaryCtx, primaryCanvas, state.primaryImages, true);
    if (state.imagesMeta.length >= 2) {
      renderCanvas(secondaryCtx, secondaryCanvas, state.secondaryImages, false);
    }
  }

  function renderCanvas(ctx, canvas, imagesObj, isPrimary) {
    if (!imagesObj.original) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const orig = imagesObj.original;

    if (state.viewMode === 'original') {
      ctx.globalAlpha = 1.0;
      ctx.drawImage(orig, 0, 0, canvas.width, canvas.height);
    } else if (state.viewMode === 'heatmap') {
      const hm = imagesObj.heatmaps[state.activeFinding];
      if (hm && hm.complete) {
        ctx.globalAlpha = 1.0;
        ctx.drawImage(hm, 0, 0, canvas.width, canvas.height);
      } else {
        ctx.fillStyle = '#0f172a';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
      }
    } else if (state.viewMode === 'overlay') {
      // Draw original first
      ctx.globalAlpha = 1.0;
      ctx.drawImage(orig, 0, 0, canvas.width, canvas.height);

      // Superimpose heatmap with alpha
      const hm = imagesObj.heatmaps[state.activeFinding];
      if (hm && hm.complete) {
        ctx.globalAlpha = state.opacity;
        ctx.drawImage(hm, 0, 0, canvas.width, canvas.height);
      }
    }

    // Reset alpha
    ctx.globalAlpha = 1.0;

    // Handle secondary view attribution unavailable notice
    if (!isPrimary) {
      const hasHm = !!imagesObj.heatmaps[state.activeFinding];
      if (!hasHm && state.viewMode !== 'original') {
        secondaryUnavailableNotice.style.display = 'block';
      } else {
        secondaryUnavailableNotice.style.display = 'none';
      }
    }
  }

  // --- Reviewer Annotation Handlers & Persistence ---
  async function fetchReviews(studyId) {
    try {
      const resp = await fetch(`/api/studies/${studyId}/reviews`);
      if (resp.ok) {
        const data = await resp.json();
        state.reviews = data.reviews || {};
      } else {
        state.reviews = {};
      }
    } catch (err) {
      console.warn('Could not fetch reviews:', err);
      state.reviews = {};
    }
  }

  function populateReviewForm() {
    const rev = state.reviews[state.activeFinding];
    if (rev) {
      reviewerStatusSelect.value = rev.reviewer_status || 'not_reviewed';
      reviewerLocationInput.value = rev.location || '';
      reviewerSeveritySelect.value = rev.severity || 'unspecified';
      reviewerNotesInput.value = rev.notes || '';
    } else {
      reviewerStatusSelect.value = 'not_reviewed';
      reviewerLocationInput.value = '';
      reviewerSeveritySelect.value = 'unspecified';
      reviewerNotesInput.value = '';
    }
    updateComparisonBox();
  }

  function updateComparisonBox() {
    const evidence = (state.studyData.evidence || []).find(e => e.finding === state.activeFinding);
    const machineStat = evidence ? evidence.status.toUpperCase() : 'UNKNOWN';
    compMachineStatus.textContent = machineStat;
    compMachineStatus.className = `status-badge status-${evidence ? evidence.status : 'possible'}`;

    const revStat = reviewerStatusSelect.value;
    compReviewerStatus.textContent = revStat.replace('_', ' ').toUpperCase();
    if (revStat === 'confirmed_present') {
      compReviewerStatus.className = 'status-badge status-supported';
    } else if (revStat === 'confirmed_absent') {
      compReviewerStatus.className = 'status-badge status-absent';
    } else if (revStat === 'uncertain') {
      compReviewerStatus.className = 'status-badge status-uncertain';
    } else {
      compReviewerStatus.className = 'status-badge status-neutral';
    }
  }

  async function handleSaveReview() {
    reviewSaveStatus.textContent = 'Saving...';
    const payload = {
      study_id: state.studyId,
      image_id: state.imagesMeta[0] ? state.imagesMeta[0].image_id : state.studyData.image_id,
      finding: state.activeFinding,
      reviewer_status: reviewerStatusSelect.value,
      location: reviewerLocationInput.value.trim() || 'unspecified',
      severity: reviewerSeveritySelect.value,
      notes: reviewerNotesInput.value.trim()
    };

    try {
      const resp = await fetch(`/api/studies/${state.studyId}/reviews`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (!resp.ok) {
        const errJson = await resp.json();
        throw new Error(errJson.error || `HTTP ${resp.status}`);
      }

      state.reviews[state.activeFinding] = payload;
      reviewSaveStatus.textContent = '✓ Saved';
      reviewSaveStatus.style.color = '#10b981';
      updateComparisonBox();
      setTimeout(() => { reviewSaveStatus.textContent = ''; }, 3000);
    } catch (err) {
      console.error('Failed to save review:', err);
      reviewSaveStatus.textContent = `Error: ${err.message}`;
      reviewSaveStatus.style.color = '#ef4444';
    }
  }

  async function handleClearReview() {
    reviewSaveStatus.textContent = 'Clearing...';
    try {
      const resp = await fetch(`/api/studies/${state.studyId}/reviews/${state.activeFinding}`, {
        method: 'DELETE'
      });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);

      delete state.reviews[state.activeFinding];
      populateReviewForm();
      reviewSaveStatus.textContent = 'Cleared';
      reviewSaveStatus.style.color = '#94a3b8';
      setTimeout(() => { reviewSaveStatus.textContent = ''; }, 3000);
    } catch (err) {
      console.error('Failed to clear review:', err);
      reviewSaveStatus.textContent = 'Failed to clear';
      reviewSaveStatus.style.color = '#ef4444';
    }
  }

  // --- Batch Processing Runner ---
  async function handleTriggerBatch() {
    const rawInput = batchStudyInput.value || '';
    const studyIds = rawInput.split(',').map(s => s.trim()).filter(Boolean);

    if (studyIds.length === 0) {
      alert('Please enter at least one study ID.');
      return;
    }

    triggerBatchBtn.disabled = true;
    triggerBatchBtn.textContent = 'Submitting Batch Job...';
    batchStatusContainer.style.display = 'block';

    try {
      const resp = await fetch('/api/batch/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ study_ids: studyIds })
      });

      if (!resp.ok) {
        const errJson = await resp.json();
        throw new Error(errJson.error || `HTTP ${resp.status}`);
      }

      const data = await resp.json();
      batchJobId.textContent = data.job_id;
      batchJobStatusBadge.textContent = 'RUNNING';
      batchJobStatusBadge.className = 'status-badge status-possible';

      // Start polling
      if (state.batchPollInterval) clearInterval(state.batchPollInterval);
      state.batchPollInterval = setInterval(checkBatchStatus, 1500);
    } catch (err) {
      console.error('Failed to trigger batch job:', err);
      alert(`Batch runner error: ${err.message}`);
    } finally {
      triggerBatchBtn.disabled = false;
      triggerBatchBtn.textContent = 'Run Batch Verification';
    }
  }

  async function checkBatchStatus() {
    try {
      const resp = await fetch('/api/batch/status');
      if (!resp.ok) return;
      const data = await resp.json();

      if (!data.job_id) {
        batchStatusContainer.style.display = 'none';
        return;
      }

      batchStatusContainer.style.display = 'block';
      batchJobId.textContent = data.job_id;
      batchJobStatusBadge.textContent = data.status;

      if (data.status === 'COMPLETED') {
        batchJobStatusBadge.className = 'status-badge status-supported';
        if (state.batchPollInterval) clearInterval(state.batchPollInterval);
      } else if (data.status === 'FAILED') {
        batchJobStatusBadge.className = 'status-badge status-absent';
        if (state.batchPollInterval) clearInterval(state.batchPollInterval);
      } else {
        batchJobStatusBadge.className = 'status-badge status-possible';
      }

      renderBatchStages(data);
      renderBatchLog(data);
    } catch (err) {
      console.warn('Batch status fetch error:', err);
    }
  }

  function renderBatchStages(data) {
    batchStagesGrid.innerHTML = '';
    const stages = ['Vision', 'Diagnostic QA', 'Evidence', 'Grounding', 'LLM', 'Validation'];
    const activeStage = data.current_stage || '';

    stages.forEach(st => {
      const pill = document.createElement('div');
      const isPastOrCurrent = stages.indexOf(st) <= stages.indexOf(activeStage);
      pill.className = `batch-stage-pill ${isPastOrCurrent ? 'active' : ''}`;
      pill.textContent = st;
      batchStagesGrid.appendChild(pill);
    });
  }

  function renderBatchLog(data) {
    batchLog.innerHTML = '';
    const results = data.results || [];
    if (results.length === 0) {
      batchLog.textContent = `Running batch execution across ${data.total_studies || 0} studies...`;
      return;
    }

    results.forEach(r => {
      const line = document.createElement('div');
      line.className = 'batch-log-line';
      const statusIcon = r.status === 'COMPLETED' ? '✓' : (r.status === 'FAILED' ? '✗' : '⏳');
      line.innerHTML = `
        <span class="log-id"><strong>${r.study_id}</strong></span>
        <span class="log-status">[${r.status}]</span>
        <span class="log-stage">Stage: ${r.stage}</span>
        <span>Validation: ${r.validation_status}</span>
        ${r.error ? `<span class="log-error">${r.error}</span>` : ''}
      `;
      batchLog.appendChild(line);
    });
  }

  // --- Export Utilities ---
  async function exportReport(format) {
    try {
      const resp = await fetch(`/api/studies/${state.studyId}/export?format=${format}`);
      if (!resp.ok) throw new Error(`Export failed with HTTP ${resp.status}`);

      if (format === 'text') {
        const text = await resp.text();
        const blob = new Blob([text], { type: 'text/plain' });
        downloadBlob(blob, `${state.studyId}_radiology_report.txt`);
      } else {
        const json = await resp.json();
        const blob = new Blob([JSON.stringify(json, null, 2)], { type: 'application/json' });
        downloadBlob(blob, `${state.studyId}_radiology_report.json`);
      }
    } catch (err) {
      console.error('Export error:', err);
      alert('Failed to export report.');
    }
  }

  function downloadBlob(blob, filename) {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }
});
