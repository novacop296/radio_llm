/**
 * LLM-Assisted Explainable Radiology Dashboard Application Logic
 * Phase 1.2 — Human-in-the-Loop Review, Report Correction & Finalization
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

    // Review Session (Phase 1.2)
    reviewSession: null,
    selectedReviewStatus: 'not_reviewed',
    activeReviewerId: 'researcher_01',

    // Multi-Reviewer Consensus (Phase 1.3)
    activeTab: 'individual', // 'individual' or 'consensus'
    consensusSession: null,
    pendingAdjudicationItem: null,

    // Research Evaluation & Experiment Tracking (Phase 1.5)
    selectedExperimentIds: new Set(),
    activeExperimentId: null,
    experimentsList: [],
    snapshotsList: [],

    // Batch polling & race-condition protection
    batchPollInterval: null,
    activeBatchJobId: null,
    requestSeq: 0
  };

  // Status Interpretation Semantic Map
  const STATUS_INTERPRETATIONS = {
    supported: 'Sufficient explicit evidence supports this finding.',
    possible: 'Model or indirect evidence suggests this finding may warrant review, but it is not independently established.',
    uncertain: 'Available evidence is equivocal or insufficient to establish presence or absence.',
    absent: 'Available evidence supports absence of this finding.'
  };

  // DOM Elements - Header & Navigation
  const studySelect = document.getElementById('studySelect');
  const studySearchInput = document.getElementById('studySearchInput');
  const studyFilterSelect = document.getElementById('studyFilterSelect');
  const batchModalBtn = document.getElementById('batchModalBtn');
  const viewTag = document.getElementById('viewTag');
  const dualViewBadge = document.getElementById('dualViewBadge');
  const syncToggleContainer = document.getElementById('syncToggleContainer');
  const syncToggle = document.getElementById('syncToggle');
  const pipelineViolationsText = document.getElementById('pipelineViolationsText');
  const reviewerNameBadge = document.getElementById('reviewerNameBadge');
  const activeReviewerSelect = document.getElementById('activeReviewerSelect');

  // DOM Elements - Top Tabs & Containers (Phase 1.4 & 1.5)
  const tabDatasetQueue = document.getElementById('tabDatasetQueue');
  const tabIndividualReview = document.getElementById('tabIndividualReview');
  const tabConsensusDashboard = document.getElementById('tabConsensusDashboard');
  const tabEvaluationAnalytics = document.getElementById('tabEvaluationAnalytics');
  const tabResearchEvaluation = document.getElementById('tabResearchEvaluation');

  const datasetQueueViewContainer = document.getElementById('datasetQueueViewContainer');
  const individualViewContainer = document.getElementById('individualViewContainer');
  const individualProgressBar = document.getElementById('individualProgressBar');
  const consensusDashboardContainer = document.getElementById('consensusDashboardContainer');
  const evaluationAnalyticsContainer = document.getElementById('evaluationAnalyticsContainer');
  const researchEvaluationViewContainer = document.getElementById('researchEvaluationViewContainer');
  const consensusStatusHeaderBadge = document.getElementById('consensusStatusHeaderBadge');

  // DOM Elements - Dataset & Review Queue (Phase 1.4)
  const statTotalStudies = document.getElementById('statTotalStudies');
  const statUnreviewed = document.getElementById('statUnreviewed');
  const statInReview = document.getElementById('statInReview');
  const statAwaitingConsensus = document.getElementById('statAwaitingConsensus');
  const statAdjudicationRequired = document.getElementById('statAdjudicationRequired');
  const statFinalized = document.getElementById('statFinalized');

  const queueSearchInput = document.getElementById('queueSearchInput');
  const queueStatusFilter = document.getElementById('queueStatusFilter');
  const queueConsensusFilter = document.getElementById('queueConsensusFilter');
  const queueSortBy = document.getElementById('queueSortBy');
  const refreshQueueBtn = document.getElementById('refreshQueueBtn');
  const exportDatasetJsonBtn = document.getElementById('exportDatasetJsonBtn');
  const exportDatasetTextBtn = document.getElementById('exportDatasetTextBtn');
  const queueTableBody = document.getElementById('queueTableBody');

  // DOM Elements - Evaluation Analytics (Phase 1.4)
  const analyticsReviewCompletion = document.getElementById('analyticsReviewCompletion');
  const analyticsReviewCompletionSub = document.getElementById('analyticsReviewCompletionSub');
  const analyticsActiveReviewers = document.getElementById('analyticsActiveReviewers');
  const analyticsActiveReviewersSub = document.getElementById('analyticsActiveReviewersSub');
  const analyticsUnanimousFindings = document.getElementById('analyticsUnanimousFindings');
  const analyticsFinalizedReports = document.getElementById('analyticsFinalizedReports');
  const analyticsCohenKappa = document.getElementById('analyticsCohenKappa');
  const analyticsCohenInterp = document.getElementById('analyticsCohenInterp');
  const analyticsCohenMeta = document.getElementById('analyticsCohenMeta');
  const analyticsFleissKappa = document.getElementById('analyticsFleissKappa');
  const analyticsFleissInterp = document.getElementById('analyticsFleissInterp');
  const analyticsFleissMeta = document.getElementById('analyticsFleissMeta');
  const reviewerActivityTableBody = document.getElementById('reviewerActivityTableBody');
  const provenanceStudySelect = document.getElementById('provenanceStudySelect');
  const loadProvenanceBtn = document.getElementById('loadProvenanceBtn');
  const provenanceTimeline = document.getElementById('provenanceTimeline');

  // DOM Elements - Research Evaluation & Experiment Tracking (Phase 1.5)
  const openNewExperimentModalBtn = document.getElementById('openNewExperimentModalBtn');
  const openNewSnapshotModalBtn = document.getElementById('openNewSnapshotModalBtn');
  const compareSelectedExperimentsBtn = document.getElementById('compareSelectedExperimentsBtn');
  const selectedExpCount = document.getElementById('selectedExpCount');
  const refreshExperimentsBtn = document.getElementById('refreshExperimentsBtn');
  const snapshotTotalCount = document.getElementById('snapshotTotalCount');
  const snapshotsGridContainer = document.getElementById('snapshotsGridContainer');
  const experimentsTotalCount = document.getElementById('experimentsTotalCount');
  const selectAllExperimentsCheckbox = document.getElementById('selectAllExperimentsCheckbox');
  const experimentsTableBody = document.getElementById('experimentsTableBody');

  const experimentDetailSection = document.getElementById('experimentDetailSection');
  const detailExpId = document.getElementById('detailExpId');
  const detailExpName = document.getElementById('detailExpName');
  const detailExpDesc = document.getElementById('detailExpDesc');
  const detailExpStatusBadge = document.getElementById('detailExpStatusBadge');
  const detailExpSnapshotId = document.getElementById('detailExpSnapshotId');
  const detailExpCreatedBy = document.getElementById('detailExpCreatedBy');
  const detailExpCreatedAt = document.getElementById('detailExpCreatedAt');
  const detailExpCompletedAt = document.getElementById('detailExpCompletedAt');
  const detailExpModelName = document.getElementById('detailExpModelName');
  const detailExpModelVersion = document.getElementById('detailExpModelVersion');
  const detailExpTargetLayer = document.getElementById('detailExpTargetLayer');
  const detailExpQAParams = document.getElementById('detailExpQAParams');
  const detailExpLLMParams = document.getElementById('detailExpLLMParams');
  const detailExpFingerprintHash = document.getElementById('detailExpFingerprintHash');
  const exportExpJsonBtn = document.getElementById('exportExpJsonBtn');
  const exportExpTextBtn = document.getElementById('exportExpTextBtn');

  const expMetricReviewCov = document.getElementById('expMetricReviewCov');
  const expMetricReviewSub = document.getElementById('expMetricReviewSub');
  const expMetricConsensusCov = document.getElementById('expMetricConsensusCov');
  const expMetricConsensusSub = document.getElementById('expMetricConsensusSub');
  const expMetricMachineAggr = document.getElementById('expMetricMachineAggr');
  const expMetricKappa = document.getElementById('expMetricKappa');
  const expMetricKappaSub = document.getElementById('expMetricKappaSub');
  const expMetricExplainCov = document.getElementById('expMetricExplainCov');
  const expMetricExplainSub = document.getElementById('expMetricExplainSub');
  const expMetricHumanCorr = document.getElementById('expMetricHumanCorr');
  const expMetricHumanCorrSub = document.getElementById('expMetricHumanCorrSub');
  const expProvenanceTimeline = document.getElementById('expProvenanceTimeline');

  const experimentComparisonSection = document.getElementById('experimentComparisonSection');
  const closeComparisonBtn = document.getElementById('closeComparisonBtn');
  const comparisonConfigHeaderRow = document.getElementById('comparisonConfigHeaderRow');
  const comparisonConfigTableBody = document.getElementById('comparisonConfigTableBody');
  const comparisonMetricsHeaderRow = document.getElementById('comparisonMetricsHeaderRow');
  const comparisonMetricsTableBody = document.getElementById('comparisonMetricsTableBody');

  // Phase 1.5 Modals
  const createExperimentModalBackdrop = document.getElementById('createExperimentModalBackdrop');
  const closeCreateExpModalBtn = document.getElementById('closeCreateExpModalBtn');
  const cancelCreateExpModalBtn = document.getElementById('cancelCreateExpModalBtn');
  const submitCreateExperimentBtn = document.getElementById('submitCreateExperimentBtn');
  const createExpFeedback = document.getElementById('createExpFeedback');
  const newExpName = document.getElementById('newExpName');
  const newExpDescription = document.getElementById('newExpDescription');
  const newExpSnapshotSelect = document.getElementById('newExpSnapshotSelect');
  const newExpModelVersion = document.getElementById('newExpModelVersion');
  const newExpTargetLayer = document.getElementById('newExpTargetLayer');
  const newExpQaThreshold = document.getElementById('newExpQaThreshold');
  const newExpQaTopK = document.getElementById('newExpQaTopK');
  const newExpCreatedBy = document.getElementById('newExpCreatedBy');
  const expFingerprintPreview = document.getElementById('expFingerprintPreview');

  const createSnapshotModalBackdrop = document.getElementById('createSnapshotModalBackdrop');
  const closeCreateSnapModalBtn = document.getElementById('closeCreateSnapModalBtn');
  const cancelCreateSnapModalBtn = document.getElementById('cancelCreateSnapModalBtn');
  const submitCreateSnapshotBtn = document.getElementById('submitCreateSnapshotBtn');
  const createSnapFeedback = document.getElementById('createSnapFeedback');
  const newSnapName = document.getElementById('newSnapName');
  const newSnapDescription = document.getElementById('newSnapDescription');
  const newSnapStudySelection = document.getElementById('newSnapStudySelection');
  const newSnapCreatedBy = document.getElementById('newSnapCreatedBy');

  // DOM Elements - Consensus Dashboard
  const consensusStatusBadge = document.getElementById('consensusStatusBadge');
  const consensusStudyBadge = document.getElementById('consensusStudyBadge');
  const consensusIdBadge = document.getElementById('consensusIdBadge');
  const consensusReviewersCountBadge = document.getElementById('consensusReviewersCountBadge');
  const reviewerRosterList = document.getElementById('reviewerRosterList');
  const rosterCountBadge = document.getElementById('rosterCountBadge');
  const registerReviewerBtn = document.getElementById('registerReviewerBtn');
  const refreshConsensusBtn = document.getElementById('refreshConsensusBtn');
  const metricAvgAgreement = document.getElementById('metricAvgAgreement');
  const metricAvgAgreementSub = document.getElementById('metricAvgAgreementSub');
  const metricCohensKappa = document.getElementById('metricCohensKappa');
  const metricCohensKappaSub = document.getElementById('metricCohensKappaSub');
  const metricFleissKappa = document.getElementById('metricFleissKappa');
  const metricFleissKappaSub = document.getElementById('metricFleissKappaSub');
  const kappaInterpretationBadge = document.getElementById('kappaInterpretationBadge');
  const kappaDetailsText = document.getElementById('kappaDetailsText');
  const consensusTableBody = document.getElementById('consensusTableBody');
  const consensusFindingsBadge = document.getElementById('consensusFindingsBadge');
  const consensusQaGrid = document.getElementById('consensusQaGrid');
  const pendingAdjudicationsBadge = document.getElementById('pendingAdjudicationsBadge');
  const pendingAdjudicationItemsList = document.getElementById('pendingAdjudicationItemsList');
  const adjudicationRecordsBody = document.getElementById('adjudicationRecordsBody');
  const consensusReportFinalBadge = document.getElementById('consensusReportFinalBadge');
  const consensusFindingsList = document.getElementById('consensusFindingsList');
  const consensusImpressionEditor = document.getElementById('consensusImpressionEditor');
  const consensusReviewerSummary = document.getElementById('consensusReviewerSummary');
  const consensusAgreementSummary = document.getElementById('consensusAgreementSummary');
  const consensusCommentaryEditor = document.getElementById('consensusCommentaryEditor');
  const resetConsensusDraftBtn = document.getElementById('resetConsensusDraftBtn');
  const saveConsensusDraftBtn = document.getElementById('saveConsensusDraftBtn');
  const finalizeConsensusBtn = document.getElementById('finalizeConsensusBtn');
  const consensusActionFeedback = document.getElementById('consensusActionFeedback');
  const exportConsensusJsonBtn = document.getElementById('exportConsensusJsonBtn');
  const exportConsensusTextBtn = document.getElementById('exportConsensusTextBtn');

  // DOM Elements - Adjudication & Register Modals
  const adjudicationModalBackdrop = document.getElementById('adjudicationModalBackdrop');
  const adjudicationModalTitle = document.getElementById('adjudicationModalTitle');
  const adjudicationModalDesc = document.getElementById('adjudicationModalDesc');
  const adjudicationTargetInfo = document.getElementById('adjudicationTargetInfo');
  const adjudicationDecisionSelect = document.getElementById('adjudicationDecisionSelect');
  const adjudicatorIdInput = document.getElementById('adjudicatorIdInput');
  const adjudicationReasonInput = document.getElementById('adjudicationReasonInput');
  const submitAdjudicationBtn = document.getElementById('submitAdjudicationBtn');
  const cancelAdjudicationBtn = document.getElementById('cancelAdjudicationBtn');
  const closeAdjudicationModalBtn = document.getElementById('closeAdjudicationModalBtn');
  const adjudicationFeedback = document.getElementById('adjudicationFeedback');

  const registerReviewerModalBackdrop = document.getElementById('registerReviewerModalBackdrop');
  const newReviewerId = document.getElementById('newReviewerId');
  const newReviewerName = document.getElementById('newReviewerName');
  const newReviewerRole = document.getElementById('newReviewerRole');
  const submitRegisterReviewerBtn = document.getElementById('submitRegisterReviewerBtn');
  const cancelRegisterModalBtn = document.getElementById('cancelRegisterModalBtn');
  const closeRegisterModalBtn = document.getElementById('closeRegisterModalBtn');
  const registerFeedback = document.getElementById('registerFeedback');

  // DOM Elements - Review Progress Section
  const stepEvidence = document.getElementById('stepEvidence');
  const stepFindings = document.getElementById('stepFindings');
  const stepReport = document.getElementById('stepReport');
  const stepValidation = document.getElementById('stepValidation');
  const stepFinalize = document.getElementById('stepFinalize');
  const findingsProgressText = document.getElementById('findingsProgressText');
  const reportProgressText = document.getElementById('reportProgressText');
  const validationProgressText = document.getElementById('validationProgressText');
  const sessionStatusBadge = document.getElementById('sessionStatusBadge');
  const validateReviewBtn = document.getElementById('validateReviewBtn');
  const finalizeReviewBtn = document.getElementById('finalizeReviewBtn');

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

  // DOM Elements - Reviewer Finding Decision
  const findingReviewStatePill = document.getElementById('findingReviewStatePill');
  const btnConfirmPresent = document.getElementById('btnConfirmPresent');
  const btnConfirmAbsent = document.getElementById('btnConfirmAbsent');
  const btnUncertain = document.getElementById('btnUncertain');
  const btnNeedsReview = document.getElementById('btnNeedsReview');
  const decisionBtns = [btnConfirmPresent, btnConfirmAbsent, btnUncertain, btnNeedsReview];
  const reviewerLocationInput = document.getElementById('reviewerLocationInput');
  const reviewerSeveritySelect = document.getElementById('reviewerSeveritySelect');
  const reviewerNotesInput = document.getElementById('reviewerNotesInput');
  const saveFindingReviewBtn = document.getElementById('saveFindingReviewBtn');
  const reviewSaveStatus = document.getElementById('reviewSaveStatus');
  const compMachineStatus = document.getElementById('compMachineStatus');
  const compReviewerStatus = document.getElementById('compReviewerStatus');
  const disagreementBanner = document.getElementById('disagreementBanner');

  // DOM Elements - Diagnostic QA & Traceability
  const qaSection = document.getElementById('qaSection');
  const qaToggleBtn = document.getElementById('qaToggleBtn');
  const qaQuestionsList = document.getElementById('qaQuestionsList');
  const whyFindingSection = document.getElementById('whyFindingSection');
  const whyFindingToggleBtn = document.getElementById('whyFindingToggleBtn');
  const whyFindingBody = document.getElementById('whyFindingBody');
  const whyModelSignal = document.getElementById('whyModelSignal');
  const whyEvidenceSource = document.getElementById('whyEvidenceSource');
  const whyQaState = document.getElementById('whyQaState');
  const whySpatialGrounding = document.getElementById('whySpatialGrounding');
  const whyFinalStatus = document.getElementById('whyFinalStatus');
  const traceabilityChain = document.getElementById('traceabilityChain');

  // DOM Elements - Report Review & Finalization
  const reportStudyId = document.getElementById('reportStudyId');
  const reportStatusBadge = document.getElementById('reportStatusBadge');
  const reportReviewId = document.getElementById('reportReviewId');
  const reportWarningHeading = document.getElementById('reportWarningHeading');
  const reportWarningSub = document.getElementById('reportWarningSub');
  const reportFindingsEditorList = document.getElementById('reportFindingsEditorList');
  const reportImpressionEditor = document.getElementById('reportImpressionEditor');
  const reportReviewerComment = document.getElementById('reportReviewerComment');
  const resetReportBtn = document.getElementById('resetReportBtn');
  const saveDraftBtn = document.getElementById('saveDraftBtn');
  const validateReportBtn = document.getElementById('validateReportBtn');
  const finalizeReportBtn = document.getElementById('finalizeReportBtn');
  const reportActionFeedback = document.getElementById('reportActionFeedback');
  const exportJsonBtn = document.getElementById('exportJsonBtn');
  const exportTextBtn = document.getElementById('exportTextBtn');

  // DOM Elements - Comparison Metrics
  const metricsReviewRate = document.getElementById('metricsReviewRate');
  const metricTotal = document.getElementById('metricTotal');
  const metricReviewed = document.getElementById('metricReviewed');
  const metricNotReviewed = document.getElementById('metricNotReviewed');
  const metricConfirmedPresent = document.getElementById('metricConfirmedPresent');
  const metricConfirmedAbsent = document.getElementById('metricConfirmedAbsent');
  const metricUncertain = document.getElementById('metricUncertain');
  const metricNeedsReview = document.getElementById('metricNeedsReview');
  const metricDisagreements = document.getElementById('metricDisagreements');

  // DOM Elements - Audit Trail
  const auditToggleBtn = document.getElementById('auditToggleBtn');
  const auditTrailBody = document.getElementById('auditTrailBody');
  const auditCountBadge = document.getElementById('auditCountBadge');
  const auditTableBody = document.getElementById('auditTableBody');

  // DOM Elements - Single Study Inference & Machine Banner
  const inferenceNotRunBanner = document.getElementById('inferenceNotRunBanner');
  const runStudyInferenceBtn = document.getElementById('runStudyInferenceBtn');
  const singleInferenceFeedback = document.getElementById('singleInferenceFeedback');
  const activeFindingCard = document.getElementById('activeFindingCard');
  const reviewerDecisionCard = document.getElementById('reviewerDecisionCard');

  // DOM Elements - Batch Modal
  const batchModalBackdrop = document.getElementById('batchModalBackdrop');
  const closeBatchModalBtn = document.getElementById('closeBatchModalBtn');
  const batchStudyInput = document.getElementById('batchStudyInput');
  const batchForceReprocess = document.getElementById('batchForceReprocess');
  const triggerBatchBtn = document.getElementById('triggerBatchBtn');
  const batchStatusContainer = document.getElementById('batchStatusContainer');
  const batchJobId = document.getElementById('batchJobId');
  const batchJobStatusBadge = document.getElementById('batchJobStatusBadge');
  const batchProgressBar = document.getElementById('batchProgressBar');
  const batchProgressText = document.getElementById('batchProgressText');
  const batchStagesGrid = document.getElementById('batchStagesGrid');
  const batchStudiesTableBody = document.getElementById('batchStudiesTableBody');
  const batchLog = document.getElementById('batchLog');

  // =========================================================================
  // Centralized Top-Level Tab Navigation System (All 9 Workspaces)
  // =========================================================================

  const TABS = {
    dataset: {
      btnId: 'tabDatasetQueue',
      containerId: 'datasetQueueViewContainer',
      extraContainerIds: [],
      onOpen: () => loadDatasetQueue()
    },
    individual: {
      btnId: 'tabIndividualReview',
      containerId: 'individualViewContainer',
      extraContainerIds: ['individualProgressBar'],
      onOpen: () => {
        if (state.reviewSession) renderReviewSession(state.reviewSession);
      }
    },
    consensus: {
      btnId: 'tabConsensusDashboard',
      containerId: 'consensusDashboardContainer',
      extraContainerIds: [],
      onOpen: () => fetchAndRenderConsensus(state.studyId)
    },
    analytics: {
      btnId: 'tabEvaluationAnalytics',
      containerId: 'evaluationAnalyticsContainer',
      extraContainerIds: [],
      onOpen: () => loadEvaluationAnalytics()
    },
    research_eval: {
      btnId: 'tabResearchEvaluation',
      containerId: 'researchEvaluationViewContainer',
      extraContainerIds: [],
      onOpen: () => loadResearchEvaluation()
    },
    registry: {
      btnId: 'tabExperimentRegistry',
      containerId: 'experimentRegistryViewContainer',
      extraContainerIds: [],
      onOpen: () => loadExperimentRegistry()
    },
    benchmarking: {
      btnId: 'tabExternalBenchmarking',
      containerId: 'externalBenchmarkingViewContainer',
      extraContainerIds: [],
      onOpen: () => loadExternalBenchmarkingDashboard()
    },
    experiments: {
      btnId: 'tabResearchExperiments',
      containerId: 'researchExperimentsContainer',
      extraContainerIds: [],
      onOpen: () => loadPhase19Dashboard()
    },
    counterfactual: {
      btnId: 'tabCounterfactualExplainability',
      containerId: 'counterfactualViewContainer',
      extraContainerIds: [],
      onOpen: () => {
        loadPhase20Dashboard();
        initPhase20Workspace();
      }
    }
  };

  const TAB_ALIASES = {
    'dataset': 'dataset',
    'queue': 'dataset',
    'dataset_queue': 'dataset',
    'individual': 'individual',
    'review': 'individual',
    'consensus': 'consensus',
    'adjudication': 'consensus',
    'analytics': 'analytics',
    'workflow_analytics': 'analytics',
    'research_eval': 'research_eval',
    'evaluation': 'research_eval',
    'registry': 'registry',
    'experiment_registry': 'registry',
    'benchmarking': 'benchmarking',
    'external_benchmarking': 'benchmarking',
    'external_benchmark': 'benchmarking',
    'experiments': 'experiments',
    'research_experiments': 'experiments',
    'phase19': 'experiments',
    'counterfactual': 'counterfactual',
    'counterfactual_explainability': 'counterfactual',
    'phase20': 'counterfactual'
  };

  function switchTab(tabName) {
    const canonicalKey = TAB_ALIASES[tabName] || (TABS[tabName] ? tabName : 'individual');
    state.activeTab = canonicalKey;

    // 1. Remove active state from all top-level tab buttons
    document.querySelectorAll('.main-tab-nav .main-tab-btn').forEach(btn => {
      btn.classList.remove('active');
    });

    // 2. Hide all workspace containers and extra containers
    const allContainerIds = new Set();
    Object.values(TABS).forEach(cfg => {
      if (cfg.containerId) allContainerIds.add(cfg.containerId);
      if (cfg.extraContainerIds) cfg.extraContainerIds.forEach(id => allContainerIds.add(id));
    });

    allContainerIds.forEach(id => {
      const el = document.getElementById(id);
      if (el) el.style.display = 'none';
    });

    // 3. Mark selected tab button active and display corresponding workspace
    const activeConfig = TABS[canonicalKey];
    if (activeConfig) {
      if (activeConfig.btnId) {
        const activeBtn = document.getElementById(activeConfig.btnId);
        if (activeBtn) activeBtn.classList.add('active');
      }

      if (activeConfig.containerId) {
        const activeContainer = document.getElementById(activeConfig.containerId);
        if (activeContainer) activeContainer.style.display = 'block';
      }

      if (activeConfig.extraContainerIds) {
        activeConfig.extraContainerIds.forEach(extraId => {
          const extraEl = document.getElementById(extraId);
          if (extraEl) extraEl.style.display = 'block';
        });
      }

      // 4. Trigger workspace loader hook if defined
      try {
        if (typeof activeConfig.onOpen === 'function') {
          activeConfig.onOpen();
        }
      } catch (err) {
        console.error(`Error opening workspace '${canonicalKey}':`, err);
      }
    }
  }

  async function init() {
    setupEventListeners();
    await fetchStudiesIndex();
    await loadStudy(state.studyId);
  }

  function setupEventListeners() {
    // Top-Level Tab Navigation (All 9 Workspaces)
    Object.entries(TABS).forEach(([tabKey, cfg]) => {
      const btn = document.getElementById(cfg.btnId);
      if (btn) {
        btn.addEventListener('click', (e) => {
          e.preventDefault();
          switchTab(tabKey);
        });
      }
    });

    // Multi-Study Search & Filter
    studySelect.addEventListener('change', (e) => {
      state.studyId = e.target.value;
      loadStudy(state.studyId);
    });

    studySearchInput.addEventListener('input', filterStudyDropdown);
    studyFilterSelect.addEventListener('change', filterStudyDropdown);

    // Single Study Inference Trigger
    if (runStudyInferenceBtn) {
      runStudyInferenceBtn.addEventListener('click', handleRunSingleStudyInference);
    }

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

    // Decision Quick-Action Buttons
    decisionBtns.forEach(btn => {
      if (!btn) return;
      btn.addEventListener('click', () => {
        if (isSessionFinalized()) return;
        decisionBtns.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        state.selectedReviewStatus = btn.dataset.status;
        updateFindingComparisonBox();
      });
    });

    // Save Finding Review Decision
    saveFindingReviewBtn.addEventListener('click', handleSaveFindingDecision);

    // Report Actions
    resetReportBtn.addEventListener('click', handleResetReport);
    saveDraftBtn.addEventListener('click', handleSaveReportDraft);
    validateReportBtn.addEventListener('click', handleValidateReview);
    finalizeReportBtn.addEventListener('click', handleFinalizeReview);
    finalizeReviewBtn.addEventListener('click', handleFinalizeReview);
    validateReviewBtn.addEventListener('click', handleValidateReview);

    // Accordions
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

    auditToggleBtn.addEventListener('click', () => {
      const isExpanded = auditToggleBtn.getAttribute('aria-expanded') === 'true';
      auditToggleBtn.setAttribute('aria-expanded', !isExpanded);
      auditTrailBody.style.display = isExpanded ? 'none' : 'block';
      const icon = auditToggleBtn.querySelector('.toggle-icon');
      if (icon) icon.textContent = isExpanded ? '▶' : '▼';
    });

    // Export Buttons
    exportJsonBtn.addEventListener('click', () => exportReview('json'));
    exportTextBtn.addEventListener('click', () => exportReview('text'));
  }

  // --- Multi-Study Browsing & Search ---
  async function fetchStudiesIndex() {
    try {
      const resp = await fetch('/api/studies');
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const data = await resp.json();
      state.allStudies = data.studies || [];
      populateStudyDropdown(state.allStudies);
    } catch (err) {
      console.error('Failed to fetch studies list:', err);
    }
  }

  function populateStudyDropdown(studies) {
    studySelect.innerHTML = '';
    studies.forEach(s => {
      const opt = document.createElement('option');
      opt.value = s.study_id;
      const viewsStr = (s.available_views || ['Frontal']).join('+');
      const valBadge = s.validation_status === 'VALIDATED' ? '✓ ' : '';
      opt.textContent = `${valBadge}${s.study_id} (${viewsStr}, ${s.images_count} img)`;
      if (s.study_id === state.studyId) opt.selected = true;
      studySelect.appendChild(opt);
    });
  }

  function filterStudyDropdown() {
    const query = (studySearchInput.value || '').trim().toLowerCase();
    const filter = studyFilterSelect.value;

    const filtered = state.allStudies.filter(s => {
      const matchQuery = s.study_id.toLowerCase().includes(query);
      if (!matchQuery) return false;
      if (filter === 'validated') return s.has_evidence || s.has_grounding || s.study_id === 'CXR1122';
      if (filter === 'dual_view') return (s.available_views || []).length > 1 || s.images_count > 1;
      return true;
    });

    populateStudyDropdown(filtered);
  }

  // --- Load Full Study Artifacts & Review Session ---
  async function loadStudy(studyId) {
    const currentSeq = ++state.requestSeq;
    primaryLoader.style.display = 'block';
    primaryLoader.textContent = `Loading ${studyId} radiograph...`;
    reportActionFeedback.textContent = '';
    if (singleInferenceFeedback) singleInferenceFeedback.textContent = '';
    
    // Clear canvases immediately to prevent showing previous study's image/heatmaps
    if (primaryCtx && primaryCanvas) primaryCtx.clearRect(0, 0, primaryCanvas.width, primaryCanvas.height);
    if (secondaryCtx && secondaryCanvas) secondaryCtx.clearRect(0, 0, secondaryCanvas.width, secondaryCanvas.height);
    state.primaryImages.original = null;
    state.primaryImages.heatmaps = {};
    state.secondaryImages.original = null;
    state.secondaryImages.heatmaps = {};

    try {
      // 1. Fetch images metadata
      const imgResp = await fetch(`/api/studies/${studyId}/images`);
      if (currentSeq !== state.requestSeq) return; // Ignore stale responses

      if (imgResp.ok) {
        const imgData = await imgResp.json();
        state.imagesMeta = imgData.images || [];
      } else {
        state.imagesMeta = [];
      }

      // Configure single vs dual view mode
      configureViewports();

      // 2. Fetch study artifacts
      const resp = await fetch(`/api/studies/${studyId}`);
      if (currentSeq !== state.requestSeq) return; // Ignore stale responses
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      state.studyData = await resp.json();

      const isInferenceRun = state.studyData.inference_status === 'COMPLETED' || state.studyData.has_artifacts;

      if (inferenceNotRunBanner) {
        inferenceNotRunBanner.style.display = isInferenceRun ? 'none' : 'block';
      }

      // 3. Preload original images and heatmaps
      await preloadPrimaryImages();
      if (state.imagesMeta.length >= 2) {
        await preloadSecondaryImages();
      }

      if (currentSeq !== state.requestSeq) return; // Ignore stale responses

      if (isInferenceRun) {
        // Set active finding to first available or Infiltration
        const evidence = state.studyData.evidence || [];
        if (evidence.length > 0) {
          const hasInfil = evidence.some(e => e.finding.toLowerCase() === 'infiltration');
          state.activeFinding = hasInfil ? 'Infiltration' : evidence[0].finding;
        }

        // Fetch or Create Review Session (Phase 1.2)
        await fetchReviewSession(studyId);
        if (currentSeq !== state.requestSeq) return;

        // Render Components
        renderHeaderMeta();
        renderFindingPills();
        renderActiveFinding();
        renderWhyFinding();
        renderQAReviews();
        renderTraceability();
        renderReportReview();
        renderBothCanvases();
        renderComparisonMetrics();
        renderAuditTrail();
        updateProgressBar();
      } else {
        // Study inference not yet triggered
        renderHeaderMeta();
        findingsSelector.innerHTML = '<div style="color: #94a3b8; font-size: 0.85rem; padding: 12px;">This study has not yet been processed through machine inference. Click <strong>[Run Inference]</strong> above to execute the 6-stage pipeline.</div>';
        findingsCountBadge.textContent = 'INFERENCE NOT RUN';
        findingsCountBadge.className = 'badge badge-neutral';
        renderBothCanvases();
      }

    } catch (err) {
      if (currentSeq === state.requestSeq) {
        console.error('Error loading study:', err);
        primaryLoader.textContent = `Error loading study: ${err.message}`;
      }
    } finally {
      if (currentSeq === state.requestSeq) {
        primaryLoader.style.display = 'none';
      }
    }
  }

  async function handleRunSingleStudyInference() {
    if (!runStudyInferenceBtn) return;
    runStudyInferenceBtn.disabled = true;
    runStudyInferenceBtn.textContent = 'Running Inference...';
    if (singleInferenceFeedback) {
      singleInferenceFeedback.textContent = 'Executing 6-stage machine pipeline (Vision → QA → Evidence → Grounding → LLM → Validation)...';
      singleInferenceFeedback.className = 'report-action-feedback';
    }

    try {
      const resp = await fetch(`/api/studies/${state.studyId}/process`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ force: true })
      });
      const data = await resp.json();
      if (!resp.ok || !data.success) {
        throw new Error(data.error || `HTTP ${resp.status}`);
      }

      if (singleInferenceFeedback) {
        singleInferenceFeedback.textContent = `✓ Inference completed successfully (${data.findings_count} findings generated).`;
        singleInferenceFeedback.className = 'report-action-feedback feedback-success';
      }

      await fetchStudiesIndex();
      await loadStudy(state.studyId);
    } catch (err) {
      console.error('Failed to run inference for study:', err);
      if (singleInferenceFeedback) {
        singleInferenceFeedback.textContent = `Inference Failed: ${err.message}`;
        singleInferenceFeedback.className = 'report-action-feedback feedback-error';
      }
    } finally {
      runStudyInferenceBtn.disabled = false;
      runStudyInferenceBtn.textContent = '▶ Run Inference';
    }
  }

  // --- Review Session Persistence (Phase 1.2) ---
  async function fetchReviewSession(studyId) {
    try {
      const resp = await fetch(`/api/studies/${studyId}/review?reviewer_id=${encodeURIComponent(state.activeReviewerId)}`);
      if (resp.ok) {
        const data = await resp.json();
        state.reviewSession = data.session || data;
      } else {
        state.reviewSession = null;
      }
    } catch (err) {
      console.warn('Could not fetch review session:', err);
      state.reviewSession = null;
    }
  }

  function isSessionFinalized() {
    return state.reviewSession && (state.reviewSession.status === 'finalized' || (state.reviewSession.report_review && state.reviewSession.report_review.finalized));
  }

  function renderHeaderMeta() {
    reportStudyId.textContent = state.studyId;
    
    if (state.reviewSession) {
      reportReviewId.textContent = state.reviewSession.review_id || '-';
      const isFinal = isSessionFinalized();
      reportStatusBadge.textContent = state.reviewSession.status.toUpperCase();
      reportStatusBadge.className = `badge ${isFinal ? 'badge-finalized' : 'badge-in-review'}`;
      
      const rev = state.reviewSession.reviewer;
      if (rev) {
        reviewerNameBadge.innerHTML = `Reviewer: <strong>${rev.display_name || rev.id}</strong>`;
      }
    }
  }

  function configureViewports() {
    if (state.imagesMeta.length >= 2) {
      viewportsContainer.className = 'viewports-container dual-view';
      secondaryViewport.style.display = 'flex';
      dualViewBadge.style.display = 'inline-block';
      syncToggleContainer.style.display = 'flex';
      viewTag.textContent = 'Dual-View (Frontal + Lateral)';
      secondaryViewportTag.textContent = `SECONDARY (${state.imagesMeta[1].view.toUpperCase()})`;
    } else {
      viewportsContainer.className = 'viewports-container single-view';
      secondaryViewport.style.display = 'none';
      dualViewBadge.style.display = 'none';
      syncToggleContainer.style.display = 'none';
      viewTag.textContent = state.imagesMeta[0] ? `${state.imagesMeta[0].view} View` : 'Frontal View';
    }
  }

  // --- Preload Canvas Images ---
  async function preloadPrimaryImages() {
    state.primaryImages.heatmaps = {};
    const primaryImgMeta = state.imagesMeta[0];
    const imgUrl = primaryImgMeta ? (primaryImgMeta.image_url || `/api/studies/${state.studyId}/image/${primaryImgMeta.image_id}`) : `/api/studies/${state.studyId}/image`;

    const origImg = new Image();
    origImg.crossOrigin = 'anonymous';
    origImg.src = imgUrl;
    await new Promise((resolve) => {
      origImg.onload = resolve;
      origImg.onerror = () => {
        console.warn('Could not load primary image url');
        resolve();
      };
    });
    state.primaryImages.original = origImg;

    primaryCanvas.width = origImg.naturalWidth || 512;
    primaryCanvas.height = origImg.naturalHeight || 624;

    const groundings = state.studyData.groundings || [];
    await Promise.all(groundings.map(g => {
      return new Promise(resolve => {
        const hmImg = new Image();
        hmImg.crossOrigin = 'anonymous';
        hmImg.src = g.heatmap_url;
        hmImg.onload = () => {
          state.primaryImages.heatmaps[g.finding] = hmImg;
          resolve();
        };
        hmImg.onerror = () => resolve();
      });
    }));
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
      const isAct = ev.finding.toLowerCase() === state.activeFinding.toLowerCase();
      
      // Determine reviewer decision status badge if reviewed
      let revStatusBadge = '';
      if (state.reviewSession) {
        const fr = (state.reviewSession.finding_reviews || []).find(r => r.finding.toLowerCase() === ev.finding.toLowerCase());
        if (fr && fr.reviewed) {
          revStatusBadge = `<span class="pill-reviewed-dot" title="${fr.reviewer_status}">✓</span>`;
        }
      }

      pill.className = `finding-pill ${isAct ? 'active' : ''} ${ev.status.toLowerCase()}`;
      pill.innerHTML = `
        <span class="pill-dot"></span>
        <span class="pill-name">${ev.finding}</span>
        ${revStatusBadge}
      `;
      pill.addEventListener('click', () => {
        state.activeFinding = ev.finding;
        renderFindingPills();
        renderActiveFinding();
        renderWhyFinding();
        renderQAReviews();
        renderBothCanvases();
      });
      findingsSelector.appendChild(pill);
    });
  }

  function renderActiveFinding() {
    const evidence = (state.studyData.evidence || []).find(e => e.finding.toLowerCase() === state.activeFinding.toLowerCase());
    if (!evidence) return;

    activeFindingName.textContent = evidence.finding.toUpperCase();
    activeFindingStatus.textContent = evidence.status.toUpperCase();
    activeFindingStatus.className = `status-badge status-${evidence.status.toLowerCase()}`;
    activeModelScore.textContent = evidence.model_score.toFixed(4);
    activeEvidenceSource.textContent = evidence.evidence_source || 'vision_model';
    activeLocation.textContent = evidence.location || 'unspecified';
    activeSeverity.textContent = evidence.severity || 'unspecified';

    const interp = STATUS_INTERPRETATIONS[evidence.status.toLowerCase()] || STATUS_INTERPRETATIONS.possible;
    statusInterpretationText.textContent = interp;

    populateReviewerFindingForm();
  }

  function updateDecisionButtonSelection(selectedStatus) {
    state.selectedReviewStatus = selectedStatus;
    decisionBtns.forEach(btn => {
      if (btn.dataset.status === selectedStatus) {
        btn.classList.add('selected');
      } else {
        btn.classList.remove('selected');
      }
    });

    const displayStatus = selectedStatus || 'not_reviewed';
    findingReviewStatePill.textContent = displayStatus.toUpperCase().replace('_', ' ');
    findingReviewStatePill.className = `review-state-indicator state-${displayStatus}`;

    // Comparison badges
    const evidence = (state.studyData && state.studyData.evidence ? state.studyData.evidence : []).find(e => e.finding.toLowerCase() === (state.activeFinding || '').toLowerCase());
    if (evidence) {
      compMachineStatus.textContent = evidence.status.toUpperCase();
      compMachineStatus.className = `status-badge status-${evidence.status.toLowerCase()}`;
    }

    compReviewerStatus.textContent = displayStatus.toUpperCase().replace('_', ' ');
    if (displayStatus === 'confirmed_present') {
      compReviewerStatus.className = 'status-badge status-supported';
    } else if (displayStatus === 'confirmed_absent') {
      compReviewerStatus.className = 'status-badge status-absent';
    } else if (displayStatus === 'uncertain') {
      compReviewerStatus.className = 'status-badge status-uncertain';
    } else if (displayStatus === 'needs_review') {
      compReviewerStatus.className = 'status-badge status-needs-review';
    } else {
      compReviewerStatus.className = 'status-badge status-neutral';
    }

    // Detect and show explicit disagreement
    const isDisagreement = (displayStatus === 'confirmed_present' && evidence && evidence.status === 'absent') ||
                           (displayStatus === 'confirmed_absent' && evidence && (evidence.status === 'supported' || evidence.status === 'possible'));
    disagreementBanner.style.display = isDisagreement ? 'block' : 'none';
  }

  function populateReviewerFindingForm() {
    const isFinal = isSessionFinalized();
    saveFindingReviewBtn.disabled = isFinal;
    reviewerLocationInput.disabled = isFinal;
    reviewerSeveritySelect.disabled = isFinal;
    reviewerNotesInput.disabled = isFinal;
    decisionBtns.forEach(btn => btn.disabled = isFinal);

    let fr = null;
    if (state.reviewSession) {
      fr = (state.reviewSession.finding_reviews || []).find(r => r.finding.toLowerCase() === (state.activeFinding || '').toLowerCase());
    }

    const currentStatus = fr ? fr.reviewer_status : 'not_reviewed';
    updateDecisionButtonSelection(currentStatus);

    reviewerLocationInput.value = fr && fr.reviewer_location !== 'unspecified' ? fr.reviewer_location : '';
    reviewerSeveritySelect.value = fr ? fr.reviewer_severity : 'unspecified';
    reviewerNotesInput.value = fr ? (fr.reviewer_comment || '') : '';
  }

  async function handleSaveFindingDecision() {
    if (isSessionFinalized()) return;
    reviewSaveStatus.textContent = 'Saving decision...';
    reviewSaveStatus.style.color = '#fbbf24';

    const selectedStat = state.selectedReviewStatus || 'not_reviewed';
    const payload = {
      finding: state.activeFinding,
      reviewer_status: selectedStat,
      reviewer_location: reviewerLocationInput.value.trim() || 'unspecified',
      reviewer_severity: reviewerSeveritySelect.value,
      reviewer_comment: reviewerNotesInput.value.trim(),
      reviewer_id: state.activeReviewerId,
      reviewed: selectedStat !== 'not_reviewed'
    };

    try {
      const resp = await fetch(`/api/studies/${state.studyId}/review/finding`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (!resp.ok) {
        const errJson = await resp.json();
        throw new Error(errJson.error || `HTTP ${resp.status}`);
      }

      const resData = await resp.json();
      
      // Update local review session immediately
      if (resData.session) {
        state.reviewSession = resData.session;
      } else if (state.reviewSession && resData.finding_review) {
        const frIdx = (state.reviewSession.finding_reviews || []).findIndex(r => r.finding.toLowerCase() === state.activeFinding.toLowerCase());
        if (frIdx >= 0) {
          state.reviewSession.finding_reviews[frIdx] = resData.finding_review;
        }
      }

      // Re-fetch authoritative review session from backend disk
      await fetchReviewSession(state.studyId);

      reviewSaveStatus.textContent = '✓ Decision Saved';
      reviewSaveStatus.style.color = '#10b981';
      renderFindingPills();
      populateReviewerFindingForm();
      renderComparisonMetrics();
      renderAuditTrail();
      updateProgressBar();
      setTimeout(() => { reviewSaveStatus.textContent = ''; }, 2500);

    } catch (err) {
      console.error('Failed to save finding decision:', err);
      reviewSaveStatus.textContent = `Error: ${err.message}`;
      reviewSaveStatus.style.color = '#ef4444';
    }
  }

  function renderWhyFinding() {
    const evidence = (state.studyData.evidence || []).find(e => e.finding.toLowerCase() === state.activeFinding.toLowerCase());
    const grounding = (state.studyData.groundings || []).find(g => g.finding.toLowerCase() === state.activeFinding.toLowerCase());
    const allQuestions = state.studyData.qa_questions || [];
    const presenceQ = allQuestions.find(q => q.finding.toLowerCase() === state.activeFinding.toLowerCase() && q.level === 1);

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

  // --- Diagnostic QA & Review Layer ---
  function renderQAReviews() {
    qaQuestionsList.innerHTML = '';
    const isFinal = isSessionFinalized();
    const allQuestions = state.studyData.qa_questions || [];
    const relevantMachine = allQuestions.filter(q => q.finding.toLowerCase() === state.activeFinding.toLowerCase());

    if (relevantMachine.length === 0) {
      qaQuestionsList.innerHTML = '<div class="qa-review-card"><span class="qa-text">No QA questions recorded for this finding.</span></div>';
      return;
    }

    relevantMachine.forEach(q => {
      const card = document.createElement('div');
      card.className = 'qa-review-card';

      // Find reviewer QA state
      let qr = null;
      if (state.reviewSession) {
        qr = (state.reviewSession.qa_reviews || []).find(r => r.question_id === q.question_id);
      }

      const machineAns = (q.answer || 'uncertain').toUpperCase();
      const revAns = qr && qr.reviewer_answer !== 'not_reviewed' ? qr.reviewer_answer : 'not_reviewed';
      const isDiff = revAns !== 'not_reviewed' && revAns.toLowerCase() !== (q.answer || '').toLowerCase();

      card.innerHTML = `
        <div class="qa-card-header">
          <span>${q.question_id} (Level ${q.level})</span>
          ${isDiff ? '<span class="qa-diff-badge">CORRECTION</span>' : ''}
        </div>
        <div class="qa-card-question">${q.question_text}</div>
        <div class="qa-answers-grid">
          <div class="qa-side-box">
            <span class="qa-side-title">MACHINE ANSWER (IMMUTABLE)</span>
            <span class="qa-val">${machineAns}</span>
          </div>
          <div class="qa-side-box">
            <span class="qa-side-title">REVIEWER ANSWER</span>
            <select class="reviewer-select qa-select" id="qa_sel_${q.question_id}" ${isFinal ? 'disabled' : ''}>
              <option value="not_reviewed" ${revAns === 'not_reviewed' ? 'selected' : ''}>Not Reviewed</option>
              <option value="yes" ${revAns === 'yes' ? 'selected' : ''}>Yes</option>
              <option value="no" ${revAns === 'no' ? 'selected' : ''}>No</option>
              <option value="uncertain" ${revAns === 'uncertain' ? 'selected' : ''}>Uncertain</option>
              <option value="unspecified" ${revAns === 'unspecified' ? 'selected' : ''}>Unspecified</option>
              <option value="right" ${revAns === 'right' ? 'selected' : ''}>Right</option>
              <option value="left" ${revAns === 'left' ? 'selected' : ''}>Left</option>
              <option value="bilateral" ${revAns === 'bilateral' ? 'selected' : ''}>Bilateral</option>
              <option value="mild" ${revAns === 'mild' ? 'selected' : ''}>Mild</option>
              <option value="moderate" ${revAns === 'moderate' ? 'selected' : ''}>Moderate</option>
              <option value="severe" ${revAns === 'severe' ? 'selected' : ''}>Severe</option>
            </select>
          </div>
        </div>
        <div class="qa-input-row">
          <input type="text" class="reviewer-input qa-comment-input" id="qa_comm_${q.question_id}" placeholder="Reviewer comment..." value="${qr ? (qr.reviewer_comment || '') : ''}" ${isFinal ? 'disabled' : ''}>
          <button class="btn btn-sm btn-primary btn-save-qa" id="qa_btn_${q.question_id}" ${isFinal ? 'disabled' : ''}>Save QA</button>
        </div>
      `;

      // Attach QA Save Handler
      const saveBtn = card.querySelector(`#qa_btn_${q.question_id}`);
      saveBtn.addEventListener('click', async () => {
        const sel = card.querySelector(`#qa_sel_${q.question_id}`);
        const comm = card.querySelector(`#qa_comm_${q.question_id}`);
        saveBtn.disabled = true;
        saveBtn.textContent = 'Saving...';

        try {
          const resp = await fetch(`/api/studies/${state.studyId}/review/qa`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              question_id: q.question_id,
              reviewer_answer: sel.value,
              reviewer_comment: comm.value.trim(),
              reviewer_id: state.activeReviewerId
            })
          });

          if (!resp.ok) {
            const errJ = await resp.json();
            throw new Error(errJ.error || `HTTP ${resp.status}`);
          }

          await fetchReviewSession(state.studyId);
          renderQAReviews();
          renderAuditTrail();
        } catch (err) {
          alert(`Failed to save QA review: ${err.message}`);
        } finally {
          saveBtn.disabled = false;
          saveBtn.textContent = 'Save QA';
        }
      });

      qaQuestionsList.appendChild(card);
    });
  }

  function renderTraceability() {
    traceabilityChain.innerHTML = '';
    const evidence = (state.studyData.evidence || []).find(e => e.finding.toLowerCase() === state.activeFinding.toLowerCase());
    const grounding = (state.studyData.groundings || []).find(g => g.finding.toLowerCase() === state.activeFinding.toLowerCase());
    const reportFinding = ((state.studyData.report || {}).findings || []).find(f => f.finding.toLowerCase() === state.activeFinding.toLowerCase());
    const allQuestions = state.studyData.qa_questions || [];
    const presenceQ = allQuestions.find(q => q.finding.toLowerCase() === state.activeFinding.toLowerCase() && q.level === 1);

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
        title: 'LLM Report Synthesis',
        sub: reportFinding ? `Statement: "${reportFinding.statement}"` : 'Statement synthesized in report',
        complete: !!reportFinding
      },
      {
        num: 6,
        title: 'Human Review & Finalization',
        sub: isSessionFinalized() ? '✓ Finalized and locked by reviewer' : 'In review draft mode',
        complete: isSessionFinalized()
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

  // --- Report Review & Finalization Editor ---
  function renderReportReview() {
    reportFindingsEditorList.innerHTML = '';
    const isFinal = isSessionFinalized();

    // Lock all report buttons if finalized
    resetReportBtn.disabled = isFinal;
    saveDraftBtn.disabled = isFinal;
    finalizeReportBtn.disabled = isFinal;
    finalizeReviewBtn.disabled = isFinal;
    reportImpressionEditor.disabled = isFinal;
    reportReviewerComment.disabled = isFinal;

    if (isFinal) {
      reportWarningHeading.textContent = 'FINALIZED RESEARCH REVIEW — IMMUTABLE';
      reportWarningHeading.style.color = '#34d399';
      reportWarningSub.textContent = `Completed by ${state.reviewSession.reviewer.display_name} on ${state.reviewSession.completed_at}. Research metadata only.`;
    } else {
      reportWarningHeading.textContent = 'RESEARCH OUTPUT — REQUIRES HUMAN REVIEW';
      reportWarningHeading.style.color = '#fbbf24';
      reportWarningSub.textContent = 'Reviewer corrections and finalized reports are research metadata only. They are not certified clinical diagnoses.';
    }

    const reportReview = (state.reviewSession && state.reviewSession.report_review) ? state.reviewSession.report_review : {};
    const findings = reportReview.final_findings || (state.studyData.report || {}).findings || [];
    const impression = reportReview.final_impression || (state.studyData.report || {}).impression || [];
    const reviewerComm = reportReview.reviewer_comment || '';

    // Render Editable Findings List
    findings.forEach((f, idx) => {
      const item = document.createElement('div');
      item.className = 'finding-editor-card';
      item.innerHTML = `
        <div class="finding-editor-header">
          <span class="finding-editor-name">${f.finding}</span>
          <select class="reviewer-select finding-status-select" data-idx="${idx}" ${isFinal ? 'disabled' : ''}>
            <option value="supported" ${f.status === 'supported' ? 'selected' : ''}>SUPPORTED</option>
            <option value="possible" ${f.status === 'possible' ? 'selected' : ''}>POSSIBLE</option>
            <option value="uncertain" ${f.status === 'uncertain' ? 'selected' : ''}>UNCERTAIN</option>
            <option value="absent" ${f.status === 'absent' ? 'selected' : ''}>ABSENT</option>
          </select>
        </div>
        <input type="text" class="finding-editor-statement-input" data-idx="${idx}" value="${f.statement}" placeholder="Finding statement..." ${isFinal ? 'disabled' : ''}>
      `;
      reportFindingsEditorList.appendChild(item);
    });

    // Render Impression Textarea
    reportImpressionEditor.value = Array.isArray(impression) ? impression.join('\n') : String(impression);
    reportReviewerComment.value = reviewerComm;
  }

  function collectReportDraftPayload() {
    const finalFindings = [];
    const cards = reportFindingsEditorList.querySelectorAll('.finding-editor-card');
    cards.forEach((card, idx) => {
      const name = card.querySelector('.finding-editor-name').textContent.trim();
      const status = card.querySelector('.finding-status-select').value;
      const statement = card.querySelector('.finding-editor-statement-input').value.trim();
      finalFindings.push({
        finding: name,
        statement: statement,
        status: status,
        location: 'unspecified',
        severity: 'unspecified'
      });
    });

    const impressionRaw = reportImpressionEditor.value.trim();
    const finalImpression = impressionRaw.split('\n').map(s => s.trim()).filter(Boolean);
    const reviewerComm = reportReviewerComment.value.trim();

    return {
      final_findings: finalFindings,
      final_impression: finalImpression,
      reviewer_comment: reviewerComm,
      reviewer_id: state.activeReviewerId
    };
  }

  async function handleResetReport() {
    if (isSessionFinalized()) return;
    if (!confirm('Are you sure you want to reset the report draft back to the original machine-generated report?')) return;

    reportActionFeedback.textContent = 'Resetting to machine baseline...';
    reportActionFeedback.className = 'report-action-feedback';

    try {
      const resp = await fetch(`/api/studies/${state.studyId}/review/report/reset`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ reviewer_id: state.activeReviewerId })
      });
      if (!resp.ok) {
        const errJ = await resp.json();
        throw new Error(errJ.error || `HTTP ${resp.status}`);
      }

      await fetchReviewSession(state.studyId);
      renderReportReview();
      renderAuditTrail();
      reportActionFeedback.textContent = '✓ Report reset to machine baseline.';
      reportActionFeedback.className = 'report-action-feedback feedback-success';
    } catch (err) {
      reportActionFeedback.textContent = `Error: ${err.message}`;
      reportActionFeedback.className = 'report-action-feedback feedback-error';
    }
  }

  async function handleSaveReportDraft() {
    if (isSessionFinalized()) return;
    reportActionFeedback.textContent = 'Saving report draft...';
    reportActionFeedback.className = 'report-action-feedback';

    const draft = collectReportDraftPayload();
    try {
      const resp = await fetch(`/api/studies/${state.studyId}/review/report`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(draft)
      });
      if (!resp.ok) {
        const errJ = await resp.json();
        throw new Error(errJ.error || `HTTP ${resp.status}`);
      }

      await fetchReviewSession(state.studyId);
      renderReportReview();
      renderAuditTrail();
      updateProgressBar();
      reportActionFeedback.textContent = '✓ Draft saved successfully.';
      reportActionFeedback.className = 'report-action-feedback feedback-success';
    } catch (err) {
      reportActionFeedback.textContent = `Error saving draft: ${err.message}`;
      reportActionFeedback.className = 'report-action-feedback feedback-error';
    }
  }

  async function handleValidateReview() {
    reportActionFeedback.textContent = 'Validating review completeness...';
    reportActionFeedback.className = 'report-action-feedback';

    if (!state.reviewSession) {
      reportActionFeedback.textContent = 'No review session found.';
      reportActionFeedback.className = 'report-action-feedback feedback-error';
      return;
    }

    const unreviewed = (state.reviewSession.finding_reviews || []).filter(fr => !fr.reviewed);
    if (unreviewed.length > 0) {
      const names = unreviewed.map(fr => fr.finding).join(', ');
      reportActionFeedback.textContent = `Validation Incomplete: ${unreviewed.length} findings unreviewed (${names}).`;
      reportActionFeedback.className = 'report-action-feedback feedback-error';
      validationProgressText.textContent = `${unreviewed.length} Unreviewed`;
      stepValidation.className = 'progress-step-item';
    } else {
      reportActionFeedback.textContent = '✓ All pre-finalization validation checks PASSED. Ready to finalize.';
      reportActionFeedback.className = 'report-action-feedback feedback-success';
      validationProgressText.textContent = 'PASSED';
      stepValidation.className = 'progress-step-item completed';
    }
  }

  async function handleFinalizeReview() {
    if (isSessionFinalized()) {
      alert('This review session is already finalized and locked.');
      return;
    }

    if (!confirm('Finalizing will permanently lock all findings, QA answers, and report contents. Proceed?')) {
      return;
    }

    reportActionFeedback.textContent = 'Finalizing review session...';
    reportActionFeedback.className = 'report-action-feedback';

    // First save latest report draft
    const draft = collectReportDraftPayload();
    await fetch(`/api/studies/${state.studyId}/review/report`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(draft)
    });

    try {
      const resp = await fetch(`/api/studies/${state.studyId}/review/finalize`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ reviewer_id: state.activeReviewerId })
      });

      const resData = await resp.json();
      if (!resp.ok || !resData.success) {
        const issues = resData.validation_errors || [resData.error];
        throw new Error(issues.join('; '));
      }

      await fetchReviewSession(state.studyId);
      renderHeaderMeta();
      renderFindingPills();
      populateReviewerFindingForm();
      renderQAReviews();
      renderTraceability();
      renderReportReview();
      renderComparisonMetrics();
      renderAuditTrail();
      updateProgressBar();

      reportActionFeedback.textContent = '✓ Review successfully validated, finalized, and permanently locked.';
      reportActionFeedback.className = 'report-action-feedback feedback-success';
      alert('Review report finalized successfully!');

    } catch (err) {
      reportActionFeedback.textContent = `Finalization Failed: ${err.message}`;
      reportActionFeedback.className = 'report-action-feedback feedback-error';
      alert(`Cannot Finalize Review:\n${err.message}`);
    }
  }

  // --- Comparison Metrics & Progress ---
  function renderComparisonMetrics() {
    if (!state.reviewSession) return;

    const findingReviews = state.reviewSession.finding_reviews || [];
    const total = findingReviews.length;
    let reviewed = 0;
    let notReviewed = 0;
    let confPres = 0;
    let confAbs = 0;
    let uncert = 0;
    let needsRev = 0;
    let disagreements = 0;

    findingReviews.forEach(fr => {
      if (fr.reviewed) {
        reviewed += 1;
      } else {
        notReviewed += 1;
      }

      if (fr.reviewer_status === 'confirmed_present') confPres += 1;
      else if (fr.reviewer_status === 'confirmed_absent') confAbs += 1;
      else if (fr.reviewer_status === 'uncertain') uncert += 1;
      else if (fr.reviewer_status === 'needs_review') needsRev += 1;

      // Disagreement check
      const mStat = fr.machine_status;
      const rStat = fr.reviewer_status;
      if ((rStat === 'confirmed_present' && mStat === 'absent') ||
          (rStat === 'confirmed_absent' && (mStat === 'supported' || mStat === 'possible'))) {
        disagreements += 1;
      }
    });

    metricTotal.textContent = total;
    metricReviewed.textContent = reviewed;
    metricNotReviewed.textContent = notReviewed;
    metricConfirmedPresent.textContent = confPres;
    metricConfirmedAbsent.textContent = confAbs;
    metricUncertain.textContent = uncert;
    metricNeedsReview.textContent = needsRev;
    metricDisagreements.textContent = disagreements;

    const rate = total > 0 ? Math.round((reviewed / total) * 100) : 0;
    metricsReviewRate.textContent = `${rate}% Complete`;
  }

  function updateProgressBar() {
    if (!state.reviewSession) return;

    const findingReviews = state.reviewSession.finding_reviews || [];
    const total = findingReviews.length;
    const reviewed = findingReviews.filter(fr => fr.reviewed).length;
    const isFinal = isSessionFinalized();

    // Step 1: Evidence
    stepEvidence.className = 'progress-step-item completed';

    // Step 2: Findings
    findingsProgressText.textContent = `${reviewed} / ${total} Reviewed`;
    if (reviewed === total && total > 0) {
      stepFindings.className = 'progress-step-item completed';
    } else if (reviewed > 0) {
      stepFindings.className = 'progress-step-item active';
    } else {
      stepFindings.className = 'progress-step-item';
    }

    // Step 3: Report Draft
    stepReport.className = 'progress-step-item active';
    reportProgressText.textContent = isFinal ? 'Locked Draft' : 'Active Draft';

    // Step 4: Validation
    if (reviewed === total && total > 0) {
      stepValidation.className = 'progress-step-item completed';
      validationProgressText.textContent = 'Valid';
    } else {
      stepValidation.className = 'progress-step-item';
      validationProgressText.textContent = 'Pending';
    }

    // Step 5: Finalization
    if (isFinal) {
      stepFinalize.className = 'progress-step-item finalize-step completed';
      sessionStatusBadge.textContent = 'FINALIZED';
      sessionStatusBadge.className = 'badge badge-finalized';
    } else {
      stepFinalize.className = 'progress-step-item finalize-step active';
      sessionStatusBadge.textContent = 'IN REVIEW';
      sessionStatusBadge.className = 'badge badge-in-review';
    }
  }

  // --- Audit Trail ---
  function renderAuditTrail() {
    auditTableBody.innerHTML = '';
    if (!state.reviewSession) return;

    const trail = state.reviewSession.audit_trail || [];
    auditCountBadge.textContent = `${trail.length} records`;

    trail.forEach(entry => {
      const tr = document.createElement('tr');
      const timeStr = entry.timestamp ? entry.timestamp.replace('T', ' ').replace('Z', '') : '-';
      tr.innerHTML = `
        <td>${timeStr}</td>
        <td>${entry.reviewer_id || 'reviewer'}</td>
        <td><strong>${entry.finding || '-'}</strong></td>
        <td>${entry.field}</td>
        <td>${entry.old_value !== null ? String(entry.old_value) : '-'}</td>
        <td><strong style="color: #60a5fa">${String(entry.new_value)}</strong></td>
      `;
      auditTableBody.appendChild(tr);
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
      ctx.globalAlpha = 1.0;
      ctx.drawImage(orig, 0, 0, canvas.width, canvas.height);

      const hm = imagesObj.heatmaps[state.activeFinding];
      if (hm && hm.complete) {
        ctx.globalAlpha = state.opacity;
        ctx.drawImage(hm, 0, 0, canvas.width, canvas.height);
      }
    }

    ctx.globalAlpha = 1.0;

    if (!isPrimary) {
      const hasHm = !!imagesObj.heatmaps[state.activeFinding];
      if (!hasHm && state.viewMode !== 'original') {
        secondaryUnavailableNotice.style.display = 'block';
      } else {
        secondaryUnavailableNotice.style.display = 'none';
      }
    }
  }

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
    primaryCanvas.style.transform = `translate(${state.primary.panX}px, ${state.primary.panY}px) scale(${state.primary.zoom})`;
    if (state.imagesMeta.length >= 2) {
      secondaryCanvas.style.transform = `translate(${state.secondary.panX}px, ${state.secondary.panY}px) scale(${state.secondary.zoom})`;
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

    const force = batchForceReprocess ? batchForceReprocess.checked : false;

    triggerBatchBtn.disabled = true;
    triggerBatchBtn.textContent = 'Submitting Batch Job...';
    batchStatusContainer.style.display = 'block';

    try {
      const resp = await fetch('/api/batch/process', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ study_ids: studyIds, force: force })
      });

      if (!resp.ok) {
        const errJson = await resp.json();
        throw new Error(errJson.error || `HTTP ${resp.status}`);
      }

      const data = await resp.json();
      state.activeBatchJobId = data.job_id;
      batchJobId.textContent = data.job_id;
      batchJobStatusBadge.textContent = 'QUEUED';
      batchJobStatusBadge.className = 'status-badge status-possible';

      if (state.batchPollInterval) clearInterval(state.batchPollInterval);
      state.batchPollInterval = setInterval(checkBatchStatus, 1000);
      checkBatchStatus();
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
      const url = state.activeBatchJobId ? `/api/batch/${state.activeBatchJobId}` : '/api/batch/status';
      const resp = await fetch(url);
      if (!resp.ok) return;
      const data = await resp.json();

      if (!data.job_id) {
        batchStatusContainer.style.display = 'none';
        return;
      }

      batchStatusContainer.style.display = 'block';
      batchJobId.textContent = data.job_id;
      batchJobStatusBadge.textContent = data.status;

      const total = data.total_studies || 1;
      const completed = data.completed_studies || 0;
      const failed = data.failed_studies || 0;
      const pct = data.progress_percentage || Math.round(((completed + failed) / total) * 100);

      if (batchProgressBar) {
        batchProgressBar.style.width = `${pct}%`;
        if (data.status === 'COMPLETED') batchProgressBar.style.background = '#10b981';
        else if (data.status === 'FAILED') batchProgressBar.style.background = '#ef4444';
        else batchProgressBar.style.background = '#3b82f6';
      }

      if (batchProgressText) {
        batchProgressText.textContent = `${completed} / ${total} Completed${failed > 0 ? ` (${failed} Failed)` : ''}`;
      }

      if (data.status === 'COMPLETED' || data.status === 'COMPLETED_WITH_ERRORS') {
        batchJobStatusBadge.className = 'status-badge status-supported';
        if (state.batchPollInterval) {
          clearInterval(state.batchPollInterval);
          state.batchPollInterval = null;
        }
        await fetchStudiesIndex();
      } else if (data.status === 'FAILED') {
        batchJobStatusBadge.className = 'status-badge status-absent';
        if (state.batchPollInterval) {
          clearInterval(state.batchPollInterval);
          state.batchPollInterval = null;
        }
      } else {
        batchJobStatusBadge.className = 'status-badge status-possible';
      }

      renderBatchStages(data);
      renderBatchStudiesTable(data);
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
      pill.className = `stage-pill ${isPastOrCurrent ? 'COMPLETED' : 'PENDING'}`;
      pill.textContent = st;
      batchStagesGrid.appendChild(pill);
    });
  }

  function renderBatchStudiesTable(data) {
    if (!batchStudiesTableBody) return;
    batchStudiesTableBody.innerHTML = '';

    const studiesProgress = data.studies_progress || {};
    const studyIds = data.study_ids || Object.keys(studiesProgress);

    studyIds.forEach(sid => {
      const sp = studiesProgress[sid] || { stage: 'PENDING', status: 'PENDING', progress: 0.0 };
      const tr = document.createElement('tr');
      
      let statusBadgeClass = 'badge badge-neutral';
      if (sp.status === 'COMPLETED') statusBadgeClass = 'status-badge status-supported';
      else if (sp.status === 'FAILED') statusBadgeClass = 'status-badge status-absent';
      else if (sp.status === 'RUNNING') statusBadgeClass = 'status-badge status-possible';

      const progressPct = Math.round((sp.progress || 0) * 100);

      tr.innerHTML = `
        <td><strong>${sid}</strong></td>
        <td><span class="info-tag">${sp.stage || 'PENDING'}</span></td>
        <td>
          <div style="display: flex; align-items: center; gap: 8px;">
            <div style="width: 60px; background: #334155; height: 6px; border-radius: 3px; overflow: hidden;">
              <div style="width: ${progressPct}%; height: 100%; background: #38bdf8;"></div>
            </div>
            <span>${progressPct}%</span>
          </div>
        </td>
        <td><span class="${statusBadgeClass}">${sp.status}</span></td>
        <td style="color: ${sp.error ? '#ef4444' : '#94a3b8'}; font-size: 0.75rem;">
          ${sp.error ? sp.error : (sp.completion_time ? `Finished ${sp.completion_time.slice(11, 19)}` : 'In Queue')}
        </td>
      `;
      batchStudiesTableBody.appendChild(tr);
    });
  }

  function renderBatchLog(data) {
    batchLog.innerHTML = '';
    const results = data.results || [];
    if (results.length === 0) {
      batchLog.textContent = `Running real machine pipeline across ${data.total_studies || 0} studies...`;
      return;
    }

    results.forEach(r => {
      const line = document.createElement('div');
      line.className = 'batch-log-line';
      line.innerHTML = `
        <span class="log-id"><strong>${r.study_id}</strong></span>
        <span class="log-status">[${r.status}]</span>
        <span class="log-stage">Stage: ${r.stage}</span>
        <span>Findings: ${r.findings_count !== undefined ? r.findings_count : '-'}</span>
        ${r.error ? `<span class="log-error" style="color: #ef4444;">${r.error}</span>` : ''}
      `;
      batchLog.appendChild(line);
    });
  }

  // --- Review Export Utilities (Phase 1.2) ---
  async function exportReview(format) {
    try {
      const resp = await fetch(`/api/studies/${state.studyId}/review/export?format=${format}&reviewer_id=${encodeURIComponent(state.activeReviewerId)}`);
      if (!resp.ok) throw new Error(`Export failed with HTTP ${resp.status}`);

      if (format === 'text') {
        const text = await resp.text();
        const blob = new Blob([text], { type: 'text/plain' });
        downloadBlob(blob, `${state.studyId}_${state.activeReviewerId}_reviewed_report.txt`);
      } else {
        const json = await resp.json();
        const blob = new Blob([JSON.stringify(json, null, 2)], { type: 'application/json' });
        downloadBlob(blob, `${state.studyId}_${state.activeReviewerId}_reviewed_report.json`);
      }
    } catch (err) {
      console.error('Export error:', err);
      alert('Failed to export review report.');
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

  // ============================================================
  // PHASE 1.4 — MULTI-STUDY DATASET, QUEUE & ANALYTICS
  // ============================================================

  async function loadDatasetQueue() {
    try {
      // 1. Fetch Stats
      const statsResp = await fetch('/api/dataset/stats');
      if (statsResp.ok) {
        const stats = await statsResp.json();
        if (statTotalStudies) statTotalStudies.textContent = stats.total_studies || 0;
        if (statUnreviewed) statUnreviewed.textContent = stats.unreviewed || 0;
        if (statInReview) statInReview.textContent = stats.in_review || 0;
        if (statAwaitingConsensus) statAwaitingConsensus.textContent = stats.awaiting_consensus || 0;
        if (statAdjudicationRequired) statAdjudicationRequired.textContent = stats.adjudication_required || 0;
        if (statFinalized) statFinalized.textContent = stats.finalized || 0;
      }

      // 2. Fetch Filtered Queue
      const params = new URLSearchParams();
      if (queueStatusFilter && queueStatusFilter.value !== 'ALL') params.set('status', queueStatusFilter.value);
      if (queueConsensusFilter && queueConsensusFilter.value !== 'ALL') params.set('consensus_status', queueConsensusFilter.value);
      if (queueSearchInput && queueSearchInput.value.trim()) params.set('q', queueSearchInput.value.trim());
      if (queueSortBy) params.set('sort_by', queueSortBy.value);

      const queueResp = await fetch(`/api/review-queue?${params.toString()}`);
      if (queueResp.ok) {
        const qData = await queueResp.json();
        renderQueueTable(qData.queue || []);
      }
    } catch (err) {
      console.error('Failed to load dataset queue:', err);
    }
  }

  function renderQueueTable(items) {
    if (!queueTableBody) return;
    queueTableBody.innerHTML = '';

    if (items.length === 0) {
      queueTableBody.innerHTML = '<tr><td colspan="11" class="empty-hint">No studies matched the selected queue filters.</td></tr>';
      return;
    }

    items.forEach(item => {
      const tr = document.createElement('tr');
      const wStat = (item.workflow_status || 'UNREVIEWED').toLowerCase();
      const viewsStr = (item.views || ['Frontal']).join(', ');
      const compPct = Math.round((item.review_completion || 0) * 100);

      tr.innerHTML = `
        <td><strong>${item.study_id}</strong></td>
        <td>${viewsStr}</td>
        <td><span class="info-tag">${item.evidence_count} Findings</span></td>
        <td>${item.machine_report ? '<span style="color: #34d399;">✓ Available</span>' : '<span class="text-muted">None</span>'}</td>
        <td>${compPct}% complete</td>
        <td><strong>${item.reviewer_count}</strong></td>
        <td><span class="status-pill ${wStat}">${(item.consensus_status || 'NOT_STARTED').replace('_', ' ')}</span></td>
        <td>${item.adjudication_required ? '<span class="status-badge status-needs-review">⚠️ Required</span>' : '<span class="text-muted">None</span>'}</td>
        <td><span class="status-pill ${wStat}">${(item.workflow_status || 'UNREVIEWED').replace('_', ' ')}</span></td>
        <td style="font-family: var(--font-mono); font-size: 0.68rem; color: var(--text-muted);">${item.last_updated ? item.last_updated.substring(0, 19).replace('T', ' ') : '-'}</td>
        <td>
          <button class="btn btn-sm btn-primary btn-open-study" data-study-id="${item.study_id}">Open Study</button>
        </td>
      `;

      const openBtn = tr.querySelector('.btn-open-study');
      openBtn.addEventListener('click', async () => {
        state.studyId = item.study_id;
        if (studySelect) studySelect.value = item.study_id;
        await loadStudy(state.studyId);
        switchTab('individual');
      });

      queueTableBody.appendChild(tr);
    });
  }

  async function loadEvaluationAnalytics() {
    try {
      const resp = await fetch('/api/dataset/analytics');
      if (!resp.ok) return;
      const data = await resp.json();

      // Overview
      const rm = data.review_metrics || {};
      const cm = data.consensus_metrics || {};
      const agr = data.agreement_analytics || {};

      if (analyticsReviewCompletion) analyticsReviewCompletion.textContent = `${Math.round((rm.review_completion_rate || 0) * 100)}%`;
      if (analyticsActiveReviewers) analyticsActiveReviewers.textContent = rm.total_reviewers || 0;
      if (analyticsUnanimousFindings) analyticsUnanimousFindings.textContent = cm.unanimous_findings || 0;
      if (analyticsFinalizedReports) analyticsFinalizedReports.textContent = cm.finalized_consensus_reports || 0;

      // Cohen's Kappa (2 Reviewers)
      const twoRev = agr.two_reviewer_studies || {};
      if (analyticsCohenKappa) analyticsCohenKappa.textContent = twoRev.average_kappa !== null && twoRev.average_kappa !== undefined ? twoRev.average_kappa.toFixed(3) : '-';
      if (analyticsCohenMeta) analyticsCohenMeta.textContent = `Sample size: N=${twoRev.sample_size_studies || 0} studies (${twoRev.count || 0} evals)`;
      if (analyticsCohenInterp) {
        const k = twoRev.average_kappa;
        analyticsCohenInterp.textContent = k === null || k === undefined ? 'Pending' : (k > 0.8 ? 'Almost Perfect' : (k > 0.6 ? 'Substantial' : (k > 0.4 ? 'Moderate' : 'Fair')));
      }

      // Fleiss' Kappa (3+ Reviewers)
      const multiRev = agr.multi_reviewer_studies || {};
      if (analyticsFleissKappa) analyticsFleissKappa.textContent = multiRev.average_kappa !== null && multiRev.average_kappa !== undefined ? multiRev.average_kappa.toFixed(3) : '-';
      if (analyticsFleissMeta) analyticsFleissMeta.textContent = `Sample size: N=${multiRev.sample_size_studies || 0} studies (${multiRev.count || 0} evals)`;
      if (analyticsFleissInterp) {
        const k = multiRev.average_kappa;
        analyticsFleissInterp.textContent = k === null || k === undefined ? 'Pending' : (k > 0.8 ? 'Almost Perfect' : (k > 0.6 ? 'Substantial' : (k > 0.4 ? 'Moderate' : 'Fair')));
      }

      // Reviewer Activity Table
      if (reviewerActivityTableBody) {
        reviewerActivityTableBody.innerHTML = '';
        const reviewers = data.reviewer_workflow || [];
        if (reviewers.length === 0) {
          reviewerActivityTableBody.innerHTML = '<tr><td colspan="6" class="empty-hint">No active reviewer workflow activity logged.</td></tr>';
        } else {
          reviewers.forEach(r => {
            const tr = document.createElement('tr');
            tr.innerHTML = `
              <td><strong>${r.reviewer_id}</strong></td>
              <td>${r.studies_assigned}</td>
              <td>${r.studies_completed}</td>
              <td>${r.findings_reviewed}</td>
              <td>${r.consensus_participation}</td>
              <td>${r.adjudications}</td>
            `;
            reviewerActivityTableBody.appendChild(tr);
          });
        }
      }

      // Populate Provenance Select
      if (provenanceStudySelect && state.allStudies && state.allStudies.length > 0) {
        provenanceStudySelect.innerHTML = '';
        state.allStudies.forEach(s => {
          const opt = document.createElement('option');
          opt.value = s.study_id;
          opt.textContent = `${s.study_id} (${(s.available_views || []).join('+')})`;
          if (s.study_id === state.studyId) opt.selected = true;
          provenanceStudySelect.appendChild(opt);
        });
      }

      // Load Provenance for current study
      loadStudyProvenance(provenanceStudySelect ? provenanceStudySelect.value : state.studyId);

    } catch (err) {
      console.error('Failed to load evaluation analytics:', err);
    }
  }

  async function loadStudyProvenance(studyId) {
    if (!provenanceTimeline) return;
    provenanceTimeline.innerHTML = '<div class="canvas-loader">Loading provenance audit trail...</div>';

    try {
      const resp = await fetch(`/api/studies/${studyId}/provenance`);
      if (!resp.ok) {
        provenanceTimeline.innerHTML = `<div class="empty-hint">Could not load provenance for study ${studyId}.</div>`;
        return;
      }
      const data = await resp.json();
      const stages = data.pipeline_stages || [];

      provenanceTimeline.innerHTML = '';
      stages.forEach(stg => {
        const card = document.createElement('div');
        const stLower = (stg.status || 'pending').toLowerCase();
        card.className = `provenance-step-card ${stLower}`;

        let detailsStr = '';
        if (stg.details) {
          detailsStr = Object.entries(stg.details).map(([k, v]) => `${k}: ${JSON.stringify(v)}`).join(' | ');
        }

        card.innerHTML = `
          <div class="prov-icon-col">${stLower === 'completed' || stLower === 'finalized' ? '✓' : '•'}</div>
          <div class="prov-content-col">
            <div class="prov-header-row">
              <span class="prov-title">${stg.stage_name}</span>
              <span class="status-badge status-${stLower === 'completed' || stLower === 'finalized' ? 'supported' : 'possible'}">${stg.status}</span>
            </div>
            <div class="prov-ts">Timestamp: ${stg.timestamp || 'Pending'}</div>
            <div class="prov-details">${detailsStr}</div>
          </div>
        `;
        provenanceTimeline.appendChild(card);
      });
    } catch (err) {
      provenanceTimeline.innerHTML = `<div class="empty-hint">Error loading provenance: ${err.message}</div>`;
    }
  }

  async function exportDataset(format) {
    try {
      const resp = await fetch(`/api/dataset/export?format=${format}`);
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      if (format === 'text') {
        const text = await resp.text();
        const blob = new Blob([text], { type: 'text/plain' });
        downloadBlob(blob, 'explainable_radiology_dataset_export.txt');
      } else {
        const json = await resp.json();
        const blob = new Blob([JSON.stringify(json, null, 2)], { type: 'application/json' });
        downloadBlob(blob, 'explainable_radiology_dataset_export.json');
      }
    } catch (err) {
      console.error('Dataset export failed:', err);
      alert('Failed to export dataset.');
    }
  }

  async function fetchAndRenderConsensus(studyId) {
    try {
      const resp = await fetch(`/api/studies/${studyId}/consensus`);
      if (!resp.ok) {
        throw new Error(`HTTP ${resp.status}`);
      }
      const data = await resp.json();
      state.consensusSession = data;

      renderConsensusBanner(data);
      renderReviewerRoster(data);
      renderAgreementMetrics(data.agreement_metrics || {});
      renderConsensusMatrix(data);
      renderConsensusQa(data);
      renderAdjudicationSection(data);
      renderConsensusReportForm(data);
    } catch (err) {
      console.warn('Could not fetch consensus session:', err);
    }
  }

  function renderConsensusBanner(session) {
    const status = (session.status || 'collecting').toUpperCase();
    consensusStatusBadge.textContent = status.replace('_', ' ');
    consensusIdBadge.textContent = session.consensus_id || `consensus_${session.study_id}`;
    consensusStudyBadge.innerHTML = `Study: <strong>${session.study_id}</strong> | ID: <code>${session.consensus_id}</code>`;

    if (status === 'UNANIMOUS' || status === 'CONSENSUS_REACHED') {
      consensusStatusBadge.className = 'status-badge status-supported';
    } else if (status === 'MAJORITY' || status === 'ADJUDICATED') {
      consensusStatusBadge.className = 'status-badge status-possible';
    } else if (status === 'ADJUDICATION_REQUIRED') {
      consensusStatusBadge.className = 'status-badge status-needs-review';
    } else if (status === 'FINALIZED') {
      consensusStatusBadge.className = 'status-badge status-supported';
    } else {
      consensusStatusBadge.className = 'status-badge status-neutral';
    }

    const reviewers = session.reviewers || [];
    const completed = reviewers.filter(r => r.review_status === 'completed').length;
    const total = reviewers.length;
    consensusReviewersCountBadge.textContent = `${completed} / ${total} Reviewers Completed`;
    consensusStatusHeaderBadge.textContent = `${status.replace('_', ' ')} (${completed}/${total})`;
  }

  function renderReviewerRoster(session) {
    reviewerRosterList.innerHTML = '';
    const reviewers = session.reviewers || [];
    rosterCountBadge.textContent = `${reviewers.length} Registered`;

    reviewers.forEach(r => {
      const item = document.createElement('div');
      item.className = 'roster-item';
      const isCompleted = r.review_status === 'completed';
      const statusClass = `roster-status-${r.review_status || 'registered'}`;
      const statusText = isCompleted ? '✓ Completed' : (r.review_status === 'in_progress' ? '⏳ In Progress' : 'Registered');

      item.innerHTML = `
        <div class="roster-info">
          <span class="roster-name">${r.display_name || r.id} (${r.id})</span>
          <span class="roster-role">${r.role || 'Clinical Reviewer'}${r.completed_at ? ' • ' + r.completed_at.replace('T', ' ').slice(0, 19) : ''}</span>
        </div>
        <span class="roster-status-chip ${statusClass}">${statusText}</span>
      `;
      reviewerRosterList.appendChild(item);
    });
  }

  function renderAgreementMetrics(metrics) {
    const avg = metrics.average_agreement_ratio;
    metricAvgAgreement.textContent = avg !== null && avg !== undefined ? `${(avg * 100).toFixed(1)}%` : '-';
    metricAvgAgreementSub.textContent = avg !== null ? `Across ${metrics.finding_count || 7} candidate findings` : 'Awaiting minimum reviewers';

    const ck = metrics.cohens_kappa;
    if (ck && ck.value !== null && ck.value !== undefined) {
      metricCohensKappa.textContent = ck.value.toFixed(4);
      metricCohensKappaSub.textContent = `Observed: ${(ck.observed_agreement * 100).toFixed(1)}%, Chance: ${(ck.expected_agreement * 100).toFixed(1)}%`;
    } else {
      metricCohensKappa.textContent = '-';
      metricCohensKappaSub.textContent = 'Applicable when N=2 reviewers';
    }

    const fk = metrics.fleiss_kappa;
    if (fk && fk.value !== null && fk.value !== undefined) {
      metricFleissKappa.textContent = fk.value.toFixed(4);
      metricFleissKappaSub.textContent = `P_o: ${(fk.observed_agreement * 100).toFixed(1)}%, P_e: ${(fk.expected_agreement * 100).toFixed(1)}%`;
    } else {
      metricFleissKappa.textContent = '-';
      metricFleissKappaSub.textContent = 'Applicable when N≥3 reviewers';
    }

    // Qualitative Interpretation
    const activeKappa = (fk && fk.value !== null) ? fk.value : (ck && ck.value !== null ? ck.value : null);
    if (activeKappa !== null) {
      let label = 'Almost Perfect Agreement';
      let badgeClass = 'status-supported';
      if (activeKappa < 0.0) {
        label = 'Poor / Systematic Disagreement';
        badgeClass = 'status-needs-review';
      } else if (activeKappa <= 0.20) {
        label = 'Slight Agreement';
        badgeClass = 'status-neutral';
      } else if (activeKappa <= 0.40) {
        label = 'Fair Agreement';
        badgeClass = 'status-neutral';
      } else if (activeKappa <= 0.60) {
        label = 'Moderate Agreement';
        badgeClass = 'status-possible';
      } else if (activeKappa <= 0.80) {
        label = 'Substantial Agreement';
        badgeClass = 'status-supported';
      }
      kappaInterpretationBadge.textContent = `Kappa Interpretation: ${label}`;
      kappaInterpretationBadge.className = `badge ${badgeClass}`;
      kappaDetailsText.textContent = `Based on Landis & Koch (1977) scale (Score: ${activeKappa.toFixed(4)}).`;
    } else {
      kappaInterpretationBadge.textContent = 'Status: Collecting';
      kappaInterpretationBadge.className = 'badge badge-neutral';
      kappaDetailsText.textContent = 'Requires at least 2 completed reviewer sessions to compute statistical agreement.';
    }
  }

  function renderConsensusMatrix(session) {
    consensusTableBody.innerHTML = '';
    const list = session.finding_consensus || [];
    consensusFindingsBadge.textContent = `${list.length} Findings`;
    const isFinal = session.status === 'finalized';

    list.forEach(fc => {
      const tr = document.createElement('tr');
      const fname = fc.finding;
      const mStat = (fc.machine_status || 'possible').toUpperCase();
      const mScore = fc.machine_score !== undefined ? fc.machine_score.toFixed(4) : '-';

      // Reviewer decision chips
      let chipsHtml = '';
      (fc.reviewer_decisions || []).forEach(rd => {
        const shortName = rd.reviewer_id;
        const dec = rd.decision;
        const decLabel = dec.replace('_', ' ');
        chipsHtml += `<span class="decision-mini-chip chip-${dec}" title="${rd.reviewer_name} (${rd.reviewer_id}): ${decLabel}">${shortName}: ${decLabel}</span>`;
      });
      if (!chipsHtml) chipsHtml = '<span class="text-muted">No decisions yet</span>';

      const consStatus = fc.consensus_status || 'pending_reviews';
      let statusBadgeClass = 'status-neutral';
      if (consStatus === 'unanimous') statusBadgeClass = 'status-supported';
      else if (consStatus === 'majority') statusBadgeClass = 'status-possible';
      else if (consStatus === 'adjudication_required') statusBadgeClass = 'status-needs-review';

      const consDec = fc.consensus_decision ? fc.consensus_decision.replace('_', ' ').toUpperCase() : 'PENDING';
      const agrRatio = fc.agreement_ratio !== null && fc.agreement_ratio !== undefined ? `${(fc.agreement_ratio * 100).toFixed(1)}%` : '-';
      const locSev = `Loc: ${fc.location_consensus || 'unspecified'} | Sev: ${fc.severity_consensus || 'unspecified'}`;

      tr.innerHTML = `
        <td><strong>${fname}</strong></td>
        <td><span class="status-badge status-${fc.machine_status}">${mStat} (${mScore})</span></td>
        <td><div class="reviewer-decisions-cell">${chipsHtml}</div></td>
        <td><span class="status-badge ${statusBadgeClass}">${consStatus.replace('_', ' ').toUpperCase()}</span></td>
        <td><strong>${consDec}</strong></td>
        <td>${agrRatio}</td>
        <td style="font-size: 0.68rem; color: var(--text-secondary);">${locSev}</td>
        <td>
          <button class="btn btn-sm btn-secondary btn-adjudicate-finding" data-finding="${fname}" ${isFinal ? 'disabled' : ''}>
            ${consStatus === 'adjudication_required' ? '⚡ Adjudicate' : 'Adjudicate'}
          </button>
        </td>
      `;

      const adjBtn = tr.querySelector('.btn-adjudicate-finding');
      adjBtn.addEventListener('click', () => {
        openAdjudicationModal('finding', fname, fc.consensus_decision, `Adjudicating consensus for '${fname}' (Distribution: ${JSON.stringify(fc.decision_distribution)})`);
      });

      consensusTableBody.appendChild(tr);
    });
  }

  function renderConsensusQa(session) {
    consensusQaGrid.innerHTML = '';
    const qas = session.qa_consensus || [];
    const isFinal = session.status === 'finalized';

    qas.forEach(q => {
      const card = document.createElement('div');
      card.className = 'consensus-qa-card';
      const isAdjudication = q.agreement_status === 'adjudication_required';

      let answersChips = '';
      (q.reviewer_answers || []).forEach(ra => {
        answersChips += `<span class="decision-mini-chip chip-confirmed_present">${ra.reviewer_id}: ${ra.answer}</span>`;
      });
      if (!answersChips) answersChips = '<span class="text-muted">No answers recorded</span>';

      card.innerHTML = `
        <div class="qa-card-header">
          <strong>${q.question_id} (Level ${q.level})</strong>
          <span class="status-badge ${isAdjudication ? 'status-needs-review' : 'status-possible'}">${(q.agreement_status || 'pending').toUpperCase()}</span>
        </div>
        <div class="qa-question-text">${q.question_text}</div>
        <div class="qa-answers-breakdown">
          <div>Machine Answer: <strong>${(q.machine_answer || 'uncertain').toUpperCase()}</strong></div>
          <div style="display: flex; gap: 4px; flex-wrap: wrap; align-items: center; margin-top: 2px;">
            Reviewers: ${answersChips}
          </div>
          <div>Consensus Answer: <strong style="color: #60a5fa;">${(q.consensus_answer || 'PENDING').toUpperCase()}</strong></div>
        </div>
        <div style="margin-top: 4px; display: flex; justify-content: flex-end;">
          <button class="btn btn-sm btn-secondary btn-adjudicate-qa" ${isFinal ? 'disabled' : ''}>Adjudicate QA</button>
        </div>
      `;

      const adjBtn = card.querySelector('.btn-adjudicate-qa');
      adjBtn.addEventListener('click', () => {
        openAdjudicationModal('qa', q.question_id, q.consensus_answer, `Adjudicating question '${q.question_id}' (${q.finding})`);
      });

      consensusQaGrid.appendChild(card);
    });
  }

  function renderAdjudicationSection(session) {
    const adj = session.adjudication || {};
    const pending = adj.items_requiring_adjudication || [];
    const records = adj.adjudication_records || [];
    const isFinal = session.status === 'finalized';

    const unresolved = pending.filter(p => p.status === 'pending');
    pendingAdjudicationsBadge.textContent = `${unresolved.length} Pending`;

    if (unresolved.length === 0) {
      pendingAdjudicationItemsList.innerHTML = '<div class="empty-hint">✓ All disagreements resolved. No pending adjudications.</div>';
    } else {
      pendingAdjudicationItemsList.innerHTML = '';
      unresolved.forEach(item => {
        const row = document.createElement('div');
        row.className = 'pending-item-row';
        row.innerHTML = `
          <div>
            <strong>[${item.target_type.toUpperCase()}] ${item.target_id}</strong>: ${item.issue}
          </div>
          <button class="btn btn-sm btn-primary btn-resolve-adj" ${isFinal ? 'disabled' : ''}>Adjudicate</button>
        `;
        row.querySelector('.btn-resolve-adj').addEventListener('click', () => {
          openAdjudicationModal(item.target_type, item.target_id, null, item.issue);
        });
        pendingAdjudicationItemsList.appendChild(row);
      });
    }

    adjudicationRecordsBody.innerHTML = '';
    if (records.length === 0) {
      adjudicationRecordsBody.innerHTML = '<tr><td colspan="5" class="empty-hint">No adjudication records in this session.</td></tr>';
    } else {
      records.forEach(r => {
        const tr = document.createElement('tr');
        const timeStr = r.timestamp ? r.timestamp.replace('T', ' ').slice(0, 19) : '-';
        tr.innerHTML = `
          <td>${timeStr}</td>
          <td>${r.adjudicator_id}</td>
          <td><strong>${r.target_type}:${r.target_id}</strong></td>
          <td><strong style="color: #34d399;">${r.decision}</strong></td>
          <td>${r.reason}</td>
        `;
        adjudicationRecordsBody.appendChild(tr);
      });
    }
  }

  function renderConsensusReportForm(session) {
    const rep = session.consensus_report || {};
    const isFinal = session.status === 'finalized' || rep.finalized;

    consensusReportFinalBadge.textContent = isFinal ? 'FINALIZED' : 'DRAFT';
    consensusReportFinalBadge.className = `status-badge ${isFinal ? 'status-supported' : 'status-possible'}`;

    resetConsensusDraftBtn.disabled = isFinal;
    saveConsensusDraftBtn.disabled = isFinal;
    finalizeConsensusBtn.disabled = isFinal;
    consensusImpressionEditor.disabled = isFinal;
    consensusReviewerSummary.disabled = isFinal;
    consensusAgreementSummary.disabled = isFinal;
    consensusCommentaryEditor.disabled = isFinal;

    consensusFindingsList.innerHTML = '';
    const findings = rep.final_findings || [];
    findings.forEach((f, idx) => {
      const row = document.createElement('div');
      row.className = 'consensus-finding-row';
      row.innerHTML = `
        <span class="consensus-finding-name">${f.finding}</span>
        <span class="status-badge status-${f.status || 'possible'}">${(f.status || 'possible').toUpperCase()}</span>
        <input type="text" class="consensus-finding-text" data-idx="${idx}" value="${f.statement || ''}" placeholder="Consensus statement..." ${isFinal ? 'disabled' : ''}>
      `;
      consensusFindingsList.appendChild(row);
    });

    consensusImpressionEditor.value = Array.isArray(rep.final_impression) ? rep.final_impression.join('\n') : (rep.final_impression || '');
    consensusReviewerSummary.value = rep.reviewer_summary || '';
    consensusAgreementSummary.value = rep.agreement_summary || '';
    consensusCommentaryEditor.value = rep.reviewer_comment || '';
  }

  // --- Adjudication Modal Logic ---
  function openAdjudicationModal(targetType, targetId, currentDecision, issueText) {
    state.pendingAdjudicationItem = { target_type: targetType, target_id: targetId };
    adjudicationModalTitle.textContent = `Adjudicate ${targetType.toUpperCase()}: ${targetId}`;
    adjudicationModalDesc.textContent = issueText || `Provide a definitive binding decision for ${targetId}.`;
    adjudicationTargetInfo.textContent = `Target: ${targetType} -> ${targetId}`;
    adjudicationReasonInput.value = '';
    adjudicationFeedback.textContent = '';
    adjudicationModalBackdrop.style.display = 'flex';
  }

  function closeAdjudicationModal() {
    adjudicationModalBackdrop.style.display = 'none';
    state.pendingAdjudicationItem = null;
  }

  async function submitAdjudication() {
    if (!state.pendingAdjudicationItem) return;
    const target = state.pendingAdjudicationItem;
    const dec = adjudicationDecisionSelect.value;
    const adjId = adjudicatorIdInput.value.trim() || 'lead_adjudicator';
    const reason = adjudicationReasonInput.value.trim();

    if (!reason) {
      adjudicationFeedback.textContent = 'Error: Mandatory justification/reason is required.';
      adjudicationFeedback.className = 'report-action-feedback feedback-error';
      return;
    }

    submitAdjudicationBtn.disabled = true;
    submitAdjudicationBtn.textContent = 'Saving...';

    try {
      const resp = await fetch(`/api/studies/${state.studyId}/consensus/adjudicate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          target_type: target.target_type,
          target_id: target.target_id,
          decision: dec,
          adjudicator_id: adjId,
          reason: reason
        })
      });

      const resData = await resp.json();
      if (!resp.ok) {
        throw new Error(resData.error || `HTTP ${resp.status}`);
      }

      closeAdjudicationModal();
      await fetchAndRenderConsensus(state.studyId);
      alert('Adjudication recorded successfully!');
    } catch (err) {
      adjudicationFeedback.textContent = `Error: ${err.message}`;
      adjudicationFeedback.className = 'report-action-feedback feedback-error';
    } finally {
      submitAdjudicationBtn.disabled = false;
      submitAdjudicationBtn.textContent = 'Save Binding Adjudication';
    }
  }

  // --- Register Reviewer Modal Logic ---
  function openRegisterModal() {
    newReviewerId.value = '';
    newReviewerName.value = '';
    registerFeedback.textContent = '';
    registerReviewerModalBackdrop.style.display = 'flex';
  }

  function closeRegisterModal() {
    registerReviewerModalBackdrop.style.display = 'none';
  }

  async function submitRegisterReviewer() {
    const rId = newReviewerId.value.trim();
    const rName = newReviewerName.value.trim();
    const rRole = newReviewerRole.value;

    if (!rId || !rName) {
      registerFeedback.textContent = 'Error: Reviewer ID and display name are required.';
      registerFeedback.className = 'report-action-feedback feedback-error';
      return;
    }

    submitRegisterReviewerBtn.disabled = true;
    submitRegisterReviewerBtn.textContent = 'Registering...';

    try {
      const resp = await fetch(`/api/studies/${state.studyId}/consensus/reviewers`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          reviewer_id: rId,
          display_name: rName,
          role: rRole
        })
      });

      const resData = await resp.json();
      if (!resp.ok) {
        throw new Error(resData.error || `HTTP ${resp.status}`);
      }

      // Add to dropdown if not present
      if (!Array.from(activeReviewerSelect.options).some(o => o.value === rId)) {
        const opt = document.createElement('option');
        opt.value = rId;
        opt.textContent = `${rName} (${rId})`;
        activeReviewerSelect.appendChild(opt);
      }

      closeRegisterModal();
      await fetchAndRenderConsensus(state.studyId);
      alert(`Reviewer '${rName}' registered successfully!`);
    } catch (err) {
      registerFeedback.textContent = `Error: ${err.message}`;
      registerFeedback.className = 'report-action-feedback feedback-error';
    } finally {
      submitRegisterReviewerBtn.disabled = false;
      submitRegisterReviewerBtn.textContent = 'Register Reviewer';
    }
  }

  // --- Consensus Report Actions ---
  async function handleResetConsensusDraft() {
    if (!confirm('Reset consensus report draft from synthesized consensus decisions?')) return;
    await fetchAndRenderConsensus(state.studyId);
    consensusActionFeedback.textContent = '✓ Consensus report draft refreshed.';
    consensusActionFeedback.className = 'report-action-feedback feedback-success';
  }

  async function handleSaveConsensusDraft() {
    consensusActionFeedback.textContent = 'Saving consensus draft...';
    consensusActionFeedback.className = 'report-action-feedback';

    const findings = [];
    const rows = consensusFindingsList.querySelectorAll('.consensus-finding-row');
    rows.forEach(row => {
      const name = row.querySelector('.consensus-finding-name').textContent.trim();
      const status = row.querySelector('.status-badge').textContent.trim().toLowerCase();
      const statement = row.querySelector('.consensus-finding-text').value.trim();
      findings.push({ finding: name, statement: statement, status: status, location: 'unspecified', severity: 'unspecified' });
    });

    const impression = consensusImpressionEditor.value.trim().split('\n').map(s => s.trim()).filter(Boolean);
    const revSum = consensusReviewerSummary.value.trim();
    const agrSum = consensusAgreementSummary.value.trim();
    const comm = consensusCommentaryEditor.value.trim();

    try {
      const resp = await fetch(`/api/studies/${state.studyId}/consensus/report`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          final_findings: findings,
          final_impression: impression,
          reviewer_summary: revSum,
          agreement_summary: agrSum,
          reviewer_comment: comm
        })
      });

      if (!resp.ok) {
        const errJ = await resp.json();
        throw new Error(errJ.error || `HTTP ${resp.status}`);
      }

      await fetchAndRenderConsensus(state.studyId);
      consensusActionFeedback.textContent = '✓ Consensus report draft saved successfully.';
      consensusActionFeedback.className = 'report-action-feedback feedback-success';
    } catch (err) {
      consensusActionFeedback.textContent = `Error: ${err.message}`;
      consensusActionFeedback.className = 'report-action-feedback feedback-error';
    }
  }

  async function handleFinalizeConsensus() {
    if (!confirm('Finalizing consensus report will permanently lock the multi-reviewer evaluation and consensus synthesis. Proceed?')) {
      return;
    }

    consensusActionFeedback.textContent = 'Finalizing consensus report...';
    consensusActionFeedback.className = 'report-action-feedback';

    // Save draft first
    await handleSaveConsensusDraft();

    try {
      const resp = await fetch(`/api/studies/${state.studyId}/consensus/finalize`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          adjudicator_id: 'lead_adjudicator',
          reason: 'All reviewer agreements verified and multi-reviewer consensus finalized.'
        })
      });

      const resData = await resp.json();
      if (!resp.ok || !resData.success) {
        const issues = resData.validation_errors || [resData.error];
        throw new Error(issues.join('; '));
      }

      await fetchAndRenderConsensus(state.studyId);
      consensusActionFeedback.textContent = '✓ Consensus report successfully finalized and locked!';
      consensusActionFeedback.className = 'report-action-feedback feedback-success';
      alert('Consensus report finalized successfully!');
    } catch (err) {
      consensusActionFeedback.textContent = `Finalization Failed: ${err.message}`;
      consensusActionFeedback.className = 'report-action-feedback feedback-error';
      alert(`Cannot Finalize Consensus:\n${err.message}`);
    }
  }

  async function exportConsensus(format) {
    try {
      const resp = await fetch(`/api/studies/${state.studyId}/consensus/export?format=${format}`);
      if (!resp.ok) throw new Error(`Export failed with HTTP ${resp.status}`);

      if (format === 'text') {
        const text = await resp.text();
        const blob = new Blob([text], { type: 'text/plain' });
        downloadBlob(blob, `${state.studyId}_consensus_report.txt`);
      } else {
        const json = await resp.json();
        const blob = new Blob([JSON.stringify(json, null, 2)], { type: 'application/json' });
        downloadBlob(blob, `${state.studyId}_consensus_package.json`);
      }
    } catch (err) {
      console.error('Export error:', err);
      alert('Failed to export consensus report.');
    }
  }

  // --- Attach Phase 1.2 & 1.3 Event Listeners ---
  if (activeReviewerSelect) {
    activeReviewerSelect.addEventListener('change', async (e) => {
      state.activeReviewerId = e.target.value;
      await fetchReviewSession(state.studyId);
      renderHeaderMeta();
      renderFindingPills();
      populateReviewerFindingForm();
      renderQAReviews();
      renderReportReview();
      renderComparisonMetrics();
      renderAuditTrail();
      updateProgressBar();
      if (state.activeTab === 'consensus') {
        fetchAndRenderConsensus(state.studyId);
      }
    });
  }

  if (refreshConsensusBtn) refreshConsensusBtn.addEventListener('click', () => fetchAndRenderConsensus(state.studyId));
  if (registerReviewerBtn) registerReviewerBtn.addEventListener('click', openRegisterModal);
  if (closeRegisterModalBtn) closeRegisterModalBtn.addEventListener('click', closeRegisterModal);
  if (cancelRegisterModalBtn) cancelRegisterModalBtn.addEventListener('click', closeRegisterModal);
  if (submitRegisterReviewerBtn) submitRegisterReviewerBtn.addEventListener('click', submitRegisterReviewer);

  if (closeAdjudicationModalBtn) closeAdjudicationModalBtn.addEventListener('click', closeAdjudicationModal);
  if (cancelAdjudicationBtn) cancelAdjudicationBtn.addEventListener('click', closeAdjudicationModal);
  if (submitAdjudicationBtn) submitAdjudicationBtn.addEventListener('click', submitAdjudication);

  if (resetConsensusDraftBtn) resetConsensusDraftBtn.addEventListener('click', handleResetConsensusDraft);
  if (saveConsensusDraftBtn) saveConsensusDraftBtn.addEventListener('click', handleSaveConsensusDraft);
  if (finalizeConsensusBtn) finalizeConsensusBtn.addEventListener('click', handleFinalizeConsensus);

  if (exportConsensusJsonBtn) exportConsensusJsonBtn.addEventListener('click', () => exportConsensus('json'));
  if (exportConsensusTextBtn) exportConsensusTextBtn.addEventListener('click', () => exportConsensus('text'));

  if (btnConfirmPresent) btnConfirmPresent.addEventListener('click', () => updateDecisionButtonSelection('confirmed_present'));
  if (btnConfirmAbsent) btnConfirmAbsent.addEventListener('click', () => updateDecisionButtonSelection('confirmed_absent'));
  if (btnUncertain) btnUncertain.addEventListener('click', () => updateDecisionButtonSelection('uncertain'));
  if (btnNeedsReview) btnNeedsReview.addEventListener('click', () => updateDecisionButtonSelection('needs_review'));
  if (saveFindingReviewBtn) saveFindingReviewBtn.addEventListener('click', handleSaveFindingDecision);

  if (resetReportBtn) resetReportBtn.addEventListener('click', handleResetReport);
  if (saveDraftBtn) saveDraftBtn.addEventListener('click', handleSaveReportDraft);
  if (validateReportBtn) validateReportBtn.addEventListener('click', handleValidateReview);
  if (finalizeReportBtn) finalizeReportBtn.addEventListener('click', handleFinalizeReview);
  if (validateReviewBtn) validateReviewBtn.addEventListener('click', handleValidateReview);
  if (finalizeReviewBtn) finalizeReviewBtn.addEventListener('click', handleFinalizeReview);

  if (exportTextBtn) exportTextBtn.addEventListener('click', () => exportReview('text'));
  if (exportJsonBtn) exportJsonBtn.addEventListener('click', () => exportReview('json'));

  if (studySelect) studySelect.addEventListener('change', (e) => loadStudy(e.target.value));
  if (studySearchInput) studySearchInput.addEventListener('input', filterStudyDropdown);
  if (studyFilterSelect) studyFilterSelect.addEventListener('change', filterStudyDropdown);

  if (batchModalBtn) batchModalBtn.addEventListener('click', () => { batchModalBackdrop.style.display = 'flex'; });
  if (closeBatchModalBtn) closeBatchModalBtn.addEventListener('click', () => { batchModalBackdrop.style.display = 'none'; });
  if (triggerBatchBtn) triggerBatchBtn.addEventListener('click', handleTriggerBatch);

  if (resetZoomBtn) resetZoomBtn.addEventListener('click', resetView);
  if (zoomInBtn) zoomInBtn.addEventListener('click', () => adjustZoom(0.2));
  if (zoomOutBtn) zoomOutBtn.addEventListener('click', () => adjustZoom(-0.2));

  if (syncToggle) syncToggle.addEventListener('change', (e) => { state.isSync = e.target.checked; });
  if (opacitySlider) {
    opacitySlider.addEventListener('input', (e) => {
      state.opacity = parseInt(e.target.value, 10) / 100;
      opacityVal.textContent = `${e.target.value}%`;
      renderBothCanvases();
    });
  }

  modeBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      modeBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      state.viewMode = btn.dataset.mode;
      renderBothCanvases();
    });
  });

  // --- Attach Phase 1.4 Event Listeners ---

  if (refreshQueueBtn) refreshQueueBtn.addEventListener('click', loadDatasetQueue);
  if (queueSearchInput) queueSearchInput.addEventListener('input', loadDatasetQueue);
  if (queueStatusFilter) queueStatusFilter.addEventListener('change', loadDatasetQueue);
  if (queueConsensusFilter) queueConsensusFilter.addEventListener('change', loadDatasetQueue);
  if (queueSortBy) queueSortBy.addEventListener('change', loadDatasetQueue);
  if (exportDatasetJsonBtn) exportDatasetJsonBtn.addEventListener('click', () => exportDataset('json'));
  if (exportDatasetTextBtn) exportDatasetTextBtn.addEventListener('click', () => exportDataset('text'));

  if (loadProvenanceBtn) loadProvenanceBtn.addEventListener('click', () => loadStudyProvenance(provenanceStudySelect.value));
  if (provenanceStudySelect) provenanceStudySelect.addEventListener('change', (e) => loadStudyProvenance(e.target.value));

  // --- Attach Phase 1.5 Research Evaluation Listeners ---
  if (refreshExperimentsBtn) refreshExperimentsBtn.addEventListener('click', loadResearchEvaluation);
  if (openNewExperimentModalBtn) openNewExperimentModalBtn.addEventListener('click', openCreateExpModal);
  if (closeCreateExpModalBtn) closeCreateExpModalBtn.addEventListener('click', closeCreateExpModal);
  if (cancelCreateExpModalBtn) cancelCreateExpModalBtn.addEventListener('click', closeCreateExpModal);
  if (submitCreateExperimentBtn) submitCreateExperimentBtn.addEventListener('click', submitCreateExperiment);

  if (openNewSnapshotModalBtn) openNewSnapshotModalBtn.addEventListener('click', openCreateSnapshotModal);
  if (closeCreateSnapModalBtn) closeCreateSnapModalBtn.addEventListener('click', closeCreateSnapshotModal);
  if (cancelCreateSnapModalBtn) cancelCreateSnapModalBtn.addEventListener('click', closeCreateSnapshotModal);
  if (submitCreateSnapshotBtn) submitCreateSnapshotBtn.addEventListener('click', submitCreateSnapshot);

  if (selectAllExperimentsCheckbox) selectAllExperimentsCheckbox.addEventListener('change', handleSelectAllExperiments);
  if (compareSelectedExperimentsBtn) compareSelectedExperimentsBtn.addEventListener('click', triggerCompareExperiments);
  if (closeComparisonBtn) closeComparisonBtn.addEventListener('click', () => {
    if (experimentComparisonSection) experimentComparisonSection.style.display = 'none';
  });

  if (exportExpJsonBtn) exportExpJsonBtn.addEventListener('click', () => exportExperiment('json'));
  if (exportExpTextBtn) exportExpTextBtn.addEventListener('click', () => exportExperiment('text'));

  // Live fingerprint calculation on input
  [newExpModelVersion, newExpTargetLayer, newExpQaThreshold, newExpQaTopK].forEach(el => {
    if (el) el.addEventListener('input', updateLiveFingerprintPreview);
  });

  // ============================================================
  // PHASE 1.5 — RESEARCH EVALUATION IMPLEMENTATION FUNCTIONS
  // ============================================================

  async function loadResearchEvaluation() {
    try {
      // 1. Fetch snapshots
      const snapResp = await fetch('/api/experiments/snapshots');
      if (snapResp.ok) {
        const snapData = await snapResp.json();
        state.snapshotsList = snapData.snapshots || [];
        renderSnapshotsGrid(state.snapshotsList);
        populateSnapshotSelect(state.snapshotsList);
      }

      // 2. Fetch experiments
      const expResp = await fetch('/api/experiments');
      if (expResp.ok) {
        const expData = await expResp.json();
        state.experimentsList = expData.experiments || [];
        renderExperimentsTable(state.experimentsList);
      }
    } catch (err) {
      console.error('Failed to load research evaluation data:', err);
    }
  }

  function renderSnapshotsGrid(snapshots) {
    if (!snapshotsGridContainer) return;
    snapshotsGridContainer.innerHTML = '';
    if (snapshotTotalCount) snapshotTotalCount.textContent = snapshots.length;

    if (snapshots.length === 0) {
      snapshotsGridContainer.innerHTML = '<div class="empty-hint">No dataset snapshots created yet. Click "Create Dataset Snapshot" to begin.</div>';
      return;
    }

    snapshots.forEach(s => {
      const card = document.createElement('div');
      card.className = 'snapshot-card';
      const shortHash = s.manifest_hash ? s.manifest_hash.substring(0, 12) + '...' : '-';
      const count = s.study_count !== undefined ? s.study_count : (s.study_ids ? s.study_ids.length : 0);

      card.innerHTML = `
        <div class="snapshot-header">
          <strong>${s.name || s.snapshot_id}</strong>
          <span class="badge badge-default">${count} Studies</span>
        </div>
        <div class="snapshot-desc">${s.description || 'No description provided.'}</div>
        <div class="snapshot-meta-row">
          <span>ID: <code>${s.snapshot_id}</code></span>
          <span>Hash: <code class="sha-hash" title="${s.manifest_hash}">${shortHash}</code></span>
        </div>
        <div class="snapshot-meta-row">
          <span>By: ${s.created_by || 'curator'}</span>
          <span>${s.created_at ? s.created_at.substring(0, 10) : ''}</span>
        </div>
      `;
      snapshotsGridContainer.appendChild(card);
    });
  }

  function populateSnapshotSelect(snapshots) {
    if (!newExpSnapshotSelect) return;
    newExpSnapshotSelect.innerHTML = '';
    if (snapshots.length === 0) {
      const opt = document.createElement('option');
      opt.value = '';
      opt.textContent = '-- No snapshots available (Create one first) --';
      newExpSnapshotSelect.appendChild(opt);
      return;
    }

    snapshots.forEach(s => {
      const opt = document.createElement('option');
      opt.value = s.snapshot_id;
      const count = s.study_count !== undefined ? s.study_count : (s.study_ids ? s.study_ids.length : 0);
      opt.textContent = `${s.name || s.snapshot_id} (${count} studies)`;
      newExpSnapshotSelect.appendChild(opt);
    });
  }

  function renderExperimentsTable(experiments) {
    if (!experimentsTableBody) return;
    experimentsTableBody.innerHTML = '';
    if (experimentsTotalCount) experimentsTotalCount.textContent = experiments.length;

    if (experiments.length === 0) {
      experimentsTableBody.innerHTML = '<tr><td colspan="10" class="empty-hint">No research experiments configured. Click "New Experiment" to create one.</td></tr>';
      return;
    }

    experiments.forEach(exp => {
      const tr = document.createElement('tr');
      const isSelected = state.selectedExperimentIds.has(exp.experiment_id);
      const st = (exp.status || 'CREATED').toUpperCase();
      let statusClass = 'status-neutral';
      if (st === 'COMPLETED') statusClass = 'status-supported';
      else if (st === 'RUNNING') statusClass = 'status-possible';
      else if (st === 'ARCHIVED') statusClass = 'status-needs-review';

      const fp = exp.configuration_fingerprint || {};
      const modelShort = `${fp.model_name || 'DenseNet-121'}`;
      const layerShort = (fp.target_layer || 'norm5').replace('model.features.', '');
      const qaShort = `T=${fp.qa_threshold || 0.15}, K=${fp.qa_top_k || 5}`;
      const snapShort = exp.snapshot_id ? exp.snapshot_id.substring(0, 16) : '-';
      const createdStr = exp.created_at ? exp.created_at.substring(0, 16).replace('T', ' ') : '-';
      const studyCount = exp.dataset_size !== undefined ? `${exp.dataset_size} studies` : '-';

      tr.innerHTML = `
        <td><input type="checkbox" class="exp-select-checkbox" data-exp-id="${exp.experiment_id}" ${isSelected ? 'checked' : ''}></td>
        <td><code>${exp.experiment_id}</code></td>
        <td><strong>${exp.name || '-'}</strong></td>
        <td><code>${snapShort}</code></td>
        <td><span class="info-tag">${modelShort} / ${layerShort}</span></td>
        <td><span style="font-size: 0.72rem;">${qaShort}</span></td>
        <td><span class="status-badge ${statusClass}">${st}</span></td>
        <td>${studyCount}</td>
        <td style="font-size: 0.72rem; color: var(--text-muted);">${createdStr}</td>
        <td>
          <div class="table-actions">
            ${st === 'CREATED' ? `<button class="btn btn-sm btn-primary btn-run-exp" data-exp-id="${exp.experiment_id}">▶ Run</button>` : ''}
            <button class="btn btn-sm btn-secondary btn-view-exp" data-exp-id="${exp.experiment_id}">👁 View</button>
            ${st === 'COMPLETED' ? `<button class="btn btn-sm btn-secondary btn-archive-exp" data-exp-id="${exp.experiment_id}">📦 Archive</button>` : ''}
          </div>
        </td>
      `;

      // Checkbox listener
      const cb = tr.querySelector('.exp-select-checkbox');
      cb.addEventListener('change', (e) => {
        if (e.target.checked) {
          state.selectedExperimentIds.add(exp.experiment_id);
        } else {
          state.selectedExperimentIds.delete(exp.experiment_id);
        }
        updateSelectedExperimentsUI();
      });

      // Action listeners
      const runBtn = tr.querySelector('.btn-run-exp');
      if (runBtn) {
        runBtn.addEventListener('click', () => triggerRunExperiment(exp.experiment_id));
      }

      const viewBtn = tr.querySelector('.btn-view-exp');
      if (viewBtn) {
        viewBtn.addEventListener('click', () => showExperimentDetails(exp.experiment_id));
      }

      const archBtn = tr.querySelector('.btn-archive-exp');
      if (archBtn) {
        archBtn.addEventListener('click', () => triggerArchiveExperiment(exp.experiment_id));
      }

      experimentsTableBody.appendChild(tr);
    });

    updateSelectedExperimentsUI();
  }

  function handleSelectAllExperiments(e) {
    const isChecked = e.target.checked;
    state.selectedExperimentIds.clear();
    if (isChecked) {
      state.experimentsList.forEach(exp => state.selectedExperimentIds.add(exp.experiment_id));
    }
    const checkboxes = experimentsTableBody.querySelectorAll('.exp-select-checkbox');
    checkboxes.forEach(cb => { cb.checked = isChecked; });
    updateSelectedExperimentsUI();
  }

  function updateSelectedExperimentsUI() {
    const count = state.selectedExperimentIds.size;
    if (selectedExpCount) selectedExpCount.textContent = count;
    if (compareSelectedExperimentsBtn) {
      compareSelectedExperimentsBtn.disabled = (count < 2);
    }
  }

  async function showExperimentDetails(experimentId) {
    state.activeExperimentId = experimentId;
    if (experimentDetailSection) experimentDetailSection.style.display = 'block';
    experimentDetailSection.scrollIntoView({ behavior: 'smooth' });

    if (detailExpId) detailExpId.textContent = experimentId;

    try {
      const resp = await fetch(`/api/experiments/${experimentId}`);
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const exp = await resp.json();

      if (detailExpName) detailExpName.textContent = exp.name || '-';
      if (detailExpDesc) detailExpDesc.textContent = exp.description || 'No description';
      if (detailExpStatusBadge) {
        const st = (exp.status || 'CREATED').toUpperCase();
        detailExpStatusBadge.textContent = st;
        detailExpStatusBadge.className = `status-badge ${st === 'COMPLETED' ? 'status-supported' : (st === 'RUNNING' ? 'status-possible' : 'status-neutral')}`;
      }
      if (detailExpSnapshotId) detailExpSnapshotId.textContent = exp.snapshot_id || '-';
      if (detailExpCreatedBy) detailExpCreatedBy.textContent = exp.created_by || '-';
      if (detailExpCreatedAt) detailExpCreatedAt.textContent = exp.created_at ? exp.created_at.substring(0, 19).replace('T', ' ') : '-';
      if (detailExpCompletedAt) detailExpCompletedAt.textContent = exp.completed_at ? exp.completed_at.substring(0, 19).replace('T', ' ') : 'Pending Execution';

      const fp = exp.configuration_fingerprint || {};
      if (detailExpModelName) detailExpModelName.textContent = fp.model_name || 'TorchXRayVision DenseNet-121';
      if (detailExpModelVersion) detailExpModelVersion.textContent = fp.model_version || 'densenet121-res224-all';
      if (detailExpTargetLayer) detailExpTargetLayer.textContent = fp.target_layer || 'model.features.norm5';
      if (detailExpQAParams) detailExpQAParams.textContent = `Threshold: ${fp.qa_threshold !== undefined ? fp.qa_threshold : 0.15}, Top-K: ${fp.qa_top_k || 5}`;
      if (detailExpLLMParams) detailExpLLMParams.textContent = `${fp.llm_provider || 'Mock / Gemini'} (${fp.llm_model || 'gemini-1.5-flash'})`;
      if (detailExpFingerprintHash) detailExpFingerprintHash.textContent = fp.fingerprint_hash || '-';

      // Render Metrics
      renderExperimentMetrics(exp.metrics || {});

      // Load Provenance
      loadExperimentProvenance(experimentId);

    } catch (err) {
      console.error('Failed to load experiment details:', err);
    }
  }

  function renderExperimentMetrics(metrics) {
    const cov = metrics.review_coverage || {};
    if (expMetricReviewCov) {
      const pct = cov.review_completion_rate !== undefined ? `${(cov.review_completion_rate * 100).toFixed(1)}%` : '-';
      expMetricReviewCov.textContent = pct;
    }
    if (expMetricReviewSub) {
      expMetricReviewSub.textContent = `${cov.completed_studies || 0} / ${cov.total_studies || 0} studies completed`;
    }

    const con = metrics.consensus_coverage || {};
    if (expMetricConsensusCov) {
      const pct = con.consensus_coverage_rate !== undefined ? `${(con.consensus_coverage_rate * 100).toFixed(1)}%` : '-';
      expMetricConsensusCov.textContent = pct;
    }
    if (expMetricConsensusSub) {
      expMetricConsensusSub.textContent = `${con.finalized_consensus_reports || 0} finalized (${con.unanimous_findings || 0} unanimous)`;
    }

    const aggr = metrics.machine_reviewer_agreement || {};
    if (expMetricMachineAggr) {
      const ratio = aggr.overall_agreement_ratio !== undefined ? `${(aggr.overall_agreement_ratio * 100).toFixed(1)}%` : '-';
      expMetricMachineAggr.textContent = ratio;
    }

    const rel = metrics.inter_rater_reliability || {};
    if (expMetricKappa) {
      const ck = rel.cohens_kappa ? rel.cohens_kappa.value : null;
      const fk = rel.fleiss_kappa ? rel.fleiss_kappa.value : null;
      const cStr = ck !== null && ck !== undefined ? `κ_c=${ck.toFixed(3)}` : 'κ_c=-';
      const fStr = fk !== null && fk !== undefined ? `κ_f=${fk.toFixed(3)}` : 'κ_f=-';
      expMetricKappa.textContent = `${cStr} | ${fStr}`;
    }
    if (expMetricKappaSub) {
      const ck = rel.cohens_kappa ? rel.cohens_kappa.interpretation : 'Pending';
      expMetricKappaSub.textContent = `Interp: ${ck || 'Awaiting 2+ reviewers'}`;
    }

    const expn = metrics.explainability_coverage || {};
    if (expMetricExplainCov) {
      const pct = expn.heatmap_generation_rate !== undefined ? `${(expn.heatmap_generation_rate * 100).toFixed(1)}%` : '-';
      expMetricExplainCov.textContent = pct;
    }
    if (expMetricExplainSub) {
      expMetricExplainSub.textContent = `${expn.grounded_findings_count || 0} heatmaps across ${expn.total_findings_evaluated || 0} candidate findings`;
    }

    const corr = metrics.human_corrections || {};
    if (expMetricHumanCorr) {
      expMetricHumanCorr.textContent = `${corr.total_corrections || 0} edits`;
    }
    if (expMetricHumanCorrSub) {
      expMetricHumanCorrSub.textContent = `Status: ${corr.status_modifications || 0}, Severity: ${corr.severity_modifications || 0}, Report: ${corr.report_edits_count || 0}`;
    }
  }

  async function loadExperimentProvenance(experimentId) {
    if (!expProvenanceTimeline) return;
    expProvenanceTimeline.innerHTML = '<div class="canvas-loader">Loading experiment provenance...</div>';

    try {
      const resp = await fetch(`/api/experiments/${experimentId}/provenance`);
      if (!resp.ok) {
        expProvenanceTimeline.innerHTML = '<div class="empty-hint">Provenance trail not yet generated for this experiment.</div>';
        return;
      }
      const data = await resp.json();
      const stages = data.stages || [];

      expProvenanceTimeline.innerHTML = '';
      if (stages.length === 0) {
        expProvenanceTimeline.innerHTML = '<div class="empty-hint">No pipeline stages recorded yet.</div>';
        return;
      }

      stages.forEach(stg => {
        const card = document.createElement('div');
        const stLower = (stg.status || 'pending').toLowerCase();
        card.className = `provenance-step-card ${stLower}`;

        let detailsStr = '';
        if (stg.details) {
          detailsStr = Object.entries(stg.details).map(([k, v]) => `${k}: ${JSON.stringify(v)}`).join(' | ');
        }

        card.innerHTML = `
          <div class="prov-icon-col">${stLower === 'completed' ? '✓' : (stLower === 'failed' ? '✗' : '•')}</div>
          <div class="prov-content-col">
            <div class="prov-header-row">
              <span class="prov-title">[Stage ${stg.stage_number || '•'}] ${stg.stage_name}</span>
              <span class="status-badge status-${stLower === 'completed' ? 'supported' : (stLower === 'failed' ? 'needs-review' : 'possible')}">${stg.status}</span>
            </div>
            <div class="prov-ts">Timestamp: ${stg.timestamp ? stg.timestamp.substring(0, 19).replace('T', ' ') : 'Pending'}</div>
            <div class="prov-details">${detailsStr}</div>
          </div>
        `;
        expProvenanceTimeline.appendChild(card);
      });
    } catch (err) {
      expProvenanceTimeline.innerHTML = `<div class="empty-hint">Error loading provenance: ${err.message}</div>`;
    }
  }

  async function triggerRunExperiment(experimentId) {
    if (!confirm(`Execute research experiment '${experimentId}' over its bound dataset snapshot?`)) return;

    try {
      const resp = await fetch(`/api/experiments/${experimentId}/run`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ executed_by: 'researcher_01' })
      });

      const resData = await resp.json();
      if (!resp.ok || !resData.success) {
        throw new Error(resData.error || `HTTP ${resp.status}`);
      }

      alert(`Experiment '${experimentId}' executed successfully!`);
      await loadResearchEvaluation();
      showExperimentDetails(experimentId);
    } catch (err) {
      console.error('Run experiment failed:', err);
      alert(`Cannot Run Experiment:\n${err.message}`);
    }
  }

  async function triggerArchiveExperiment(experimentId) {
    if (!confirm(`Archive experiment '${experimentId}'? It will be marked as ARCHIVED and preserved as a permanent historical record.`)) return;

    try {
      const resp = await fetch(`/api/experiments/${experimentId}/archive`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ reason: 'Manual archival from Research Evaluation dashboard' })
      });

      const resData = await resp.json();
      if (!resp.ok || !resData.success) {
        throw new Error(resData.error || `HTTP ${resp.status}`);
      }

      await loadResearchEvaluation();
      showExperimentDetails(experimentId);
    } catch (err) {
      console.error('Archive experiment failed:', err);
      alert(`Cannot Archive Experiment:\n${err.message}`);
    }
  }

  async function triggerCompareExperiments() {
    const ids = Array.from(state.selectedExperimentIds);
    if (ids.length < 2) {
      alert('Please select at least 2 experiments using the checkboxes to perform side-by-side comparison.');
      return;
    }

    try {
      const resp = await fetch('/api/experiments/compare', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ experiment_ids: ids })
      });

      const resData = await resp.json();
      if (!resp.ok) {
        throw new Error(resData.error || `HTTP ${resp.status}`);
      }

      renderExperimentComparison(resData);
    } catch (err) {
      console.error('Comparison error:', err);
      alert(`Cannot Compare Experiments:\n${err.message}`);
    }
  }

  function renderExperimentComparison(comparisonData) {
    if (!experimentComparisonSection) return;
    experimentComparisonSection.style.display = 'block';
    experimentComparisonSection.scrollIntoView({ behavior: 'smooth' });

    const expIds = comparisonData.experiment_ids || [];
    const cfgDiffs = comparisonData.configuration_differences || {};
    const metricComps = comparisonData.metric_comparisons || {};

    // 1. Config Header Row
    if (comparisonConfigHeaderRow) {
      comparisonConfigHeaderRow.innerHTML = '<th>Configuration Parameter</th>';
      expIds.forEach(id => {
        const th = document.createElement('th');
        th.innerHTML = `<code>${id}</code>`;
        comparisonConfigHeaderRow.appendChild(th);
      });
    }

    // 1. Config Table Body
    if (comparisonConfigTableBody) {
      comparisonConfigTableBody.innerHTML = '';
      const params = Object.keys(cfgDiffs);
      if (params.length === 0) {
        comparisonConfigTableBody.innerHTML = '<tr><td colspan="' + (expIds.length + 1) + '" class="empty-hint">Identical configurations across selected experiments.</td></tr>';
      } else {
        params.forEach(param => {
          const tr = document.createElement('tr');
          const isDiff = cfgDiffs[param].differing;
          if (isDiff) tr.className = 'highlight-diff-row';

          let cols = `<td><strong>${param}</strong> ${isDiff ? '<span class="status-badge status-needs-review">DIFFERENCE</span>' : ''}</td>`;
          const values = cfgDiffs[param].values || {};
          expIds.forEach(id => {
            const val = values[id] !== undefined ? JSON.stringify(values[id]) : '-';
            cols += `<td><code>${val}</code></td>`;
          });
          tr.innerHTML = cols;
          comparisonConfigTableBody.appendChild(tr);
        });
      }
    }

    // 2. Metrics Header Row
    if (comparisonMetricsHeaderRow) {
      comparisonMetricsHeaderRow.innerHTML = '<th>Research Metric Category</th>';
      expIds.forEach(id => {
        const th = document.createElement('th');
        th.innerHTML = `<code>${id}</code>`;
        comparisonMetricsHeaderRow.appendChild(th);
      });
    }

    // 2. Metrics Table Body
    if (comparisonMetricsTableBody) {
      comparisonMetricsTableBody.innerHTML = '';
      const catKeys = Object.keys(metricComps);
      if (catKeys.length === 0) {
        comparisonMetricsTableBody.innerHTML = '<tr><td colspan="' + (expIds.length + 1) + '" class="empty-hint">No observed metric data available for comparison.</td></tr>';
      } else {
        catKeys.forEach(cat => {
          const tr = document.createElement('tr');
          const catData = metricComps[cat] || {};
          let cols = `<td><strong>${cat.replace(/_/g, ' ').toUpperCase()}</strong></td>`;
          expIds.forEach(id => {
            const valObj = catData[id] || {};
            const summaryStr = Object.entries(valObj).map(([k, v]) => `${k}: ${typeof v === 'number' ? v.toFixed(3) : JSON.stringify(v)}`).join('<br>');
            cols += `<td style="font-size: 0.75rem;">${summaryStr || '-'}</td>`;
          });
          tr.innerHTML = cols;
          comparisonMetricsTableBody.appendChild(tr);
        });
      }
    }
  }

  function openCreateExpModal() {
    if (createExpFeedback) createExpFeedback.textContent = '';
    if (newExpName) newExpName.value = '';
    if (newExpDescription) newExpDescription.value = '';
    updateLiveFingerprintPreview();
    if (createExperimentModalBackdrop) createExperimentModalBackdrop.style.display = 'flex';
  }

  function closeCreateExpModal() {
    if (createExperimentModalBackdrop) createExperimentModalBackdrop.style.display = 'none';
  }

  function updateLiveFingerprintPreview() {
    if (!expFingerprintPreview) return;
    const model = newExpModelVersion ? newExpModelVersion.value.trim() : 'densenet121';
    const layer = newExpTargetLayer ? newExpTargetLayer.value : 'norm5';
    const thresh = newExpQaThreshold ? newExpQaThreshold.value : '0.15';
    const topk = newExpQaTopK ? newExpQaTopK.value : '5';
    
    // Quick deterministic preview hash representation
    const previewStr = `${model}::${layer}::${thresh}::${topk}`;
    let hash = 0;
    for (let i = 0; i < previewStr.length; i++) {
      hash = ((hash << 5) - hash) + previewStr.charCodeAt(i);
      hash |= 0;
    }
    const hex = Math.abs(hash).toString(16).padStart(8, '0');
    expFingerprintPreview.textContent = `sha256:fp_${hex}... (Computed on creation)`;
  }

  async function submitCreateExperiment() {
    const name = newExpName ? newExpName.value.trim() : '';
    const desc = newExpDescription ? newExpDescription.value.trim() : '';
    const snapId = newExpSnapshotSelect ? newExpSnapshotSelect.value : '';
    const modelVer = newExpModelVersion ? newExpModelVersion.value.trim() : 'densenet121-res224-all';
    const layer = newExpTargetLayer ? newExpTargetLayer.value : 'model.features.norm5';
    const thresh = newExpQaThreshold ? parseFloat(newExpQaThreshold.value) : 0.15;
    const topk = newExpQaTopK ? parseInt(newExpQaTopK.value, 10) : 5;
    const createdBy = newExpCreatedBy ? newExpCreatedBy.value.trim() : 'researcher_01';

    if (!name) {
      if (createExpFeedback) {
        createExpFeedback.textContent = 'Error: Experiment name is required.';
        createExpFeedback.className = 'report-action-feedback feedback-error';
      }
      return;
    }

    if (!snapId) {
      if (createExpFeedback) {
        createExpFeedback.textContent = 'Error: Target dataset snapshot must be selected.';
        createExpFeedback.className = 'report-action-feedback feedback-error';
      }
      return;
    }

    if (submitCreateExperimentBtn) {
      submitCreateExperimentBtn.disabled = true;
      submitCreateExperimentBtn.textContent = 'Creating...';
    }

    try {
      const resp = await fetch('/api/experiments', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: name,
          description: desc,
          snapshot_id: snapId,
          model_name: 'TorchXRayVision DenseNet-121',
          model_version: modelVer,
          target_layer: layer,
          qa_threshold: thresh,
          qa_top_k: topk,
          created_by: createdBy
        })
      });

      const resData = await resp.json();
      if (!resp.ok) {
        throw new Error(resData.error || `HTTP ${resp.status}`);
      }

      closeCreateExpModal();
      await loadResearchEvaluation();
      showExperimentDetails(resData.experiment_id);
      alert(`Experiment '${name}' created successfully!`);
    } catch (err) {
      if (createExpFeedback) {
        createExpFeedback.textContent = `Error: ${err.message}`;
        createExpFeedback.className = 'report-action-feedback feedback-error';
      }
    } finally {
      if (submitCreateExperimentBtn) {
        submitCreateExperimentBtn.disabled = false;
        submitCreateExperimentBtn.textContent = 'Create Experiment Record';
      }
    }
  }

  function openCreateSnapshotModal() {
    if (createSnapFeedback) createSnapFeedback.textContent = '';
    if (newSnapName) newSnapName.value = '';
    if (newSnapDescription) newSnapDescription.value = '';
    if (createSnapshotModalBackdrop) createSnapshotModalBackdrop.style.display = 'flex';
  }

  function closeCreateSnapshotModal() {
    if (createSnapshotModalBackdrop) createSnapshotModalBackdrop.style.display = 'none';
  }

  async function submitCreateSnapshot() {
    const name = newSnapName ? newSnapName.value.trim() : '';
    const desc = newSnapDescription ? newSnapDescription.value.trim() : '';
    const sel = newSnapStudySelection ? newSnapStudySelection.value : 'all';
    const createdBy = newSnapCreatedBy ? newSnapCreatedBy.value.trim() : 'dataset_curator';

    if (!name) {
      if (createSnapFeedback) {
        createSnapFeedback.textContent = 'Error: Snapshot name is required.';
        createSnapFeedback.className = 'report-action-feedback feedback-error';
      }
      return;
    }

    if (submitCreateSnapshotBtn) {
      submitCreateSnapshotBtn.disabled = true;
      submitCreateSnapshotBtn.textContent = 'Creating Snapshot...';
    }

    try {
      let studyIds = undefined;
      if (sel === 'cxr1122_only') {
        studyIds = ['CXR1122'];
      }

      const resp = await fetch('/api/experiments/snapshots', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: name,
          description: desc,
          study_ids: studyIds,
          created_by: createdBy
        })
      });

      const resData = await resp.json();
      if (!resp.ok) {
        throw new Error(resData.error || `HTTP ${resp.status}`);
      }

      closeCreateSnapshotModal();
      await loadResearchEvaluation();
      alert(`Dataset snapshot '${name}' created and cryptographically verified!`);
    } catch (err) {
      if (createSnapFeedback) {
        createSnapFeedback.textContent = `Error: ${err.message}`;
        createSnapFeedback.className = 'report-action-feedback feedback-error';
      }
    } finally {
      if (submitCreateSnapshotBtn) {
        submitCreateSnapshotBtn.disabled = false;
        submitCreateSnapshotBtn.textContent = 'Create Immutable Snapshot';
      }
    }
  }

  async function exportExperiment(format) {
    if (!state.activeExperimentId) {
      alert('No experiment currently selected for export.');
      return;
    }

    try {
      const resp = await fetch(`/api/experiments/${state.activeExperimentId}/export?format=${format}`);
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);

      if (format === 'text') {
        const text = await resp.text();
        const blob = new Blob([text], { type: 'text/plain' });
        downloadBlob(blob, `${state.activeExperimentId}_research_report.txt`);
      } else {
        const json = await resp.json();
        const blob = new Blob([JSON.stringify(json, null, 2)], { type: 'application/json' });
        downloadBlob(blob, `${state.activeExperimentId}_research_package.json`);
      }
    } catch (err) {
      console.error('Experiment export error:', err);
      alert('Failed to export experiment.');
    }
  }

  // ============================================================
  // PHASE 1.6 — RESEARCH EVALUATION & BENCHMARKING FUNCTIONS
  // ============================================================

  // Phase 1.6 DOM Elements
  const evalOverviewTotal = document.getElementById('evalOverviewTotal');
  const evalOverviewCompleted = document.getElementById('evalOverviewCompleted');
  const evalOverviewValidated = document.getElementById('evalOverviewValidated');
  const evalOverviewDatasets = document.getElementById('evalOverviewDatasets');
  const evalOverviewStudies = document.getElementById('evalOverviewStudies');

  const openNewEvalModalBtn = document.getElementById('openNewEvalModalBtn');
  const openNewEvalDatasetModalBtn = document.getElementById('openNewEvalDatasetModalBtn');
  const createEvalModalBackdrop = document.getElementById('createEvalModalBackdrop');
  const closeCreateEvalModalBtn = document.getElementById('closeCreateEvalModalBtn');
  const cancelCreateEvalModalBtn = document.getElementById('cancelCreateEvalModalBtn');
  const submitCreateEvalBtn = document.getElementById('submitCreateEvalBtn');
  const createEvalFeedback = document.getElementById('createEvalFeedback');

  const newEvalTitle = document.getElementById('newEvalTitle');
  const newEvalDescription = document.getElementById('newEvalDescription');
  const newEvalExpSelect = document.getElementById('newEvalExpSelect');
  const newEvalDatasetSelect = document.getElementById('newEvalDatasetSelect');
  const evalFingerprintPreview = document.getElementById('evalFingerprintPreview');

  const createEvalDatasetModalBackdrop = document.getElementById('createEvalDatasetModalBackdrop');
  const closeCreateEvalDatasetModalBtn = document.getElementById('closeCreateEvalDatasetModalBtn');
  const cancelCreateEvalDatasetModalBtn = document.getElementById('cancelCreateEvalDatasetModalBtn');
  const submitCreateEvalDatasetBtn = document.getElementById('submitCreateEvalDatasetBtn');
  const createEvalDatasetFeedback = document.getElementById('createEvalDatasetFeedback');
  const newEvalDatasetId = document.getElementById('newEvalDatasetId');
  const newEvalDatasetDesc = document.getElementById('newEvalDatasetDesc');

  const evaluationsTable = document.getElementById('evaluationsTable');
  const evaluationsTableBody = document.getElementById('evaluationsTableBody');
  const evaluationsTotalCount = document.getElementById('evaluationsTotalCount');

  const evaluationDetailSection = document.getElementById('evaluationDetailSection');
  const detailEvalId = document.getElementById('detailEvalId');
  const detailEvalTitle = document.getElementById('detailEvalTitle');
  const detailEvalStatusBadge = document.getElementById('detailEvalStatusBadge');
  const detailEvalExpId = document.getElementById('detailEvalExpId');
  const detailEvalDatasetId = document.getElementById('detailEvalDatasetId');
  const detailEvalStudyCount = document.getElementById('detailEvalStudyCount');
  const detailEvalCompletedAt = document.getElementById('detailEvalCompletedAt');
  const detailEvalFingerprintHash = document.getElementById('detailEvalFingerprintHash');
  const detailEvalDatasetFingerprint = document.getElementById('detailEvalDatasetFingerprint');
  const detailEvalConfigFingerprint = document.getElementById('detailEvalConfigFingerprint');

  const validateEvalRunBtn = document.getElementById('validateEvalRunBtn');
  const archiveEvalRunBtn = document.getElementById('archiveEvalRunBtn');
  const exportEvalJsonBtn = document.getElementById('exportEvalJsonBtn');
  const exportEvalTextBtn = document.getElementById('exportEvalTextBtn');

  const evalMetricAcc = document.getElementById('evalMetricAcc');
  const evalMetricAccCI = document.getElementById('evalMetricAccCI');
  const evalMetricPrec = document.getElementById('evalMetricPrec');
  const evalMetricPrecCI = document.getElementById('evalMetricPrecCI');
  const evalMetricRecall = document.getElementById('evalMetricRecall');
  const evalMetricRecallCI = document.getElementById('evalMetricRecallCI');
  const evalMetricF1 = document.getElementById('evalMetricF1');
  const evalMetricF1CI = document.getElementById('evalMetricF1CI');
  const evalMetricSpec = document.getElementById('evalMetricSpec');
  const evalMetricSpecCI = document.getElementById('evalMetricSpecCI');
  const evalMetricBalAcc = document.getElementById('evalMetricBalAcc');
  const evalMetricBalAccCI = document.getElementById('evalMetricBalAccCI');
  const evalMetricCohenK = document.getElementById('evalMetricCohenK');
  const evalMetricFleissK = document.getElementById('evalMetricFleissK');

  const cmValTP = document.getElementById('cmValTP');
  const cmValFP = document.getElementById('cmValFP');
  const cmValFN = document.getElementById('cmValFN');
  const cmValTN = document.getElementById('cmValTN');

  const errValTotal = document.getElementById('errValTotal');
  const errValMachineReviewer = document.getElementById('errValMachineReviewer');
  const errValInterReviewer = document.getElementById('errValInterReviewer');
  const errValAdjudicationFreq = document.getElementById('errValAdjudicationFreq');

  const evalFindingsTableBody = document.getElementById('evalFindingsTableBody');
  const evalStatsTableBody = document.getElementById('evalStatsTableBody');
  const evalProvenanceTimeline = document.getElementById('evalProvenanceTimeline');

  // Attach Phase 1.6 Event Listeners
  if (openNewEvalModalBtn) openNewEvalModalBtn.addEventListener('click', openCreateEvalModal);
  if (closeCreateEvalModalBtn) closeCreateEvalModalBtn.addEventListener('click', closeCreateEvalModal);
  if (cancelCreateEvalModalBtn) cancelCreateEvalModalBtn.addEventListener('click', closeCreateEvalModal);
  if (submitCreateEvalBtn) submitCreateEvalBtn.addEventListener('click', submitCreateEvaluation);

  if (openNewEvalDatasetModalBtn) openNewEvalDatasetModalBtn.addEventListener('click', openCreateEvalDatasetModal);
  if (closeCreateEvalDatasetModalBtn) closeCreateEvalDatasetModalBtn.addEventListener('click', closeCreateEvalDatasetModal);
  if (cancelCreateEvalDatasetModalBtn) cancelCreateEvalDatasetModalBtn.addEventListener('click', closeCreateEvalDatasetModal);
  if (submitCreateEvalDatasetBtn) submitCreateEvalDatasetBtn.addEventListener('click', submitCreateEvaluationDataset);

  if (validateEvalRunBtn) validateEvalRunBtn.addEventListener('click', () => triggerValidateEvaluation(state.activeEvaluationId));
  if (archiveEvalRunBtn) archiveEvalRunBtn.addEventListener('click', () => triggerArchiveEvaluation(state.activeEvaluationId));
  if (exportEvalJsonBtn) exportEvalJsonBtn.addEventListener('click', () => exportEvaluation(state.activeEvaluationId, 'json'));
  if (exportEvalTextBtn) exportEvalTextBtn.addEventListener('click', () => exportEvaluation(state.activeEvaluationId, 'text'));

  // Extended loadResearchEvaluation
  const origLoadResearchEvaluation = loadResearchEvaluation;
  loadResearchEvaluation = async function() {
    await origLoadResearchEvaluation();
    try {
      // Fetch Phase 1.6 dashboard stats
      const dResp = await fetch('/api/evaluation-dashboard');
      if (dResp.ok) {
        const dData = await dResp.json();
        renderEvaluationDashboard(dData);
      }

      // Fetch evaluations
      const eResp = await fetch('/api/evaluations');
      if (eResp.ok) {
        const eData = await eResp.json();
        state.evaluationsList = eData.evaluations || [];
        renderEvaluationsTable(state.evaluationsList);
      }
    } catch (err) {
      console.error('Failed to load evaluation data:', err);
    }
  };

  function renderEvaluationDashboard(dData) {
    if (evalOverviewTotal) evalOverviewTotal.textContent = dData.total_evaluations || 0;
    if (evalOverviewCompleted) evalOverviewCompleted.textContent = dData.completed_evaluations || 0;
    if (evalOverviewValidated) evalOverviewValidated.textContent = dData.validated_evaluations || 0;
    if (evalOverviewDatasets) evalOverviewDatasets.textContent = dData.total_evaluation_datasets || 0;
    if (evalOverviewStudies) evalOverviewStudies.textContent = dData.total_evaluated_studies || 0;
  }

  function renderEvaluationsTable(evaluations) {
    if (!evaluationsTableBody) return;
    evaluationsTableBody.innerHTML = '';
    if (evaluationsTotalCount) evaluationsTotalCount.textContent = evaluations.length;

    if (evaluations.length === 0) {
      evaluationsTableBody.innerHTML = '<tr><td colspan="9" class="empty-hint">No evaluation runs recorded. Click "New Evaluation Run" to benchmark an experiment.</td></tr>';
      return;
    }

    evaluations.forEach(ev => {
      const tr = document.createElement('tr');
      const st = (ev.status || 'CREATED').toUpperCase();
      let statusClass = 'status-neutral';
      if (st === 'COMPLETED' || st === 'VALIDATED') statusClass = 'status-supported';
      else if (st === 'RUNNING') statusClass = 'status-possible';
      else if (st === 'ARCHIVED') statusClass = 'status-needs-review';

      const metrics = ev.aggregate_metrics || {};
      const acc = metrics.accuracy ? metrics.accuracy.metric_value.toFixed(3) : '-';
      const f1 = metrics.f1 ? metrics.f1.metric_value.toFixed(3) : '-';
      const kappa = metrics.cohen_kappa ? metrics.cohen_kappa.metric_value.toFixed(3) : '-';

      tr.innerHTML = `
        <td><strong><code>${ev.evaluation_id}</code></strong></td>
        <td>
          <div style="font-weight: 600; color: var(--text-primary);">${ev.title || ev.evaluation_id}</div>
          <div style="font-size: 0.72rem; color: var(--text-secondary);">${ev.description || ''}</div>
        </td>
        <td><code>${ev.experiment_id}</code></td>
        <td><code>${ev.evaluation_dataset_id}</code></td>
        <td><span class="status-badge ${statusClass}">${st}</span></td>
        <td><code>${acc}</code></td>
        <td><code>${f1}</code></td>
        <td><code>${kappa}</code></td>
        <td>
          <div class="action-buttons-group">
            <button class="btn btn-xs btn-primary btn-view-eval" data-id="${ev.evaluation_id}">Details</button>
            ${st === 'CREATED' ? `<button class="btn btn-xs btn-secondary btn-run-eval" data-id="${ev.evaluation_id}">▶ Run</button>` : ''}
          </div>
        </td>
      `;

      tr.querySelector('.btn-view-eval').addEventListener('click', () => showEvaluationDetails(ev.evaluation_id));
      const runBtn = tr.querySelector('.btn-run-eval');
      if (runBtn) runBtn.addEventListener('click', () => triggerRunEvaluation(ev.evaluation_id));

      evaluationsTableBody.appendChild(tr);
    });
  }

  async function showEvaluationDetails(evalId) {
    state.activeEvaluationId = evalId;
    try {
      const resp = await fetch(`/api/evaluations/${evalId}`);
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const ev = await resp.json();

      if (!evaluationDetailSection) return;
      evaluationDetailSection.style.display = 'block';
      evaluationDetailSection.scrollIntoView({ behavior: 'smooth' });

      if (detailEvalId) detailEvalId.textContent = ev.evaluation_id;
      if (detailEvalTitle) detailEvalTitle.textContent = ev.title || ev.evaluation_id;
      if (detailEvalStatusBadge) {
        detailEvalStatusBadge.textContent = ev.status;
        detailEvalStatusBadge.className = `status-badge ${ev.status === 'COMPLETED' || ev.status === 'VALIDATED' ? 'status-supported' : 'status-neutral'}`;
      }
      if (detailEvalExpId) detailEvalExpId.textContent = ev.experiment_id;
      if (detailEvalDatasetId) detailEvalDatasetId.textContent = ev.evaluation_dataset_id;
      if (detailEvalStudyCount) detailEvalStudyCount.textContent = `${ev.study_count || 0} Studies`;
      if (detailEvalCompletedAt) detailEvalCompletedAt.textContent = ev.completed_at || 'In progress / Created';

      if (detailEvalFingerprintHash) detailEvalFingerprintHash.textContent = ev.evaluation_fingerprint || '-';
      if (detailEvalDatasetFingerprint) detailEvalDatasetFingerprint.textContent = ev.dataset_fingerprint || '-';
      if (detailEvalConfigFingerprint) detailEvalConfigFingerprint.textContent = ev.configuration_fingerprint || '-';

      // Aggregate Metrics Cards
      const m = ev.aggregate_metrics || {};
      const setMetric = (valEl, ciEl, mObj) => {
        if (!valEl) return;
        if (!mObj || mObj.metric_value === undefined || mObj.metric_value === null) {
          valEl.textContent = '-';
          if (ciEl) ciEl.textContent = '95% CI: -';
        } else {
          valEl.textContent = mObj.metric_value.toFixed(4);
          if (ciEl && mObj.confidence_interval_95) {
            ciEl.textContent = `95% CI: [${mObj.confidence_interval_95.lower} - ${mObj.confidence_interval_95.upper}]`;
          }
        }
      };

      setMetric(evalMetricAcc, evalMetricAccCI, m.accuracy);
      setMetric(evalMetricPrec, evalMetricPrecCI, m.precision);
      setMetric(evalMetricRecall, evalMetricRecallCI, m.recall);
      setMetric(evalMetricF1, evalMetricF1CI, m.f1);
      setMetric(evalMetricSpec, evalMetricSpecCI, m.specificity);
      setMetric(evalMetricBalAcc, evalMetricBalAccCI, m.balanced_accuracy);
      setMetric(evalMetricCohenK, null, m.cohen_kappa);
      setMetric(evalMetricFleissK, null, m.fleiss_kappa);

      // Confusion Matrix
      const cm = ev.confusion_matrix || { tp: 0, tn: 0, fp: 0, fn: 0 };
      if (cmValTP) cmValTP.textContent = `TP: ${cm.tp}`;
      if (cmValFP) cmValFP.textContent = `FP: ${cm.fp}`;
      if (cmValFN) cmValFN.textContent = `FN: ${cm.fn}`;
      if (cmValTN) cmValTN.textContent = `TN: ${cm.tn}`;

      // Error Breakdown
      const err = ev.error_analysis || {};
      if (errValTotal) errValTotal.textContent = err.total_disagreements || 0;
      if (errValMachineReviewer) errValMachineReviewer.textContent = err.machine_reviewer_disagreements || 0;
      if (errValInterReviewer) errValInterReviewer.textContent = err.inter_reviewer_disagreements || 0;
      if (errValAdjudicationFreq) errValAdjudicationFreq.textContent = (err.adjudication_frequency || 0.0).toFixed(4);

      // Findings Table
      if (evalFindingsTableBody) {
        evalFindingsTableBody.innerHTML = '';
        const findings = ev.finding_evaluations || [];
        if (findings.length === 0) {
          evalFindingsTableBody.innerHTML = '<tr><td colspan="9" class="empty-hint">Run evaluation to compute finding-level metrics.</td></tr>';
        } else {
          findings.forEach(f => {
            const fcm = f.confusion_matrix || {};
            const fm = f.metrics || {};
            const tr = document.createElement('tr');
            tr.innerHTML = `
              <td><strong>${f.finding_type}</strong></td>
              <td>${f.sample_count}</td>
              <td><code>${fcm.tp || 0}</code></td>
              <td><code>${fcm.tn || 0}</code></td>
              <td><code>${fcm.fp || 0}</code></td>
              <td><code>${fcm.fn || 0}</code></td>
              <td><code>${fm.precision !== undefined ? fm.precision.toFixed(3) : '-'}</code></td>
              <td><code>${fm.recall !== undefined ? fm.recall.toFixed(3) : '-'}</code></td>
              <td><code>${fm.f1 !== undefined ? fm.f1.toFixed(3) : '-'}</code></td>
            `;
            evalFindingsTableBody.appendChild(tr);
          });
        }
      }

      // Statistical Distributions
      if (evalStatsTableBody) {
        evalStatsTableBody.innerHTML = '';
        const dists = (ev.statistical_summary && ev.statistical_summary.metric_distributions) || {};
        const cis = (ev.statistical_summary && ev.statistical_summary.uncertainty_intervals) || {};
        const keys = Object.keys(dists);
        if (keys.length === 0) {
          evalStatsTableBody.innerHTML = '<tr><td colspan="10" class="empty-hint">No distribution data calculated yet.</td></tr>';
        } else {
          keys.forEach(k => {
            const d = dists[k];
            const ci = cis[k];
            const tr = document.createElement('tr');
            tr.innerHTML = `
              <td><strong>${k.toUpperCase()}</strong></td>
              <td><code>${d.mean.toFixed(3)}</code></td>
              <td><code>${d.median.toFixed(3)}</code></td>
              <td><code>${d.std.toFixed(3)}</code></td>
              <td><code>${d.min.toFixed(3)}</code></td>
              <td><code>${d.max.toFixed(3)}</code></td>
              <td><code>${d.q25.toFixed(3)}</code></td>
              <td><code>${d.q75.toFixed(3)}</code></td>
              <td><code>${d.iqr.toFixed(3)}</code></td>
              <td><code>${ci ? `[${ci.lower_95.toFixed(3)} - ${ci.upper_95.toFixed(3)}]` : '-'}</code></td>
            `;
            evalStatsTableBody.appendChild(tr);
          });
        }
      }

      // Provenance Timeline
      if (evalProvenanceTimeline) {
        evalProvenanceTimeline.innerHTML = '';
        const history = (ev.provenance && ev.provenance.status_history) || [{ status: ev.status, timestamp: ev.created_at }];
        history.forEach((h, idx) => {
          const item = document.createElement('div');
          item.className = 'timeline-item';
          item.innerHTML = `
            <div class="timeline-dot"></div>
            <div class="timeline-content">
              <strong>Stage ${idx + 1}: ${h.status}</strong>
              <div style="font-size: 0.72rem; color: var(--text-secondary);">${h.timestamp || ''}</div>
            </div>
          `;
          evalProvenanceTimeline.appendChild(item);
        });
      }

    } catch (err) {
      console.error('Failed to show evaluation details:', err);
      alert(`Cannot Load Evaluation Details:\n${err.message}`);
    }
  }

  async function triggerRunEvaluation(evalId) {
    try {
      const resp = await fetch(`/api/evaluations/${evalId}/run`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });
      const resData = await resp.json();
      if (!resp.ok) throw new Error(resData.error || `HTTP ${resp.status}`);

      await loadResearchEvaluation();
      showEvaluationDetails(evalId);
      alert(`Evaluation run '${evalId}' completed successfully!`);
    } catch (err) {
      console.error('Run evaluation failed:', err);
      alert(`Cannot Run Evaluation:\n${err.message}`);
    }
  }

  async function triggerValidateEvaluation(evalId) {
    try {
      const resp = await fetch(`/api/evaluations/${evalId}/validate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });
      const resData = await resp.json();
      if (!resp.ok) throw new Error(resData.error || `HTTP ${resp.status}`);

      await loadResearchEvaluation();
      showEvaluationDetails(evalId);
      alert(`Evaluation run '${evalId}' validated and locked!`);
    } catch (err) {
      console.error('Validate evaluation failed:', err);
      alert(`Cannot Validate Evaluation:\n${err.message}`);
    }
  }

  async function triggerArchiveEvaluation(evalId) {
    try {
      const resp = await fetch(`/api/evaluations/${evalId}/archive`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });
      const resData = await resp.json();
      if (!resp.ok) throw new Error(resData.error || `HTTP ${resp.status}`);

      await loadResearchEvaluation();
      showEvaluationDetails(evalId);
      alert(`Evaluation run '${evalId}' archived.`);
    } catch (err) {
      console.error('Archive evaluation failed:', err);
      alert(`Cannot Archive Evaluation:\n${err.message}`);
    }
  }

  async function exportEvaluation(evalId, format) {
    if (!evalId) {
      alert('No evaluation selected for export.');
      return;
    }
    try {
      const resp = await fetch(`/api/evaluations/${evalId}/export?format=${format}`);
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);

      if (format === 'text') {
        const text = await resp.text();
        const blob = new Blob([text], { type: 'text/plain' });
        downloadBlob(blob, `${evalId}_evaluation_report.txt`);
      } else {
        const json = await resp.json();
        const blob = new Blob([JSON.stringify(json, null, 2)], { type: 'application/json' });
        downloadBlob(blob, `${evalId}_evaluation_package.json`);
      }
    } catch (err) {
      console.error('Evaluation export error:', err);
      alert('Failed to export evaluation report.');
    }
  }

  function openCreateEvalModal() {
    if (createEvalFeedback) createEvalFeedback.textContent = '';
    if (newEvalTitle) newEvalTitle.value = '';
    if (newEvalDescription) newEvalDescription.value = '';

    // Populate experiment select
    if (newEvalExpSelect) {
      newEvalExpSelect.innerHTML = '';
      (state.experimentsList || []).forEach(exp => {
        const opt = document.createElement('option');
        opt.value = exp.experiment_id;
        opt.textContent = `${exp.experiment_name || exp.experiment_id} (${exp.status})`;
        newEvalExpSelect.appendChild(opt);
      });
    }

    // Populate evaluation dataset select
    if (newEvalDatasetSelect) {
      newEvalDatasetSelect.innerHTML = '';
      fetch('/api/evaluation-datasets').then(r => r.json()).then(d => {
        const datasets = d.datasets || [];
        if (datasets.length === 0) {
          const opt = document.createElement('option');
          opt.value = 'EVALSET_DEFAULT';
          opt.textContent = 'EVALSET_DEFAULT (Auto-Created Reference Dataset)';
          newEvalDatasetSelect.appendChild(opt);
        } else {
          datasets.forEach(ds => {
            const opt = document.createElement('option');
            opt.value = ds.dataset_id;
            opt.textContent = `${ds.dataset_id} (${ds.study_count} studies)`;
            newEvalDatasetSelect.appendChild(opt);
          });
        }
      });
    }

    if (createEvalModalBackdrop) createEvalModalBackdrop.style.display = 'flex';
  }

  function closeCreateEvalModal() {
    if (createEvalModalBackdrop) createEvalModalBackdrop.style.display = 'none';
  }

  async function submitCreateEvaluation() {
    const title = newEvalTitle ? newEvalTitle.value.trim() : '';
    const desc = newEvalDescription ? newEvalDescription.value.trim() : '';
    const expId = newEvalExpSelect ? newEvalExpSelect.value : '';
    let dsId = newEvalDatasetSelect ? newEvalDatasetSelect.value : '';

    if (!title) {
      if (createEvalFeedback) {
        createEvalFeedback.textContent = 'Error: Title is required.';
        createEvalFeedback.className = 'report-action-feedback feedback-error';
      }
      return;
    }

    if (!expId) {
      if (createEvalFeedback) {
        createEvalFeedback.textContent = 'Error: Target experiment must be selected.';
        createEvalFeedback.className = 'report-action-feedback feedback-error';
      }
      return;
    }

    if (!dsId) {
      dsId = 'EVALSET_DEFAULT';
    }

    try {
      const resp = await fetch('/api/evaluations', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          title: title,
          description: desc,
          experiment_id: expId,
          evaluation_dataset_id: dsId
        })
      });

      const resData = await resp.json();
      if (!resp.ok) throw new Error(resData.error || `HTTP ${resp.status}`);

      closeCreateEvalModal();
      await loadResearchEvaluation();
      showEvaluationDetails(resData.evaluation_id);
      alert(`Evaluation run '${title}' created successfully!`);
    } catch (err) {
      if (createEvalFeedback) {
        createEvalFeedback.textContent = `Error: ${err.message}`;
        createEvalFeedback.className = 'report-action-feedback feedback-error';
      }
    }
  }

  function openCreateEvalDatasetModal() {
    if (createEvalDatasetFeedback) createEvalDatasetFeedback.textContent = '';
    if (newEvalDatasetId) newEvalDatasetId.value = '';
    if (newEvalDatasetDesc) newEvalDatasetDesc.value = '';
    if (createEvalDatasetModalBackdrop) createEvalDatasetModalBackdrop.style.display = 'flex';
  }

  function closeCreateEvalDatasetModal() {
    if (createEvalDatasetModalBackdrop) createEvalDatasetModalBackdrop.style.display = 'none';
  }

  async function submitCreateEvaluationDataset() {
    const dsId = newEvalDatasetId ? newEvalDatasetId.value.trim() : '';
    const desc = newEvalDatasetDesc ? newEvalDatasetDesc.value.trim() : '';

    try {
      const resp = await fetch('/api/evaluation-datasets', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          dataset_id: dsId || undefined,
          source_description: desc || 'Isolated Reference Ground-Truth Manifest'
        })
      });

      const resData = await resp.json();
      if (!resp.ok) throw new Error(resData.error || `HTTP ${resp.status}`);

      closeCreateEvalDatasetModal();
      await loadResearchEvaluation();
      alert(`Evaluation dataset '${resData.dataset_id}' created with SHA-256 cryptographic manifest!`);
    } catch (err) {
      if (createEvalDatasetFeedback) {
        createEvalDatasetFeedback.textContent = `Error: ${err.message}`;
        createEvalDatasetFeedback.className = 'report-action-feedback feedback-error';
      }
    }
  }

  // ============================================================
  // PHASE 1.7 — REPRODUCIBLE EXPERIMENT REGISTRY & VERSIONING
  // ============================================================

  async function loadExperimentRegistry() {
    try {
      const resp = await fetch('/api/experiment-dashboard');
      if (!resp.ok) return;
      const data = await resp.json();

      // Stats
      const statRegTotalExp = document.getElementById('statRegTotalExp');
      const statRegTotalModels = document.getElementById('statRegTotalModels');
      const statRegTotalDsv = document.getElementById('statRegTotalDsv');
      const statRegTotalFinalized = document.getElementById('statRegTotalFinalized');

      if (statRegTotalExp) statRegTotalExp.textContent = data.total_experiments || 0;
      if (statRegTotalModels) statRegTotalModels.textContent = data.total_models || 0;
      if (statRegTotalDsv) statRegTotalDsv.textContent = data.total_dataset_versions || 0;
      if (statRegTotalFinalized) statRegTotalFinalized.textContent = data.finalized_experiments || 0;

      // Render Tables
      renderModelTable(data.models || []);
      renderDatasetVersionTable(data.dataset_versions || []);
      renderExperimentRegistryTable(data.experiments || []);

      // Populate comparison dropdowns
      populateComparisonDropdowns(data.experiments || []);
    } catch (err) {
      console.error('Failed to load experiment registry dashboard:', err);
    }
  }

  function renderModelTable(models) {
    const tbody = document.getElementById('modelRegistryTableBody');
    if (!tbody) return;
    tbody.innerHTML = '';

    if (models.length === 0) {
      tbody.innerHTML = '<tr><td colspan="8" class="empty-hint">No models registered yet.</td></tr>';
      return;
    }

    models.forEach(m => {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td><code>${m.model_id}</code></td>
        <td><strong>${m.model_name || m.model_id}</strong></td>
        <td>${m.architecture || 'DenseNet-121'}</td>
        <td>${m.framework || 'PyTorch'}</td>
        <td><code>${m.target_layer || 'model.features.norm5'}</code></td>
        <td><span class="status-badge ${m.status === 'FINALIZED' ? 'status-confirmed-present' : 'status-in-review'}">${m.status || 'REGISTERED'}</span></td>
        <td style="font-family: var(--font-mono); font-size: 0.7rem;">${m.created_at ? m.created_at.substring(0, 10) : '-'}</td>
        <td>
          <button class="btn btn-sm btn-secondary btn-verify-model" data-id="${m.model_id}">Verify Hash</button>
        </td>
      `;

      tr.querySelector('.btn-verify-model').addEventListener('click', async () => {
        try {
          const vresp = await fetch(`/api/models/${m.model_id}/verify`);
          const vdata = await vresp.json();
          alert(`Model Verification: ${m.model_id}\n\nStatus: ${vdata.status_note}\nComputed SHA-256: ${vdata.computed_sha256 || 'None'}`);
        } catch (e) {
          alert(`Verification failed: ${e.message}`);
        }
      });

      tbody.appendChild(tr);
    });
  }

  function renderDatasetVersionTable(dsvs) {
    const tbody = document.getElementById('datasetVersionsTableBody');
    if (!tbody) return;
    tbody.innerHTML = '';

    if (dsvs.length === 0) {
      tbody.innerHTML = '<tr><td colspan="7" class="empty-hint">No dataset versions registered.</td></tr>';
      return;
    }

    dsvs.forEach(d => {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td><code>${d.dataset_version_id}</code></td>
        <td>${d.source_dataset || 'IU_XRAY'}</td>
        <td><strong>${d.study_count || 0}</strong> studies</td>
        <td><code class="sha-hash">${d.manifest_sha256 ? d.manifest_sha256.substring(0, 16) + '...' : '-'}</code></td>
        <td><span class="status-badge ${d.status === 'FINALIZED' ? 'status-confirmed-present' : 'status-in-review'}">${d.status || 'DRAFT'}</span></td>
        <td style="font-family: var(--font-mono); font-size: 0.7rem;">${d.created_at ? d.created_at.substring(0, 10) : '-'}</td>
        <td>
          <button class="btn btn-sm btn-secondary btn-val-dsv" data-id="${d.dataset_version_id}">Validate</button>
          ${d.status !== 'FINALIZED' ? `<button class="btn btn-sm btn-primary btn-fin-dsv" data-id="${d.dataset_version_id}">Finalize</button>` : ''}
        </td>
      `;

      tr.querySelector('.btn-val-dsv').addEventListener('click', async () => {
        try {
          const vresp = await fetch(`/api/dataset-versions/${d.dataset_version_id}/validate`, { method: 'POST' });
          const vdata = await vresp.json();
          alert(`Dataset Version Validation:\n\nManifest Valid: ${vdata.validation?.manifest_valid}\nStudy Count: ${vdata.validation?.study_count}`);
          loadExperimentRegistry();
        } catch (e) {
          alert(`Validation error: ${e.message}`);
        }
      });

      const finBtn = tr.querySelector('.btn-fin-dsv');
      if (finBtn) {
        finBtn.addEventListener('click', async () => {
          try {
            const fresp = await fetch(`/api/dataset-versions/${d.dataset_version_id}/finalize`, { method: 'POST' });
            if (fresp.ok) {
              alert(`Dataset version '${d.dataset_version_id}' finalized and locked as immutable.`);
              loadExperimentRegistry();
            }
          } catch (e) {
            alert(`Finalization error: ${e.message}`);
          }
        });
      }

      tbody.appendChild(tr);
    });
  }

  function renderExperimentRegistryTable(exps) {
    const tbody = document.getElementById('experimentRegistryTableBody');
    if (!tbody) return;
    tbody.innerHTML = '';

    if (exps.length === 0) {
      tbody.innerHTML = '<tr><td colspan="8" class="empty-hint">No experiments registered yet.</td></tr>';
      return;
    }

    exps.forEach(e => {
      const tr = document.createElement('tr');
      const stat = e.status || 'REGISTERED';
      tr.innerHTML = `
        <td><code>${e.experiment_id}</code></td>
        <td><strong>${e.experiment_name || e.experiment_id}</strong></td>
        <td><code>${e.model_id || 'model_densenet121_txrv'}</code></td>
        <td><code>${e.dataset_version_id || '-'}</code></td>
        <td><span class="status-badge ${stat === 'FINALIZED' ? 'status-confirmed-present' : (stat === 'COMPLETED' ? 'status-needs-review' : 'status-in-review')}">${stat}</span></td>
        <td><code class="sha-hash">${e.fingerprint ? e.fingerprint.substring(0, 12) + '...' : '-'}</code></td>
        <td style="font-family: var(--font-mono); font-size: 0.7rem;">${e.created_at ? e.created_at.substring(0, 10) : '-'}</td>
        <td>
          <button class="btn btn-sm btn-primary btn-view-reg-exp" data-id="${e.experiment_id}">Details</button>
        </td>
      `;

      tr.querySelector('.btn-view-reg-exp').addEventListener('click', () => {
        showRegExpDetails(e.experiment_id);
      });

      tbody.appendChild(tr);
    });
  }

  async function showRegExpDetails(expId) {
    try {
      const resp = await fetch(`/api/experiments/${expId}`);
      if (!resp.ok) return;
      const exp = await resp.json();

      const sec = document.getElementById('regExpDetailSection');
      if (sec) sec.style.display = 'block';

      document.getElementById('regDetailExpId').textContent = exp.experiment_id;
      document.getElementById('regDetailExpName').textContent = exp.experiment_name || exp.name || '-';
      document.getElementById('regDetailExpDesc').textContent = exp.description || '-';
      document.getElementById('regDetailExpStatusBadge').textContent = exp.status;
      document.getElementById('regDetailExpModelId').textContent = exp.model_id || '-';
      document.getElementById('regDetailExpDsvId').textContent = exp.dataset_version_id || '-';
      document.getElementById('regDetailExpMethodology').textContent = exp.methodology || 'standard_qa_evidence_evaluation';
      document.getElementById('regDetailExpFingerprint').textContent = exp.fingerprint_sha256 || '-';
      document.getElementById('regDetailExpCreatedAt').textContent = exp.created_at || '-';
      document.getElementById('regDetailExpCompletedAt').textContent = exp.completed_at || 'Not completed';
      document.getElementById('regDetailExpEvalRunId').textContent = exp.evaluation_run_id || '-';
      document.getElementById('regDetailExpSnapshotHash').textContent = exp.snapshot_sha256 || 'None (Draft/In-Progress)';

      // Wire action buttons
      const runBtn = document.getElementById('regExpRunBtn');
      const valBtn = document.getElementById('regExpValidateBtn');
      const finBtn = document.getElementById('regExpFinalizeBtn');
      const archBtn = document.getElementById('regExpArchiveBtn');
      const jsonBtn = document.getElementById('regExpExportJsonBtn');
      const textBtn = document.getElementById('regExpExportTextBtn');

      if (runBtn) {
        runBtn.onclick = async () => {
          try {
            const rresp = await fetch(`/api/experiments/${expId}/run`, { method: 'POST' });
            if (rresp.ok) {
              alert(`Experiment '${expId}' executed successfully!`);
              loadExperimentRegistry();
              showRegExpDetails(expId);
            }
          } catch (e) {
            alert(`Execution failed: ${e.message}`);
          }
        };
      }

      if (valBtn) {
        valBtn.onclick = async () => {
          try {
            const vresp = await fetch(`/api/experiments/${expId}/validate`, { method: 'POST' });
            const vdata = await vresp.json();
            alert(`Validation Result: ${vdata.validation?.valid ? 'PASSED' : 'ISSUES DETECTED'}`);
            loadExperimentRegistry();
            showRegExpDetails(expId);
          } catch (e) {
            alert(`Validation failed: ${e.message}`);
          }
        };
      }

      if (finBtn) {
        finBtn.onclick = async () => {
          try {
            const fresp = await fetch(`/api/experiments/${expId}/finalize`, { method: 'POST' });
            if (fresp.ok) {
              alert(`Experiment '${expId}' finalized and locked as immutable snapshot!`);
              loadExperimentRegistry();
              showRegExpDetails(expId);
            }
          } catch (e) {
            alert(`Finalization failed: ${e.message}`);
          }
        };
      }

      if (archBtn) {
        archBtn.onclick = async () => {
          try {
            const aresp = await fetch(`/api/experiments/${expId}/archive`, { method: 'POST' });
            if (aresp.ok) {
              alert(`Experiment '${expId}' archived.`);
              loadExperimentRegistry();
              showRegExpDetails(expId);
            }
          } catch (e) {
            alert(`Archive failed: ${e.message}`);
          }
        };
      }

      if (jsonBtn) {
        jsonBtn.onclick = () => exportExperimentPackage(expId, 'json');
      }
      if (textBtn) {
        textBtn.onclick = () => exportExperimentPackage(expId, 'text');
      }

      // Render Metrics Grid
      const mGrid = document.getElementById('regDetailMetricsGrid');
      if (mGrid) {
        mGrid.innerHTML = '';
        const metrics = exp.metrics || {};
        const displayedMetrics = [
          { key: 'accuracy', label: 'ACCURACY', val: metrics.accuracy !== undefined ? metrics.accuracy : '-' },
          { key: 'precision', label: 'PRECISION', val: metrics.precision !== undefined ? metrics.precision : '-' },
          { key: 'recall', label: 'RECALL', val: metrics.recall !== undefined ? metrics.recall : '-' },
          { key: 'f1_score', label: 'F1 SCORE', val: metrics.f1_score !== undefined ? metrics.f1_score : '-' },
          { key: 'specificity', label: 'SPECIFICITY', val: metrics.specificity !== undefined ? metrics.specificity : '-' },
          { key: 'sensitivity', label: 'SENSITIVITY', val: metrics.sensitivity !== undefined ? metrics.sensitivity : '-' }
        ];

        displayedMetrics.forEach(dm => {
          const card = document.createElement('div');
          card.className = 'stat-card';
          card.innerHTML = `
            <div class="stat-label">${dm.label}</div>
            <div class="stat-val">${typeof dm.val === 'number' ? dm.val.toFixed(4) : dm.val}</div>
            <div class="stat-sub">Observed Benchmark Metric</div>
          `;
          mGrid.appendChild(card);
        });
      }

      // Render Longitudinal Timeline
      const tContainer = document.getElementById('regExpHistoryTimeline');
      if (tContainer) {
        tContainer.innerHTML = '';
        const hresp = await fetch(`/api/experiments/${expId}/history`);
        if (hresp.ok) {
          const hdata = await hresp.json();
          const timelineEvents = hdata.timeline || [];
          if (timelineEvents.length === 0) {
            tContainer.innerHTML = '<div class="empty-hint">No history events recorded yet.</div>';
          } else {
            timelineEvents.forEach(ev => {
              const div = document.createElement('div');
              div.className = 'provenance-stage';
              div.innerHTML = `
                <div class="stage-badge stage-pass">✓</div>
                <div class="stage-info">
                  <div class="stage-name">${ev.action} (${ev.actor})</div>
                  <div class="stage-desc">${ev.summary || ''}</div>
                  <div class="stage-time">${ev.timestamp ? ev.timestamp.replace('T', ' ').substring(0, 19) : ''}</div>
                </div>
              `;
              tContainer.appendChild(div);
            });
          }
        }
      }

      sec.scrollIntoView({ behavior: 'smooth' });
    } catch (err) {
      console.error('Error displaying experiment details:', err);
    }
  }

  async function exportExperimentPackage(expId, format) {
    try {
      const resp = await fetch(`/api/experiments/${expId}/export?format=${format}`);
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      if (format === 'text') {
        const text = await resp.text();
        const blob = new Blob([text], { type: 'text/plain' });
        downloadBlob(blob, `${expId}_experiment_report.txt`);
      } else {
        const json = await resp.json();
        const blob = new Blob([JSON.stringify(json, null, 2)], { type: 'application/json' });
        downloadBlob(blob, `${expId}_experiment_package.json`);
      }
    } catch (e) {
      alert(`Export error: ${e.message}`);
    }
  }

  function populateComparisonDropdowns(exps) {
    const selA = document.getElementById('compExpASelect');
    const selB = document.getElementById('compExpBSelect');
    if (!selA || !selB) return;
    selA.innerHTML = '';
    selB.innerHTML = '';

    exps.forEach(e => {
      const optA = document.createElement('option');
      optA.value = e.experiment_id;
      optA.textContent = `${e.experiment_name || e.experiment_id} (${e.status})`;
      selA.appendChild(optA);

      const optB = document.createElement('option');
      optB.value = e.experiment_id;
      optB.textContent = `${e.experiment_name || e.experiment_id} (${e.status})`;
      selB.appendChild(optB);
    });

    if (selB.options.length > 1) selB.selectedIndex = 1;

    const compBtn = document.getElementById('runRegComparisonBtn');
    if (compBtn) {
      compBtn.onclick = async () => {
        const expA = selA.value;
        const expB = selB.value;
        if (!expA || !expB) {
          alert('Please select two experiments to compare.');
          return;
        }
        try {
          const resp = await fetch('/api/experiments/compare', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ experiment_ids: [expA, expB] })
          });
          const compData = await resp.json();
          if (!resp.ok) throw new Error(compData.error || `HTTP ${resp.status}`);

          document.getElementById('regComparisonResults').style.display = 'block';
          const notBar = document.getElementById('regCompCompatibilityNotice');
          if (notBar) notBar.textContent = compData.comparison_notes || compData.evaluation_notice || 'Non-evaluative side-by-side comparison';

          // Config Diff Table
          const cfgBody = document.getElementById('regCompConfigBody');
          if (cfgBody) {
            cfgBody.innerHTML = '';
            const diffs = compData.configuration_differences || [];
            if (diffs.length === 0) {
              cfgBody.innerHTML = '<tr><td colspan="3" class="empty-hint">Configurations are identical across both runs.</td></tr>';
            } else {
              diffs.forEach(d => {
                const tr = document.createElement('tr');
                const vmap = d.values_by_experiment || {};
                tr.innerHTML = `
                  <td><strong>${d.configuration_field}</strong></td>
                  <td><code>${vmap[expA] || '-'}</code></td>
                  <td><code>${vmap[expB] || '-'}</code></td>
                `;
                cfgBody.appendChild(tr);
              });
            }
          }

          // Metrics Delta Table
          const mBody = document.getElementById('regCompMetricsBody');
          if (mBody) {
            mBody.innerHTML = '';
            const deltas = compData.metric_deltas || {};
            const relDeltas = compData.relative_deltas || {};
            const keys = Object.keys(deltas);
            if (keys.length === 0) {
              mBody.innerHTML = '<tr><td colspan="5" class="empty-hint">No comparable numeric metrics found.</td></tr>';
            } else {
              keys.forEach(k => {
                const tr = document.createElement('tr');
                const dVal = deltas[k];
                const rVal = relDeltas[k];
                tr.innerHTML = `
                  <td><strong>${k}</strong></td>
                  <td>-</td>
                  <td>-</td>
                  <td>${dVal !== undefined ? (dVal >= 0 ? '+' : '') + dVal : '-'}</td>
                  <td>${rVal !== undefined && rVal !== null ? (rVal * 100).toFixed(2) + '%' : '-'}</td>
                `;
                mBody.appendChild(tr);
              });
            }
          }
        } catch (e) {
          alert(`Comparison failed: ${e.message}`);
        }
      };
    }
  }

  // Modals for Register Model & Create Dataset Version
  const openRegModelBtn = document.getElementById('openRegisterModelModalBtn');
  const closeRegModelBtn = document.getElementById('closeRegisterModelModalBtn');
  const cancelRegModelBtn = document.getElementById('cancelRegisterModelModalBtn');
  const submitRegModelBtn = document.getElementById('submitRegisterModelBtn');
  const regModelBackdrop = document.getElementById('registerModelModalBackdrop');

  if (openRegModelBtn) openRegModelBtn.onclick = () => { if (regModelBackdrop) regModelBackdrop.style.display = 'flex'; };
  if (closeRegModelBtn) closeRegModelBtn.onclick = () => { if (regModelBackdrop) regModelBackdrop.style.display = 'none'; };
  if (cancelRegModelBtn) cancelRegModelBtn.onclick = () => { if (regModelBackdrop) regModelBackdrop.style.display = 'none'; };
  if (submitRegModelBtn) {
    submitRegModelBtn.onclick = async () => {
      const mid = document.getElementById('newModelId')?.value.trim();
      const mname = document.getElementById('newModelName')?.value.trim();
      const arch = document.getElementById('newModelArchitecture')?.value.trim();
      const fw = document.getElementById('newModelFramework')?.value.trim();
      const wgt = document.getElementById('newModelWeights')?.value.trim();
      const tlay = document.getElementById('newModelTargetLayer')?.value.trim();
      const desc = document.getElementById('newModelDesc')?.value.trim();

      try {
        const resp = await fetch('/api/models', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            model_id: mid || undefined,
            model_name: mname || 'TorchXRayVision DenseNet-121 Custom',
            architecture: arch || 'DenseNet-121',
            framework: fw || 'PyTorch / TorchXRayVision',
            weights_identifier: wgt || 'densenet121-res224-all',
            target_layer: tlay || 'model.features.norm5',
            description: desc || ''
          })
        });
        const resData = await resp.json();
        if (!resp.ok) throw new Error(resData.error || `HTTP ${resp.status}`);
        if (regModelBackdrop) regModelBackdrop.style.display = 'none';
        loadExperimentRegistry();
        alert(`Model '${resData.model_id}' registered successfully!`);
      } catch (e) {
        alert(`Model registration error: ${e.message}`);
      }
    };
  }

  const openDsvBtn = document.getElementById('openCreateDsvModalBtn');
  const closeDsvBtn = document.getElementById('closeCreateDsvModalBtn');
  const cancelDsvBtn = document.getElementById('cancelCreateDsvModalBtn');
  const submitDsvBtn = document.getElementById('submitCreateDsvBtn');
  const dsvBackdrop = document.getElementById('createDsvModalBackdrop');

  if (openDsvBtn) openDsvBtn.onclick = () => { if (dsvBackdrop) dsvBackdrop.style.display = 'flex'; };
  if (closeDsvBtn) closeDsvBtn.onclick = () => { if (dsvBackdrop) dsvBackdrop.style.display = 'none'; };
  if (cancelDsvBtn) cancelDsvBtn.onclick = () => { if (dsvBackdrop) dsvBackdrop.style.display = 'none'; };
  if (submitDsvBtn) {
    submitDsvBtn.onclick = async () => {
      const dsvId = document.getElementById('newDsvId')?.value.trim();
      const src = document.getElementById('newDsvSource')?.value.trim();

      try {
        const resp = await fetch('/api/dataset-versions', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            dataset_version_id: dsvId || undefined,
            source_dataset: src || 'IU_XRAY'
          })
        });
        const resData = await resp.json();
        if (!resp.ok) throw new Error(resData.error || `HTTP ${resp.status}`);
        if (dsvBackdrop) dsvBackdrop.style.display = 'none';
        loadExperimentRegistry();
        alert(`Dataset version '${resData.dataset_version_id}' created with manifest SHA-256!`);
      } catch (e) {
        alert(`Dataset version creation error: ${e.message}`);
      }
    };
  }

  // =========================================================================
  // Phase 1.8: External Benchmarking & Portable Bundles
  // =========================================================================

  const tabExternalBenchmarking = document.getElementById('tabExternalBenchmarking');
  const externalBenchmarkingViewContainer = document.getElementById('externalBenchmarkingViewContainer');

  // Sub-tab Navigation
  const extSubtabs = [
    { btn: 'subtabExtDatasets', view: 'subviewExtDatasets' },
    { btn: 'subtabDatasetCompatibility', view: 'subviewDatasetCompatibility' },
    { btn: 'subtabBenchmarkRuns', view: 'subviewBenchmarkRuns' },
    { btn: 'subtabMultiViewExps', view: 'subviewMultiViewExps' },
    { btn: 'subtabExtEvaluation', view: 'subviewExtEvaluation' },
    { btn: 'subtabCrossDatasetComp', view: 'subviewCrossDatasetComp' },
    { btn: 'subtabPortableBundles', view: 'subviewPortableBundles' },
    { btn: 'subtabAirGappedVerify', view: 'subviewAirGappedVerify' }
  ];

  extSubtabs.forEach(st => {
    const btnElem = document.getElementById(st.btn);
    if (btnElem) {
      btnElem.addEventListener('click', () => {
        extSubtabs.forEach(other => {
          const ob = document.getElementById(other.btn);
          const ov = document.getElementById(other.view);
          if (ob) ob.classList.remove('active');
          if (ov) ov.style.display = 'none';
        });
        btnElem.classList.add('active');
        const targetView = document.getElementById(st.view);
        if (targetView) targetView.style.display = 'block';
      });
    }
  });

  async function loadExternalBenchmarkingDashboard() {
    try {
      const resp = await fetch('/api/external-benchmark-dashboard');
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const data = await resp.json();

      document.getElementById('bmStatExternalDatasets').textContent = data.total_external_datasets || 0;
      document.getElementById('bmStatTotalBenchmarks').textContent = data.total_benchmarks || 0;
      document.getElementById('bmStatMultiViewRuns').textContent = data.multi_view_benchmarks || 0;
      document.getElementById('bmStatVerifiedBundles').textContent = data.verified_benchmarks || 0;

      renderExternalDatasetsTable(data.external_datasets || []);
      renderBenchmarkRunsTable(data.benchmarks || []);
      populateBenchmarkComparisonDropdowns(data.benchmarks || []);
    } catch (err) {
      console.error('Failed to load benchmark dashboard:', err);
    }
  }

  function renderExternalDatasetsTable(datasets) {
    const tbody = document.getElementById('extDatasetsTableBody');
    if (!tbody) return;
    tbody.innerHTML = '';
    if (datasets.length === 0) {
      tbody.innerHTML = '<tr><td colspan="8" class="text-center text-muted">No external datasets registered yet.</td></tr>';
      return;
    }

    datasets.forEach(ds => {
      const tr = document.createElement('tr');
      const viewsBadges = (ds.image_views || []).map(v => `<span class="badge badge-info">${v}</span>`).join(' ');
      const statusClass = ds.status === 'FINALIZED' ? 'badge-finalized' : 'badge-warning';

      tr.innerHTML = `
        <td><strong>${ds.dataset_id}</strong></td>
        <td>${ds.dataset_name}</td>
        <td><code>${ds.modality}</code></td>
        <td>${viewsBadges}</td>
        <td>${ds.study_count} (${ds.image_count} imgs)</td>
        <td><span class="badge ${statusClass}">${ds.status}</span></td>
        <td><code title="${ds.manifest_sha256}">${(ds.manifest_sha256 || '').substring(0, 16)}...</code></td>
        <td>
          <div class="action-btn-group">
            ${ds.status !== 'FINALIZED' ? `<button class="btn btn-xs btn-primary" onclick="triggerFinalizeExtDataset('${ds.dataset_id}')">Finalize</button>` : `<span class="badge badge-neutral">Locked</span>`}
          </div>
        </td>
      `;
      tbody.appendChild(tr);
    });
  }

  window.triggerFinalizeExtDataset = async function(datasetId) {
    try {
      const resp = await fetch(`/api/external-datasets/${datasetId}/finalize`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.error || `HTTP ${resp.status}`);
      loadExternalBenchmarkingDashboard();
      alert(`External dataset '${datasetId}' finalized & frozen!`);
    } catch (e) {
      alert(`Finalize error: ${e.message}`);
    }
  };

  function renderBenchmarkRunsTable(benchmarks) {
    const tbody = document.getElementById('benchmarkRunsTableBody');
    if (!tbody) return;
    tbody.innerHTML = '';
    if (benchmarks.length === 0) {
      tbody.innerHTML = '<tr><td colspan="8" class="text-center text-muted">No benchmark runs registered yet.</td></tr>';
      return;
    }

    benchmarks.forEach(bm => {
      const tr = document.createElement('tr');
      const statusClass = bm.status === 'FINALIZED' ? 'badge-finalized' : (bm.status === 'EXECUTED' ? 'badge-success' : 'badge-warning');
      const repBadge = bm.reproducibility_status === 'VERIFIED' ? '<span class="badge badge-success">✓ Verified</span>' : '<span class="badge badge-warning">Unverified</span>';

      tr.innerHTML = `
        <td><strong>${bm.benchmark_id}</strong></td>
        <td>${bm.benchmark_name}</td>
        <td><code>${bm.modality}</code></td>
        <td><span class="badge badge-info">${bm.view_configuration}</span></td>
        <td><span class="badge ${statusClass}">${bm.status}</span></td>
        <td>${repBadge}</td>
        <td><span style="font-size: 0.75rem;">${(bm.created_at || '').split('T')[0]}</span></td>
        <td>
          <div class="action-btn-group">
            ${bm.status === 'REGISTERED' ? `<button class="btn btn-xs btn-primary" onclick="triggerRunBenchmark('${bm.benchmark_id}')">Run</button>` : ''}
            ${bm.status === 'EXECUTED' ? `<button class="btn btn-xs btn-success" onclick="triggerFinalizeBenchmark('${bm.benchmark_id}')">Finalize</button>` : ''}
            <button class="btn btn-xs btn-secondary" onclick="triggerExportBenchmark('${bm.benchmark_id}', 'json')">Export</button>
          </div>
        </td>
      `;
      tbody.appendChild(tr);
    });
  }

  window.triggerRunBenchmark = async function(benchmarkId) {
    try {
      const resp = await fetch(`/api/benchmarks/${benchmarkId}/run`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.error || `HTTP ${resp.status}`);
      loadExternalBenchmarkingDashboard();
      alert(`Benchmark '${benchmarkId}' executed successfully!`);
    } catch (e) {
      alert(`Benchmark execution error: ${e.message}`);
    }
  };

  window.triggerFinalizeBenchmark = async function(benchmarkId) {
    try {
      const resp = await fetch(`/api/benchmarks/${benchmarkId}/finalize`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.error || `HTTP ${resp.status}`);
      loadExternalBenchmarkingDashboard();
      alert(`Benchmark '${benchmarkId}' finalized & permanently locked!`);
    } catch (e) {
      alert(`Finalize error: ${e.message}`);
    }
  };

  window.triggerExportBenchmark = async function(benchmarkId, format) {
    try {
      const resp = await fetch(`/api/benchmarks/${benchmarkId}/export?format=${format}`);
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const text = await resp.text();
      const blob = new Blob([text], { type: format === 'json' ? 'application/json' : 'text/plain' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${benchmarkId}_benchmark_report.${format === 'json' ? 'json' : 'txt'}`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (e) {
      alert(`Export error: ${e.message}`);
    }
  };

  function populateBenchmarkComparisonDropdowns(benchmarks) {
    const selA = document.getElementById('compareBenchSelectA');
    const selB = document.getElementById('compareBenchSelectB');
    if (!selA || !selB) return;
    selA.innerHTML = '';
    selB.innerHTML = '';

    benchmarks.forEach(b => {
      const optA = document.createElement('option');
      optA.value = b.benchmark_id;
      optA.textContent = `${b.benchmark_name} (${b.view_configuration})`;
      selA.appendChild(optA);

      const optB = document.createElement('option');
      optB.value = b.benchmark_id;
      optB.textContent = `${b.benchmark_name} (${b.view_configuration})`;
      selB.appendChild(optB);
    });

    if (selB.options.length > 1) selB.selectedIndex = 1;

    const compBtn = document.getElementById('runBenchComparisonBtn');
    if (compBtn) {
      compBtn.onclick = async () => {
        const idA = selA.value;
        const idB = selB.value;
        if (!idA || !idB) return;
        try {
          const resp = await fetch('/api/experiments/compare-external', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ benchmark_id_a: idA, benchmark_id_b: idB })
          });
          const compData = await resp.json();
          if (!resp.ok) throw new Error(compData.error || `HTTP ${resp.status}`);

          const container = document.getElementById('benchComparisonResultContainer');
          if (container) {
            let warnHtml = '';
            if (compData.compatibility_warnings && compData.compatibility_warnings.length > 0) {
              warnHtml = `<div class="warning-callout" style="margin-bottom: 1rem;"><strong>Compatibility Notice:</strong> ${compData.compatibility_warnings.join('<br>')}</div>`;
            }

            let rows = '';
            (compData.metric_deltas || []).forEach(md => {
              rows += `
                <tr>
                  <td><strong>${md.metric_name}</strong></td>
                  <td><code>${md.value_a !== undefined ? md.value_a : '-'}</code></td>
                  <td><code>${md.value_b !== undefined ? md.value_b : '-'}</code></td>
                  <td><code>${md.absolute_delta >= 0 ? '+' : ''}${md.absolute_delta}</code></td>
                  <td><code>${md.relative_delta !== null ? (md.relative_delta * 100).toFixed(2) + '%' : '-'}</code></td>
                  <td><span class="badge badge-info">${md.observation}</span></td>
                </tr>
              `;
            });

            container.innerHTML = `
              ${warnHtml}
              <table class="data-table">
                <thead>
                  <tr>
                    <th>Metric</th>
                    <th>Run A</th>
                    <th>Run B</th>
                    <th>Absolute Delta</th>
                    <th>Relative Delta</th>
                    <th>Observation</th>
                  </tr>
                </thead>
                <tbody>${rows}</tbody>
              </table>
            `;
          }
        } catch (e) {
          alert(`Benchmark comparison failed: ${e.message}`);
        }
      };
    }
  }

  // Air-Gapped Reproducibility Verification Handler
  const runVerifyBtn = document.getElementById('runBundleVerifyBtn');
  if (runVerifyBtn) {
    runVerifyBtn.onclick = async () => {
      const bid = document.getElementById('verifyBundleInput')?.value.trim();
      if (!bid) {
        alert('Please enter a valid Bundle ID.');
        return;
      }
      try {
        const resp = await fetch('/api/bundles/verify', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ bundle_id: bid })
        });
        const resData = await resp.json();
        const container = document.getElementById('bundleVerifyResultContainer');
        if (container) {
          const statusClass = resData.is_reproducible ? 'badge-success' : 'badge-error';
          container.innerHTML = `
            <div class="p-4 border rounded mt-3">
              <h4>Reproducibility Verification: <span class="badge ${statusClass}">${resData.status}</span></h4>
              <p><strong>Stored Fingerprint:</strong> <code>${resData.stored_fingerprint_sha256 || '-'}</code></p>
              <p><strong>Recalculated Fingerprint:</strong> <code>${resData.recalculated_fingerprint_sha256 || '-'}</code></p>
              <p><strong>Verified Files:</strong> ${resData.verified_files} / ${resData.file_count}</p>
              <div class="badge badge-neutral mt-2">${resData.disclaimer}</div>
            </div>
          `;
        }
      } catch (e) {
        alert(`Verification failed: ${e.message}`);
      }
    };
  }

  // Phase 1.8 Modals Handlers
  const openExtDsModalBtn = document.getElementById('openRegisterExtDsModalBtn');
  const closeExtDsModalBtn = document.getElementById('closeRegisterExtDsModalBtn');
  const cancelExtDsModalBtn = document.getElementById('cancelRegisterExtDsModalBtn');
  const submitExtDsBtn = document.getElementById('submitRegisterExtDsBtn');
  const extDsBackdrop = document.getElementById('registerExtDsModalBackdrop');

  if (openExtDsModalBtn) openExtDsModalBtn.onclick = () => { if (extDsBackdrop) extDsBackdrop.style.display = 'flex'; };
  if (closeExtDsModalBtn) closeExtDsModalBtn.onclick = () => { if (extDsBackdrop) extDsBackdrop.style.display = 'none'; };
  if (cancelExtDsModalBtn) cancelExtDsModalBtn.onclick = () => { if (extDsBackdrop) extDsBackdrop.style.display = 'none'; };
  if (submitExtDsBtn) {
    submitExtDsBtn.onclick = async () => {
      const dsId = document.getElementById('newExtDsId')?.value.trim();
      const dsName = document.getElementById('newExtDsName')?.value.trim();
      const mod = document.getElementById('newExtDsModality')?.value;
      const status = document.getElementById('newExtDsStatus')?.value;

      try {
        const resp = await fetch('/api/external-datasets', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            dataset_id: dsId || undefined,
            dataset_name: dsName || 'External Test Dataset',
            modality: mod || 'CHEST_XRAY',
            access_status: status || 'SYNTHETIC_FIXTURE'
          })
        });
        const resData = await resp.json();
        if (!resp.ok) throw new Error(resData.error || `HTTP ${resp.status}`);
        if (extDsBackdrop) extDsBackdrop.style.display = 'none';
        loadExternalBenchmarkingDashboard();
        alert(`External dataset '${resData.dataset_id}' registered successfully!`);
      } catch (e) {
        alert(`Registration error: ${e.message}`);
      }
    };
  }

  const openBmModalBtn = document.getElementById('openCreateBenchmarkModalBtn');
  const closeBmModalBtn = document.getElementById('closeCreateBenchmarkModalBtn');
  const cancelBmModalBtn = document.getElementById('cancelCreateBenchmarkModalBtn');
  const submitBmBtn = document.getElementById('submitCreateBenchmarkBtn');
  const bmBackdrop = document.getElementById('createBenchmarkModalBackdrop');

  if (openBmModalBtn) openBmModalBtn.onclick = () => { if (bmBackdrop) bmBackdrop.style.display = 'flex'; };
  if (closeBmModalBtn) closeBmModalBtn.onclick = () => { if (bmBackdrop) bmBackdrop.style.display = 'none'; };
  if (cancelBmModalBtn) cancelBmModalBtn.onclick = () => { if (bmBackdrop) bmBackdrop.style.display = 'none'; };
  if (submitBmBtn) {
    submitBmBtn.onclick = async () => {
      const bmId = document.getElementById('newBmId')?.value.trim();
      const bmName = document.getElementById('newBmName')?.value.trim();
      const mod = document.getElementById('newBmModality')?.value;
      const viewCfg = document.getElementById('newBmViewConfig')?.value;

      try {
        const resp = await fetch('/api/benchmarks', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            benchmark_id: bmId || undefined,
            benchmark_name: bmName || 'Multi-View Benchmark',
            modality: mod || 'INTERNAL_BENCHMARK',
            view_configuration: viewCfg || 'SINGLE_VIEW'
          })
        });
        const resData = await resp.json();
        if (!resp.ok) throw new Error(resData.error || `HTTP ${resp.status}`);
        if (bmBackdrop) bmBackdrop.style.display = 'none';
        loadExternalBenchmarkingDashboard();
        alert(`Benchmark '${resData.benchmark_id}' registered successfully!`);
      } catch (e) {
        alert(`Benchmark registration error: ${e.message}`);
      }
    };
  }

  const openBundleModalBtn = document.getElementById('openCreateBundleModalBtn');
  const closeBundleModalBtn = document.getElementById('closeCreateBundleModalBtn');
  const cancelBundleModalBtn = document.getElementById('cancelCreateBundleModalBtn');
  const submitBundleBtn = document.getElementById('submitCreateBundleBtn');
  const bundleBackdrop = document.getElementById('createBundleModalBackdrop');

  if (openBundleModalBtn) openBundleModalBtn.onclick = () => { if (bundleBackdrop) bundleBackdrop.style.display = 'flex'; };
  if (closeBundleModalBtn) closeBundleModalBtn.onclick = () => { if (bundleBackdrop) bundleBackdrop.style.display = 'none'; };
  if (cancelBundleModalBtn) cancelBundleModalBtn.onclick = () => { if (bundleBackdrop) bundleBackdrop.style.display = 'none'; };
  if (submitBundleBtn) {
    submitBundleBtn.onclick = async () => {
      const expId = document.getElementById('newBundleExpId')?.value.trim();
      const bId = document.getElementById('newBundleId')?.value.trim();
      if (!expId) {
        alert('Source Experiment ID is required.');
        return;
      }
      try {
        const resp = await fetch(`/api/experiments/${expId}/bundle`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ bundle_id: bId || undefined })
        });
        const resData = await resp.json();
        if (!resp.ok) throw new Error(resData.error || `HTTP ${resp.status}`);
        if (bundleBackdrop) bundleBackdrop.style.display = 'none';
        alert(`Portable experiment bundle '${resData.bundle_id}' created with manifest & files!`);
      } catch (e) {
        alert(`Bundle creation error: ${e.message}`);
      }
    };
  }

  // =========================================================================
  // Phase 1.9: Research Experiment Orchestration & Reproducibility Dashboard
  // =========================================================================

  const tabResearchExperiments = document.getElementById('tabResearchExperiments');
  const researchExperimentsContainer = document.getElementById('researchExperimentsContainer');
  let selectedPhase19ExpId = null;

  // Phase 1.9 Sub-tab navigation
  const p19Subtabs = [
    { btn: 'p19SubTabRegistry', view: 'p19ExpRegistrySubView' },
    { btn: 'p19SubTabBuilder', view: 'p19ExpBuilderSubView' },
    { btn: 'p19SubTabResults', view: 'p19ExpResultsSubView' },
    { btn: 'p19SubTabComparison', view: 'p19ExpComparisonSubView' },
    { btn: 'p19SubTabReproducibility', view: 'p19ReproducibilitySubView' },
    { btn: 'p19SubTabReport', view: 'p19ResearchReportSubView' },
    { btn: 'p19SubTabProvenance', view: 'p19ProvenanceSubView' }
  ];

  p19Subtabs.forEach(st => {
    const btnElem = document.getElementById(st.btn);
    if (btnElem) {
      btnElem.addEventListener('click', () => {
        p19Subtabs.forEach(other => {
          const ob = document.getElementById(other.btn);
          const ov = document.getElementById(other.view);
          if (ob) ob.classList.remove('active');
          if (ov) ov.style.display = 'none';
        });
        btnElem.classList.add('active');
        const targetView = document.getElementById(st.view);
        if (targetView) targetView.style.display = 'block';
      });
    }
  });

  async function loadPhase19Dashboard() {
    try {
      const resp = await fetch('/api/phase19/dashboard');
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const data = await resp.json();
      const experiments = data.experiments || [];

      const badge = document.getElementById('p19ExpCountBadge');
      if (badge) badge.textContent = `${experiments.length} experiments (${data.published_experiments || 0} published)`;

      renderP19ExpTable(experiments);
      populateP19CompareDropdowns(experiments);
    } catch (err) {
      console.error('Failed to load Phase 1.9 dashboard:', err);
    }
  }

  function renderP19ExpTable(experiments) {
    const tbody = document.getElementById('p19ExpTableBody');
    if (!tbody) return;
    tbody.innerHTML = '';

    if (experiments.length === 0) {
      tbody.innerHTML = '<tr><td colspan="9" style="text-align: center; color: #94a3b8;">No research experiments registered. Click "New Experiment" to configure one.</td></tr>';
      return;
    }

    experiments.forEach(exp => {
      const tr = document.createElement('tr');
      const eid = exp.experiment_id || 'exp_unknown';
      const name = exp.experiment_name || 'Unnamed';
      const ds = exp.dataset_ref ? exp.dataset_ref.dataset_name : (exp.dataset_snapshot_id || 'IU-Xray');
      const model = exp.model_ref ? exp.model_ref.model_name : (exp.model_name || 'DenseNet-121');
      const viewCfg = exp.configuration ? exp.configuration.view_configuration : 'SINGLE_VIEW';
      const st = (exp.status || 'CONFIGURED').toUpperCase();
      const repro = exp.reproducibility_manifest ? exp.reproducibility_manifest.reproducibility_status : 'REPRODUCIBLE';
      const created = exp.created_at ? exp.created_at.substring(0, 16).replace('T', ' ') : '-';

      let stColor = '#94a3b8';
      if (st === 'COMPLETED') stColor = '#38bdf8';
      else if (st === 'VALIDATED') stColor = '#a78bfa';
      else if (st === 'PUBLISHED') stColor = '#34d399';

      let reproBadge = `<span class="badge" style="background: rgba(52, 211, 153, 0.2); color: #34d399;">✓ MATCH</span>`;
      if (repro === 'DRIFT_DETECTED') {
        reproBadge = `<span class="badge" style="background: rgba(251, 146, 60, 0.2); color: #fb923c;">⚠ DRIFT</span>`;
      }

      tr.innerHTML = `
        <td><code>${eid}</code></td>
        <td><strong>${name}</strong></td>
        <td>${ds}</td>
        <td><span class="info-tag">${model}</span></td>
        <td><span class="info-tag">${viewCfg}</span></td>
        <td><span class="badge" style="background: rgba(255,255,255,0.05); color: ${stColor}; border: 1px solid ${stColor};">${st}</span></td>
        <td>${reproBadge}</td>
        <td style="font-size: 0.75rem; color: #94a3b8;">${created}</td>
        <td>
          <div style="display: flex; gap: 4px;">
            <button class="btn btn-sm btn-secondary btn-p19-select" data-eid="${eid}">Inspect</button>
            ${st !== 'PUBLISHED' && st !== 'ARCHIVED' ? `<button class="btn btn-sm btn-primary btn-p19-run" data-eid="${eid}">▶ Run</button>` : ''}
          </div>
        </td>
      `;

      tr.querySelector('.btn-p19-select').onclick = () => selectPhase19Experiment(eid);
      const runBtn = tr.querySelector('.btn-p19-run');
      if (runBtn) runBtn.onclick = () => runPhase19Experiment(eid);

      tbody.appendChild(tr);
    });
  }

  async function selectPhase19Experiment(experimentId) {
    selectedPhase19ExpId = experimentId;
    try {
      const resp = await fetch(`/api/phase19/experiments/${experimentId}`);
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const exp = await resp.json();

      // Update Results view header
      document.getElementById('p19ResultsExpTitle').textContent = `Experiment: ${exp.experiment_name || experimentId}`;
      document.getElementById('p19ResultsExpMeta').textContent = `ID: ${experimentId} | Status: ${exp.status} | Dataset: ${exp.dataset_ref ? exp.dataset_ref.dataset_name : '-'} | Model: ${exp.model_ref ? exp.model_ref.model_name : '-'}`;

      // Update Metrics Summary
      const macro = (exp.metrics && exp.metrics.macro_avg) || {};
      const stats = (exp.statistical_summary && exp.statistical_summary.metrics) || {};

      document.getElementById('p19ResAuc').textContent = macro.auc !== undefined ? macro.auc.toFixed(4) : '—';
      const aucCi = stats.auc && stats.auc.ci_95 ? `95% CI: [${stats.auc.ci_95.lower.toFixed(4)} – ${stats.auc.ci_95.upper.toFixed(4)}]` : '95% CI: —';
      document.getElementById('p19ResAucCi').textContent = aucCi;

      document.getElementById('p19ResSens').textContent = macro.sensitivity !== undefined ? macro.sensitivity.toFixed(4) : '—';
      const sensCi = stats.sensitivity && stats.sensitivity.ci_95 ? `95% CI: [${stats.sensitivity.ci_95.lower.toFixed(4)} – ${stats.sensitivity.ci_95.upper.toFixed(4)}]` : '95% CI: —';
      document.getElementById('p19ResSensCi').textContent = sensCi;

      document.getElementById('p19ResSpec').textContent = macro.specificity !== undefined ? macro.specificity.toFixed(4) : '—';
      const specCi = stats.specificity && stats.specificity.ci_95 ? `95% CI: [${stats.specificity.ci_95.lower.toFixed(4)} – ${stats.specificity.ci_95.upper.toFixed(4)}]` : '95% CI: —';
      document.getElementById('p19ResSpecCi').textContent = specCi;

      document.getElementById('p19ResF1').textContent = macro.f1 !== undefined ? macro.f1.toFixed(4) : '—';
      const f1Ci = stats.f1 && stats.f1.ci_95 ? `95% CI: [${stats.f1.ci_95.lower.toFixed(4)} – ${stats.f1.ci_95.upper.toFixed(4)}]` : '95% CI: —';
      document.getElementById('p19ResF1Ci').textContent = f1Ci;

      // Render Per-Finding Table
      const pfTbody = document.getElementById('p19PerFindingTableBody');
      if (pfTbody) {
        pfTbody.innerHTML = '';
        const pf = (exp.metrics && exp.metrics.per_finding) || {};
        if (Object.keys(pf).length === 0) {
          pfTbody.innerHTML = '<tr><td colspan="6" style="text-align: center; color: #94a3b8;">No per-finding metrics recorded. Run benchmark to compute.</td></tr>';
        } else {
          Object.entries(pf).forEach(([find, m]) => {
            const row = document.createElement('tr');
            row.innerHTML = `
              <td><strong>${find}</strong></td>
              <td><code>${m.auc !== undefined ? m.auc.toFixed(4) : '-'}</code></td>
              <td><code>${m.sensitivity !== undefined ? m.sensitivity.toFixed(4) : '-'}</code></td>
              <td><code>${m.specificity !== undefined ? m.specificity.toFixed(4) : '-'}</code></td>
              <td><code>${m.f1 !== undefined ? m.f1.toFixed(4) : '-'}</code></td>
              <td><code>${m.accuracy !== undefined ? m.accuracy.toFixed(4) : '-'}</code></td>
            `;
            pfTbody.appendChild(row);
          });
        }
      }

      // Update Reproducibility Hashes
      const repro = exp.reproducibility_manifest || {};
      document.getElementById('p19Hds').textContent = repro.dataset_fingerprint || '—';
      document.getElementById('p19Hmod').textContent = repro.model_fingerprint || '—';
      document.getElementById('p19Hcfg').textContent = repro.configuration_fingerprint || '—';
      document.getElementById('p19Henv').textContent = repro.environment_fingerprint || '—';
      document.getElementById('p19Hcan').textContent = repro.experiment_fingerprint || '—';

      // Load Research Report text
      loadPhase19Report(experimentId);

      // Switch to Results sub-tab automatically
      document.getElementById('p19SubTabResults').click();

    } catch (err) {
      alert(`Failed to load experiment: ${err.message}`);
    }
  }

  async function loadPhase19Report(experimentId) {
    try {
      const resp = await fetch(`/api/phase19/reports/${experimentId}`);
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const rep = await resp.json();
      const container = document.getElementById('p19ReportContainer');
      if (container) {
        let text = `============================================================\n`;
        text += `RESEARCH EXPERIMENT REPORT: ${rep.experiment_name}\n`;
        text += `ID: ${rep.experiment_id} | Created: ${rep.created_at}\n`;
        text += `============================================================\n\n`;
        text += `DISCLAIMER:\n${rep.research_disclaimer}\n\n`;
        Object.entries(rep.sections || {}).forEach(([secKey, secVal]) => {
          text += `------------------------------------------------------------\n`;
          text += `[SECTION: ${secKey.toUpperCase()}]\n`;
          text += `------------------------------------------------------------\n`;
          text += `${secVal}\n\n`;
        });
        container.textContent = text;
      }
    } catch (err) {
      console.warn('Could not load report:', err);
    }
  }

  async function runPhase19Experiment(experimentId) {
    try {
      const resp = await fetch(`/api/phase19/experiments/${experimentId}/run`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({})
      });
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.error || `HTTP ${resp.status}`);
      alert(`Experiment '${experimentId}' benchmark execution completed!`);
      await loadPhase19Dashboard();
      await selectPhase19Experiment(experimentId);
    } catch (err) {
      alert(`Execution failed: ${err.message}`);
    }
  }

  async function finalizePhase19Experiment(experimentId) {
    try {
      const resp = await fetch(`/api/phase19/experiments/${experimentId}/finalize`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({})
      });
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.error || `HTTP ${resp.status}`);
      alert(`Experiment '${experimentId}' finalized and validated!`);
      await loadPhase19Dashboard();
      await selectPhase19Experiment(experimentId);
    } catch (err) {
      alert(`Finalization failed: ${err.message}`);
    }
  }

  async function publishPhase19Experiment(experimentId) {
    if (!confirm(`Publishing experiment '${experimentId}' will permanently lock it against any future modifications. Proceed?`)) return;
    try {
      const resp = await fetch(`/api/phase19/experiments/${experimentId}/publish`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({})
      });
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.error || `HTTP ${resp.status}`);
      alert(`Experiment '${experimentId}' successfully published and permanently locked!`);
      await loadPhase19Dashboard();
      await selectPhase19Experiment(experimentId);
    } catch (err) {
      alert(`Publishing failed: ${err.message}`);
    }
  }

  function populateP19CompareDropdowns(experiments) {
    const selA = document.getElementById('p19CompareExpA');
    const selB = document.getElementById('p19CompareExpB');
    if (!selA || !selB) return;
    selA.innerHTML = '';
    selB.innerHTML = '';

    experiments.forEach(exp => {
      const optA = document.createElement('option');
      optA.value = exp.experiment_id;
      optA.textContent = `${exp.experiment_name || exp.experiment_id} (${exp.status})`;
      selA.appendChild(optA);

      const optB = document.createElement('option');
      optB.value = exp.experiment_id;
      optB.textContent = `${exp.experiment_name || exp.experiment_id} (${exp.status})`;
      selB.appendChild(optB);
    });

    if (selB.options.length > 1) selB.selectedIndex = 1;
  }

  // Hook Phase 1.9 Results Action Buttons
  const p19RunBtn = document.getElementById('p19RunSelectedExpBtn');
  if (p19RunBtn) p19RunBtn.onclick = () => { if (selectedPhase19ExpId) runPhase19Experiment(selectedPhase19ExpId); else alert('Please select an experiment first.'); };

  const p19FinBtn = document.getElementById('p19FinalizeExpBtn');
  if (p19FinBtn) p19FinBtn.onclick = () => { if (selectedPhase19ExpId) finalizePhase19Experiment(selectedPhase19ExpId); else alert('Please select an experiment first.'); };

  const p19PubBtn = document.getElementById('p19PublishExpBtn');
  if (p19PubBtn) p19PubBtn.onclick = () => { if (selectedPhase19ExpId) publishPhase19Experiment(selectedPhase19ExpId); else alert('Please select an experiment first.'); };

  // Hook Phase 1.9 Builder Submit
  const bldSubmitBtn = document.getElementById('bldSubmitCreateBtn');
  if (bldSubmitBtn) {
    bldSubmitBtn.onclick = async () => {
      const expId = document.getElementById('bldExpId')?.value.trim();
      const expName = document.getElementById('bldExpName')?.value.trim();
      const expDesc = document.getElementById('bldExpDesc')?.value.trim();
      const dsId = document.getElementById('bldExpDataset')?.value;
      const viewCfg = document.getElementById('bldExpViewConfig')?.value;
      const modelId = document.getElementById('bldExpModel')?.value;
      const bootstraps = parseInt(document.getElementById('bldExpBootstraps')?.value || '1000', 10);
      const seed = parseInt(document.getElementById('bldExpSeed')?.value || '42', 10);
      const thresh = parseFloat(document.getElementById('bldExpThreshold')?.value || '0.5');

      if (!expName) {
        alert('Experiment Name is required.');
        return;
      }

      const payload = {
        experiment_id: expId || undefined,
        experiment_name: expName,
        description: expDesc,
        dataset_ref: {
          dataset_id: dsId,
          dataset_name: dsId === 'iu_xray_default' ? 'IU-Xray Default Research Split' : dsId,
          dataset_type: dsId === 'iu_xray_default' ? 'IU_XRAY' : 'EXTERNAL',
          version: '1.0.0',
          split: 'test',
          sample_count: 100,
          manifest_sha256: 'manual_builder_manifest_sha256'
        },
        model_ref: {
          model_id: modelId,
          model_name: modelId === 'txrv_densenet121' ? 'TorchXRayVision DenseNet-121' : 'TorchXRayVision ResNet-50',
          model_architecture: modelId === 'txrv_densenet121' ? 'densenet121' : 'resnet50'
        },
        configuration: {
          view_configuration: viewCfg,
          bootstrap_sample_count: bootstraps,
          random_seed: seed,
          threshold: thresh
        }
      };

      try {
        const resp = await fetch('/api/phase19/experiments', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        const data = await resp.json();
        if (!resp.ok) throw new Error(data.error || `HTTP ${resp.status}`);
        alert(`Experiment '${data.experiment_id}' created successfully!`);
        await loadPhase19Dashboard();
        selectPhase19Experiment(data.experiment_id);
      } catch (err) {
        alert(`Creation failed: ${err.message}`);
      }
    };
  }

  // Hook Phase 1.9 Compare Modal
  const openCompModalBtn = document.getElementById('p19OpenCompareModalBtn');
  const closeCompModalBtn = document.getElementById('closeP19CompareModalBtn');
  const cancelCompModalBtn = document.getElementById('cancelP19CompareModalBtn');
  const submitCompBtn = document.getElementById('submitP19CompareBtn');
  const compBackdrop = document.getElementById('p19CompareModalBackdrop');

  if (openCompModalBtn) openCompModalBtn.onclick = () => { if (compBackdrop) compBackdrop.style.display = 'flex'; };
  if (closeCompModalBtn) closeCompModalBtn.onclick = () => { if (compBackdrop) compBackdrop.style.display = 'none'; };
  if (cancelCompModalBtn) cancelCompModalBtn.onclick = () => { if (compBackdrop) compBackdrop.style.display = 'none'; };

  if (submitCompBtn) {
    submitCompBtn.onclick = async () => {
      const idA = document.getElementById('p19CompareExpA')?.value;
      const idB = document.getElementById('p19CompareExpB')?.value;
      if (!idA || !idB || idA === idB) {
        alert('Please select two distinct experiments to compare.');
        return;
      }
      try {
        const resp = await fetch('/api/phase19/experiments/compare', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ experiment_ids: [idA, idB] })
        });
        const comp = await resp.json();
        if (!resp.ok) throw new Error(comp.error || `HTTP ${resp.status}`);
        if (compBackdrop) compBackdrop.style.display = 'none';

        // Render comparison table
        document.getElementById('p19ComparisonSummaryText').textContent = comp.neutral_summary || 'Comparative differences calculated.';
        const tbody = document.getElementById('p19ComparisonTableBody');
        if (tbody) {
          tbody.innerHTML = '';
          Object.entries(comp.metric_deltas || {}).forEach(([metric, delta]) => {
            const row = document.createElement('tr');
            const sign = delta.delta_abs >= 0 ? '+' : '';
            row.innerHTML = `
              <td><strong>${metric.toUpperCase()}</strong></td>
              <td><code>${delta.baseline_value.toFixed(4)}</code></td>
              <td><code>${delta.comparator_value.toFixed(4)}</code></td>
              <td><code>${sign}${delta.delta_abs.toFixed(4)}</code></td>
              <td><code>${sign}${delta.delta_rel.toFixed(2)}%</code></td>
            `;
            tbody.appendChild(row);
          });
        }
        document.getElementById('p19SubTabComparison').click();
      } catch (err) {
        alert(`Comparison failed: ${err.message}`);
      }
    };
  }

  // Hook Phase 1.9 Export Buttons
  const p19ExpJsonBtn = document.getElementById('p19ExportJsonBtn');
  if (p19ExpJsonBtn) p19ExpJsonBtn.onclick = () => {
    if (!selectedPhase19ExpId) return alert('Select an experiment first.');
    window.open(`/api/phase19/experiments/${selectedPhase19ExpId}/export?format=json`, '_blank');
  };
  const p19ExpCsvBtn = document.getElementById('p19ExportCsvBtn');
  if (p19ExpCsvBtn) p19ExpCsvBtn.onclick = () => {
    if (!selectedPhase19ExpId) return alert('Select an experiment first.');
    window.open(`/api/phase19/experiments/${selectedPhase19ExpId}/export?format=csv`, '_blank');
  };
  const p19ExpTxtBtn = document.getElementById('p19ExportTextBtn');
  if (p19ExpTxtBtn) p19ExpTxtBtn.onclick = () => {
    if (!selectedPhase19ExpId) return alert('Select an experiment first.');
    window.open(`/api/phase19/experiments/${selectedPhase19ExpId}/export?format=text`, '_blank');
  };
  const p19ExpMdBtn = document.getElementById('p19ExportMdBtn');
  // ============================================================
  // PHASE 2.0 — INTERACTIVE COUNTERFACTUAL EXPLAINABILITY
  // ============================================================
  const tabCounterfactualBtn = document.getElementById('tabCounterfactualExplainability');
  const counterfactualViewContainer = document.getElementById('counterfactualViewContainer');

  const cfState = {
    studyId: 'CXR1122',
    sourceView: 'PA',
    experiment: null,
    roi: { x: 50, y: 50, width: 80, height: 80 },
    isDrawingRoi: false,
    drawStartX: 0,
    drawStartY: 0,
    originalImg: null,
    baseHeatmapImg: null,
    perturbedImg: null,
    cfHeatmapImg: null,
    diffHeatmapImg: null
  };

  async function loadPhase20Dashboard() {
    try {
      const resp = await fetch('/api/phase20/dashboard');
      if (!resp.ok) return;
      const data = await resp.json();
      const setEl = (id, val) => { const el = document.getElementById(id); if (el) el.textContent = val; };
      setEl('cfStatTotal', data.total_experiments || 0);
      setEl('cfStatCompleted', data.completed || 0);
      setEl('cfStatValidated', data.validated || 0);
      setEl('cfStatPublished', data.published || 0);
      setEl('cfStatReproducible', data.reproducible || 0);
      setEl('cfStatDrift', data.drift_detected || 0);
    } catch (err) {
      console.warn('Phase 2.0 dashboard load error:', err);
    }
  }

  const origCanvas = document.getElementById('cfOriginalCanvas');
  const pertCanvas = document.getElementById('cfPerturbedCanvas');
  const diffCanvas = document.getElementById('cfDiffCanvas');

  function initPhase20Workspace() {
    // Populate study dropdown
    const studySel = document.getElementById('cfStudySelect');
    if (studySel && state.allStudies && state.allStudies.length > 0) {
      studySel.innerHTML = '';
      state.allStudies.forEach(s => {
        const opt = document.createElement('option');
        opt.value = s.study_id;
        opt.textContent = s.study_id;
        if (s.study_id === cfState.studyId) opt.selected = true;
        studySel.appendChild(opt);
      });
    }

    // Bind slider
    const slider = document.getElementById('cfStrengthSlider');
    const sliderVal = document.getElementById('cfStrengthVal');
    if (slider && sliderVal) {
      slider.oninput = () => { sliderVal.textContent = parseFloat(slider.value).toFixed(1); };
    }

    // Load initial source image for canvas
    loadPhase20SourceImage();
  }

  function loadPhase20SourceImage() {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.src = `/api/studies/${cfState.studyId}/image`;
    img.onload = () => {
      cfState.originalImg = img;
      renderPhase20OriginalCanvas();
    };
  }

  function renderPhase20OriginalCanvas() {
    if (!origCanvas || !cfState.originalImg) return;
    const ctx = origCanvas.getContext('2d');
    ctx.clearRect(0, 0, origCanvas.width, origCanvas.height);
    ctx.drawImage(cfState.originalImg, 0, 0, origCanvas.width, origCanvas.height);

    // Draw baseline heatmap overlay if available and toggled
    const toggleHm = document.getElementById('cfToggleBaseHeatmap');
    const opVal = parseFloat(document.getElementById('cfBaseOpacity')?.value || '0.6');
    if (toggleHm && toggleHm.checked && cfState.baseHeatmapImg) {
      ctx.globalAlpha = opVal;
      ctx.drawImage(cfState.baseHeatmapImg, 0, 0, origCanvas.width, origCanvas.height);
      ctx.globalAlpha = 1.0;
    }

    // Draw ROI bounding box
    if (cfState.roi) {
      ctx.strokeStyle = '#38bdf8';
      ctx.lineWidth = 2;
      ctx.fillStyle = 'rgba(56, 189, 248, 0.2)';
      ctx.strokeRect(cfState.roi.x, cfState.roi.y, cfState.roi.width, cfState.roi.height);
      ctx.fillRect(cfState.roi.x, cfState.roi.y, cfState.roi.width, cfState.roi.height);

      ctx.fillStyle = '#38bdf8';
      ctx.font = '10px monospace';
      ctx.fillText(`ROI (${cfState.roi.width}x${cfState.roi.height})`, cfState.roi.x + 4, Math.max(14, cfState.roi.y - 4));
    }
  }

  function renderPhase20PerturbedCanvas() {
    if (!pertCanvas) return;
    const ctx = pertCanvas.getContext('2d');
    ctx.clearRect(0, 0, pertCanvas.width, pertCanvas.height);
    if (cfState.perturbedImg) {
      ctx.drawImage(cfState.perturbedImg, 0, 0, pertCanvas.width, pertCanvas.height);
    } else if (cfState.originalImg) {
      ctx.drawImage(cfState.originalImg, 0, 0, pertCanvas.width, pertCanvas.height);
    }

    const toggleHm = document.getElementById('cfToggleCfHeatmap');
    const opVal = parseFloat(document.getElementById('cfCfOpacity')?.value || '0.6');
    if (toggleHm && toggleHm.checked && cfState.cfHeatmapImg) {
      ctx.globalAlpha = opVal;
      ctx.drawImage(cfState.cfHeatmapImg, 0, 0, pertCanvas.width, pertCanvas.height);
      ctx.globalAlpha = 1.0;
    }
  }

  function renderPhase20DiffCanvas() {
    if (!diffCanvas) return;
    const ctx = diffCanvas.getContext('2d');
    ctx.clearRect(0, 0, diffCanvas.width, diffCanvas.height);
    if (cfState.diffHeatmapImg) {
      ctx.drawImage(cfState.diffHeatmapImg, 0, 0, diffCanvas.width, diffCanvas.height);
    }
  }

  // Canvas ROI Selection Interactivity
  if (origCanvas) {
    origCanvas.addEventListener('mousedown', (e) => {
      const rect = origCanvas.getBoundingClientRect();
      const scaleX = origCanvas.width / rect.width;
      const scaleY = origCanvas.height / rect.height;
      cfState.isDrawingRoi = true;
      cfState.drawStartX = (e.clientX - rect.left) * scaleX;
      cfState.drawStartY = (e.clientY - rect.top) * scaleY;
    });

    origCanvas.addEventListener('mousemove', (e) => {
      if (!cfState.isDrawingRoi) return;
      const rect = origCanvas.getBoundingClientRect();
      const scaleX = origCanvas.width / rect.width;
      const scaleY = origCanvas.height / rect.height;
      const currX = (e.clientX - rect.left) * scaleX;
      const currY = (e.clientY - rect.top) * scaleY;

      const x = Math.min(cfState.drawStartX, currX);
      const y = Math.min(cfState.drawStartY, currY);
      const w = Math.max(5, Math.abs(currX - cfState.drawStartX));
      const h = Math.max(5, Math.abs(currY - cfState.drawStartY));

      cfState.roi = { x: Math.round(x), y: Math.round(y), width: Math.round(w), height: Math.round(h) };
      renderPhase20OriginalCanvas();
    });

    window.addEventListener('mouseup', () => {
      if (cfState.isDrawingRoi) {
        cfState.isDrawingRoi = false;
      }
    });
  }

  // Clear ROI Button
  const clearRoiBtn = document.getElementById('cfClearRoiBtn');
  if (clearRoiBtn) {
    clearRoiBtn.onclick = () => {
      cfState.roi = { x: 50, y: 50, width: 80, height: 80 };
      renderPhase20OriginalCanvas();
    };
  }

  // Random Control ROI Button
  const randomCtrlBtn = document.getElementById('cfRandomControlBtn');
  if (randomCtrlBtn) {
    randomCtrlBtn.onclick = () => {
      const w = cfState.roi ? cfState.roi.width : 80;
      const h = cfState.roi ? cfState.roi.height : 80;
      const maxX = (origCanvas ? origCanvas.width : 400) - w;
      const maxY = (origCanvas ? origCanvas.height : 400) - h;
      const rx = Math.floor(Math.random() * Math.max(1, maxX));
      const ry = Math.floor(Math.random() * Math.max(1, maxY));
      cfState.roi = { x: rx, y: ry, width: w, height: h };
      renderPhase20OriginalCanvas();
    };
  }

  // Opacity & Toggle Listeners
  ['cfToggleBaseHeatmap', 'cfBaseOpacity'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.oninput = el.onchange = () => renderPhase20OriginalCanvas();
  });
  ['cfToggleCfHeatmap', 'cfCfOpacity'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.oninput = el.onchange = () => renderPhase20PerturbedCanvas();
  });

  // Study Selector change
  const cfStudySel = document.getElementById('cfStudySelect');
  if (cfStudySel) {
    cfStudySel.onchange = (e) => {
      cfState.studyId = e.target.value;
      loadPhase20SourceImage();
    };
  }

  // Run Counterfactual Pipeline
  const runCfBtn = document.getElementById('cfRunBtn');
  if (runCfBtn) {
    runCfBtn.onclick = async () => {
      runCfBtn.disabled = true;
      runCfBtn.textContent = '⏳ Running Pipeline...';

      try {
        const method = document.getElementById('cfMethodSelect')?.value || 'REGION_MASK';
        const strength = parseFloat(document.getElementById('cfStrengthSlider')?.value || '1.0');
        const seed = parseInt(document.getElementById('cfSeedInput')?.value || '42', 10);
        const targetP = document.getElementById('cfTargetPathology')?.value || 'Atelectasis';

        // 1. Create Counterfactual
        const createResp = await fetch('/api/phase20/counterfactuals', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ study_id: cfState.studyId, source_view: cfState.sourceView })
        });
        const createData = await createResp.json();
        if (!createResp.ok) throw new Error(createData.error || `HTTP ${createResp.status}`);
        const cfId = createData.counterfactual_id;

        // 2. Configure Perturbation with Normalized ROI
        const canvasW = origCanvas ? origCanvas.width : 400;
        const canvasH = origCanvas ? origCanvas.height : 400;
        const normRoi = {
          x: cfState.roi.x,
          y: cfState.roi.y,
          width: cfState.roi.width,
          height: cfState.roi.height,
          norm_x: cfState.roi.x / canvasW,
          norm_y: cfState.roi.y / canvasH,
          norm_width: cfState.roi.width / canvasW,
          norm_height: cfState.roi.height / canvasH
        };

        const confResp = await fetch(`/api/phase20/counterfactuals/${cfId}/configure`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            study_id: cfState.studyId,
            method: method,
            roi: normRoi,
            strength: strength,
            seed: seed,
            target_pathology: targetP
          })
        });
        const confData = await confResp.json();
        if (!confResp.ok) throw new Error(confData.error || `HTTP ${confResp.status}`);

        // 3. Run Pipeline
        const runResp = await fetch(`/api/phase20/counterfactuals/${cfId}/run`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ study_id: cfState.studyId })
        });
        const runData = await runResp.json();
        if (!runResp.ok) throw new Error(runData.error || `HTTP ${runResp.status}`);

        const exp = runData.counterfactual;
        cfState.experiment = exp;
        renderPhase20ExperimentResults(exp);
        loadPhase20Dashboard();

      } catch (err) {
        alert(`Counterfactual run error: ${err.message}`);
      } finally {
        runCfBtn.disabled = false;
        runCfBtn.textContent = '▶️ Run Counterfactual';
      }
    };
  }

  function renderPhase20ExperimentResults(exp) {
    // Badges
    const lifeBadge = document.getElementById('cfLifecycleBadge');
    if (lifeBadge) {
      lifeBadge.textContent = exp.status;
      lifeBadge.style.background = exp.status === 'PUBLISHED' ? '#a855f7' : (exp.status === 'COMPLETED' ? '#38bdf8' : '#10b981');
    }
    const reproBadge = document.getElementById('cfReproBadge');
    if (reproBadge && exp.reproducibility) {
      reproBadge.textContent = `REPRODUCIBILITY: ${exp.reproducibility.status}`;
      reproBadge.style.color = exp.reproducibility.status === 'REPRODUCIBLE' ? '#10b981' : '#f59e0b';
    }

    // Load Heatmaps & Perturbed Image
    const cfId = exp.counterfactual_id;
    const studyId = exp.study_id;

    const baseHm = new Image();
    baseHm.crossOrigin = 'anonymous';
    baseHm.src = `/api/counterfactual-artifacts/${studyId}/artifacts/${cfId}_baseline_gradcam.png`;
    baseHm.onload = () => { cfState.baseHeatmapImg = baseHm; renderPhase20OriginalCanvas(); };

    const pertImg = new Image();
    pertImg.crossOrigin = 'anonymous';
    pertImg.src = `/api/counterfactual-artifacts/${studyId}/artifacts/${cfId}_counterfactual_image.png`;
    pertImg.onload = () => { cfState.perturbedImg = pertImg; renderPhase20PerturbedCanvas(); };

    const cfHm = new Image();
    cfHm.crossOrigin = 'anonymous';
    cfHm.src = `/api/counterfactual-artifacts/${studyId}/artifacts/${cfId}_counterfactual_gradcam.png`;
    cfHm.onload = () => { cfState.cfHeatmapImg = cfHm; renderPhase20PerturbedCanvas(); };

    const diffHm = new Image();
    diffHm.crossOrigin = 'anonymous';
    diffHm.src = `/api/counterfactual-artifacts/${studyId}/artifacts/${cfId}_attribution_diff.png`;
    diffHm.onload = () => { cfState.diffHeatmapImg = diffHm; renderPhase20DiffCanvas(); };

    // Finding Deltas Table
    const tbody = document.getElementById('cfFindingDeltasTbody');
    if (tbody && exp.finding_deltas) {
      tbody.innerHTML = '';
      exp.finding_deltas.forEach(fd => {
        const tr = document.createElement('tr');
        const dirColor = fd.direction === 'INCREASED' ? '#10b981' : (fd.direction === 'DECREASED' ? '#38bdf8' : '#64748b');
        const sign = fd.delta_abs >= 0 ? '+' : '';
        tr.innerHTML = `
          <td><strong>${fd.pathology}</strong></td>
          <td><code>${fd.baseline_score.toFixed(4)}</code></td>
          <td><code>${fd.counterfactual_score.toFixed(4)}</code></td>
          <td style="color: ${dirColor}; font-weight: 700;"><code>${sign}${fd.delta_abs.toFixed(4)}</code></td>
          <td style="color: ${dirColor};"><code>${sign}${(fd.delta_rel * 100).toFixed(1)}%</code></td>
          <td><span class="badge" style="background: ${dirColor}; color: white; font-size: 0.7rem;">${fd.direction}</span></td>
        `;
        tbody.appendChild(tr);
      });
    }

    // Attribution Metrics
    if (exp.attribution_deltas && exp.attribution_deltas.length > 0) {
      const ad = exp.attribution_deltas[0];
      const setEl = (id, val) => { const el = document.getElementById(id); if (el) el.textContent = val; };
      setEl('cfAttrMeanDiff', ad.mean_abs_difference.toFixed(4));
      setEl('cfAttrOverlap', `${(ad.overlap_fraction * 100).toFixed(1)}%`);
      setEl('cfAttrRoiBase', `${(ad.roi_attribution_fraction_baseline * 100).toFixed(1)}%`);
      setEl('cfAttrRoiCf', `${(ad.roi_attribution_fraction_counterfactual * 100).toFixed(1)}%`);
      setEl('cfAttrShift', `${ad.attribution_shift >= 0 ? '+' : ''}${(ad.attribution_shift * 100).toFixed(1)}%`);
      const meanDiffBadge = document.getElementById('cfMeanDiffBadge');
      if (meanDiffBadge) meanDiffBadge.textContent = `Mean |Δ|: ${ad.mean_abs_difference.toFixed(4)}`;
    }

    // QA Impact
    const qaTbody = document.getElementById('cfQaImpactTbody');
    if (qaTbody && exp.qa_impact) {
      qaTbody.innerHTML = '';
      exp.qa_impact.slice(0, 8).forEach(qa => {
        const tr = document.createElement('tr');
        const changedBadge = qa.changed ? '<span class="badge" style="background: #f59e0b; color: white;">CHANGED</span>' : '<span class="badge" style="background: #334155; color: #94a3b8;">UNCHANGED</span>';
        tr.innerHTML = `
          <td><span style="font-size: 0.75rem; color: #94a3b8;">${qa.level}</span></td>
          <td><strong>${qa.finding}</strong></td>
          <td><code>${qa.baseline_answer}</code></td>
          <td><code>${qa.counterfactual_answer}</code></td>
          <td>${changedBadge} <span style="font-size: 0.7rem; color: #cbd5e1;">${qa.evidence_effect}</span></td>
        `;
        qaTbody.appendChild(tr);
      });
    }

    // Report Impact
    if (exp.report_impact) {
      const ri = exp.report_impact;
      const baseEl = document.getElementById('cfBaseReportSummary');
      if (baseEl) baseEl.textContent = ri.baseline_summary || 'No focal abnormality.';
      const cfEl = document.getElementById('cfPerturbedReportSummary');
      if (cfEl) cfEl.textContent = ri.counterfactual_summary || 'No focal abnormality.';
      const transEl = document.getElementById('cfStatusChangesSummary');
      if (transEl) {
        if (ri.status_changes && ri.status_changes.length > 0) {
          transEl.innerHTML = `Status transitions: ${ri.status_changes.map(s => `<strong>${s.finding}</strong> (${s.baseline_status} → ${s.counterfactual_status})`).join(', ')}`;
        } else {
          transEl.textContent = 'Status transitions: None observed.';
        }
      }
    }

    // Reproducibility & Audit Trail
    if (exp.reproducibility) {
      const setEl = (id, val) => { const el = document.getElementById(id); if (el) el.textContent = val; };
      setEl('cfFingerprintBadge', `FINGERPRINT: ${(exp.reproducibility.fingerprint || '-').substring(0, 16)}...`);
      setEl('cfSrcHash', (exp.reproducibility.source_image_hash || '-').substring(0, 16) + '...');
      setEl('cfConfigHash', (exp.reproducibility.config_hash || '-').substring(0, 16) + '...');
    }

    const auditList = document.getElementById('cfAuditTrailList');
    if (auditList && exp.audit_trail) {
      auditList.innerHTML = '';
      exp.audit_trail.forEach(at => {
        const li = document.createElement('li');
        li.style.marginBottom = '4px';
        li.innerHTML = `<code>[${at.action}]</code> ${at.details} <span style="color: #64748b;">(${at.timestamp ? at.timestamp.substring(11, 19) : ''})</span>`;
        auditList.appendChild(li);
      });
    }
  }

  // Validate Invariants Action
  const valCfBtn = document.getElementById('cfValidateBtn');
  if (valCfBtn) {
    valCfBtn.onclick = async () => {
      if (!cfState.experiment) return alert('Run a counterfactual experiment first.');
      try {
        const resp = await fetch(`/api/phase20/counterfactuals/${cfState.experiment.counterfactual_id}/validate`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ study_id: cfState.studyId })
        });
        const data = await resp.json();
        if (!resp.ok) throw new Error(data.error || `HTTP ${resp.status}`);
        cfState.experiment = data.counterfactual;
        renderPhase20ExperimentResults(data.counterfactual);
        alert('Counterfactual experiment successfully validated and invariants verified!');
      } catch (err) {
        alert(`Validation error: ${err.message}`);
      }
    };
  }

  // Publish & Lock Action
  const pubCfBtn = document.getElementById('cfPublishBtn');
  if (pubCfBtn) {
    pubCfBtn.onclick = async () => {
      if (!cfState.experiment) return alert('Run a counterfactual experiment first.');
      if (!confirm('Publishing will permanently lock this counterfactual experiment as immutable. Proceed?')) return;
      try {
        const resp = await fetch(`/api/phase20/counterfactuals/${cfState.experiment.counterfactual_id}/publish`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ study_id: cfState.studyId })
        });
        const data = await resp.json();
        if (!resp.ok) throw new Error(data.error || `HTTP ${resp.status}`);
        cfState.experiment = data.counterfactual;
        renderPhase20ExperimentResults(data.counterfactual);
        alert('Counterfactual experiment published and permanently locked!');
      } catch (err) {
        alert(`Publish error: ${err.message}`);
      }
    };
  }

  // Control ROI Comparison
  const compCtrlBtn = document.getElementById('cfCompareControlBtn');
  if (compCtrlBtn) {
    compCtrlBtn.onclick = async () => {
      if (!cfState.experiment) return alert('Run a counterfactual experiment first.');
      try {
        const resp = await fetch(`/api/phase20/counterfactuals/${cfState.experiment.counterfactual_id}/control`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ study_id: cfState.studyId })
        });
        const data = await resp.json();
        if (!resp.ok) throw new Error(data.error || `HTTP ${resp.status}`);
        const comp = data.control_comparison;
        alert(`Control Comparison Result:\n\n${comp.neutral_summary}\n\nMean Absolute Difference: ${comp.statistics.mean}\nBootstrap 95% CI: [${comp.bootstrap_ci.ci_lower} - ${comp.bootstrap_ci.ci_upper}]`);
      } catch (err) {
        alert(`Control comparison error: ${err.message}`);
      }
    };
  }

  // Export JSON Action
  const expJsonBtn = document.getElementById('cfExportJsonBtn');
  if (expJsonBtn) {
    expJsonBtn.onclick = () => {
      if (!cfState.experiment) return alert('Run a counterfactual experiment first.');
      window.open(`/api/phase20/counterfactuals/${cfState.experiment.counterfactual_id}/export?study_id=${cfState.studyId}`, '_blank');
    };
  }

  // Initialize application and top-level tab navigation
  init();
});





