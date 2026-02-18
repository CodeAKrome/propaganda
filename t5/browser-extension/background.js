/**
 * Bias Detector - Background Service Worker
 * 
 * Handles API communication, caching, and state management.
 * Supports both MCP server and direct HTTP API endpoints.
 */

// Configuration defaults
const DEFAULT_CONFIG = {
  apiEndpoint: 'http://localhost:8000',
  apiType: 'http', // 'http' or 'mcp'
  autoAnalyze: false,
  cacheEnabled: true,
  cacheTTL: 3600000, // 1 hour in ms
  minTextLength: 100,
  debounceMs: 500,
  youtubeBufferSeconds: 30
};

// State
let config = { ...DEFAULT_CONFIG };
let analysisCache = new Map();
let pendingAnalysis = null;

// Initialize
chrome.runtime.onInstalled.addListener(async () => {
  const stored = await chrome.storage.sync.get('config');
  config = { ...DEFAULT_CONFIG, ...stored.config };
  console.log('[BiasDetector] Initialized with config:', config);
});

// Also initialize on startup
chrome.runtime.onStartup.addListener(async () => {
  const stored = await chrome.storage.sync.get('config');
  config = { ...DEFAULT_CONFIG, ...stored.config };
  console.log('[BiasDetector] Startup with config:', config);
});

// Listen for config changes
chrome.storage.onChanged.addListener((changes, area) => {
  if (area === 'sync' && changes.config) {
    config = { ...config, ...changes.config.newValue };
    console.log('[BiasDetector] Config updated:', config);
  }
});

/**
 * Analyze text for political bias
 * @param {string} text - Text to analyze
 * @param {boolean} useCache - Whether to use cached results
 * @returns {Promise<Object>} Bias analysis result
 */
async function analyzeText(text, useCache = true) {
  if (!text || text.length < config.minTextLength) {
    return {
      error: 'Text too short for analysis',
      minRequired: config.minTextLength,
      provided: text?.length || 0
    };
  }

  // Check cache
  const cacheKey = hashText(text);
  if (useCache && config.cacheEnabled && analysisCache.has(cacheKey)) {
    const cached = analysisCache.get(cacheKey);
    if (Date.now() - cached.timestamp < config.cacheTTL) {
      console.log('[BiasDetector] Returning cached result');
      return { ...cached.result, cached: true };
    }
  }

  try {
    const result = await callBiasAPI(text);
    
    // Cache result
    if (config.cacheEnabled && result && !result.error) {
      analysisCache.set(cacheKey, {
        result,
        timestamp: Date.now()
      });
    }
    
    return result;
  } catch (error) {
    console.error('[BiasDetector] API error:', error);
    return {
      error: error.message,
      apiEndpoint: config.apiEndpoint
    };
  }
}

/**
 * Call the bias detection API
 * @param {string} text - Text to analyze
 * @returns {Promise<Object>} API response
 */
async function callBiasAPI(text) {
  const endpoint = `${config.apiEndpoint}/predict`;

  console.log('[BiasDetector] Calling API:', endpoint);
  
  const response = await fetch(endpoint, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ text })
  });

  if (!response.ok) {
    const errorText = await response.text();
    console.error('[BiasDetector] API error response:', errorText);
    throw new Error(`API returned ${response.status}: ${response.statusText}`);
  }

  const data = await response.json();
  console.log('[BiasDetector] API response:', data);
  
  // Parse the result - handle both parsed and raw_output formats
  let result = data.result || data;
  
  // If result has raw_output, try to parse it
  if (result.raw_output) {
    console.log('[BiasDetector] Parsing raw_output...');
    result = parseRawOutput(result.raw_output);
  }
  
  return {
    dir: result.dir || { L: 0, C: 0, R: 0 },
    deg: result.deg || { L: 0, M: 0, H: 0 },
    reason: result.reason || result.reasoning || 'Analysis complete',
    device: data.device || 'unknown'
  };
}

/**
 * Parse raw_output from model into structured format
 * @param {string} rawOutput - Raw model output string
 * @returns {Object} Parsed result with dir, deg, reason
 */
function parseRawOutput(rawOutput) {
  console.log('[BiasDetector] Parsing:', rawOutput);
  
  // Default values
  const result = {
    dir: { L: 0, C: 0, R: 0 },
    deg: { L: 0, M: 0, H: 0 },
    reason: ''
  };
  
  try {
    // Extract direction values - look for pattern after "dir"
    const lMatch = rawOutput.match(/"dir"[^d]*?"L"\s*:\s*([\d.]+)/);
    const cMatch = rawOutput.match(/"dir"[^d]*?"C"\s*:\s*([\d.]+)/);
    const rMatch = rawOutput.match(/"dir"[^d]*?"R"\s*:\s*([\d.]+)/);
    
    if (lMatch) result.dir.L = parseFloat(lMatch[1]);
    if (cMatch) result.dir.C = parseFloat(cMatch[1]);
    if (rMatch) result.dir.R = parseFloat(rMatch[1]);
    
    // Extract degree values - look for pattern after "deg"
    const dlMatch = rawOutput.match(/"deg"[^r]*?"L"\s*:\s*([\d.]+)/);
    const dmMatch = rawOutput.match(/"deg"[^r]*?"M"\s*:\s*([\d.]+)/);
    const dhMatch = rawOutput.match(/"deg"[^r]*?"H"\s*:\s*([\d.]+)/);
    
    if (dlMatch) result.deg.L = parseFloat(dlMatch[1]);
    if (dmMatch) result.deg.M = parseFloat(dmMatch[1]);
    if (dhMatch) result.deg.H = parseFloat(dhMatch[1]);
    
    // Extract reason
    const reasonMatch = rawOutput.match(/"reason"\s*:\s*"([^"]+)"/);
    if (reasonMatch) result.reason = reasonMatch[1];
    
  } catch (e) {
    console.error('[BiasDetector] Parse error:', e);
  }
  
  console.log('[BiasDetector] Parsed result:', result);
  return result;
}

/**
 * Simple hash function for cache keys
 */
function hashText(text) {
  let hash = 0;
  for (let i = 0; i < text.length; i++) {
    const char = text.charCodeAt(i);
    hash = ((hash << 5) - hash) + char;
    hash = hash & hash;
  }
  return hash.toString(16);
}

/**
 * Debounce helper
 */
function debounce(fn, delay) {
  let timeoutId;
  return (...args) => {
    clearTimeout(timeoutId);
    timeoutId = setTimeout(() => fn(...args), delay);
  };
}

// Message handler for content scripts
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  console.log('[BiasDetector] Received message:', message.type);

  switch (message.type) {
    case 'ANALYZE_TEXT':
      analyzeText(message.text, message.useCache ?? true)
        .then(result => sendResponse({ success: true, result }))
        .catch(error => sendResponse({ success: false, error: error.message }));
      return true; // Keep channel open for async response

    case 'GET_CONFIG':
      sendResponse({ config });
      return false;

    case 'UPDATE_CONFIG':
      config = { ...config, ...message.config };
      chrome.storage.sync.set({ config });
      sendResponse({ success: true, config });
      return false;

    case 'CLEAR_CACHE':
      analysisCache.clear();
      sendResponse({ success: true });
      return false;

    case 'GET_CACHE_STATS':
      sendResponse({
        size: analysisCache.size,
        keys: Array.from(analysisCache.keys())
      });
      return false;

    case 'CHECK_API_STATUS':
      checkAPIStatus()
        .then(status => sendResponse(status))
        .catch(error => sendResponse({ available: false, error: error.message }));
      return true;

    default:
      sendResponse({ error: 'Unknown message type' });
      return false;
  }
});

/**
 * Check if API server is available
 */
async function checkAPIStatus() {
  try {
    const endpoint = config.apiType === 'mcp'
      ? `${config.apiEndpoint}/model_info`
      : `${config.apiEndpoint}/health`;
    
    const response = await fetch(endpoint, { method: 'GET' });
    
    if (response.ok) {
      const data = await response.json();
      return {
        available: true,
        endpoint: config.apiEndpoint,
        type: config.apiType,
        ...data
      };
    }
    
    return {
      available: false,
      status: response.status
    };
  } catch (error) {
    return {
      available: false,
      error: error.message
    };
  }
}

// Context menu for manual analysis
chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: 'analyze-selection',
    title: 'Analyze bias',
    contexts: ['selection']
  });
});

chrome.contextMenus.onClicked.addListener((info, tab) => {
  if (info.menuItemId === 'analyze-selection' && info.selectionText) {
    analyzeText(info.selectionText)
      .then(result => {
        // Send result to content script for display
        chrome.tabs.sendMessage(tab.id, {
          type: 'SHOW_RESULT',
          result,
          selection: true
        });
      });
  }
});

console.log('[BiasDetector] Background service worker loaded');