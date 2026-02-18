/**
 * Bias Detector - Popup Script
 * 
 * Coordinates between popup UI, content script, and background service.
 * Handles result display in both popup and page overlay.
 */

document.addEventListener('DOMContentLoaded', init);

let config = null;

/**
   * Initialize popup
   */
  async function init() {
    console.log('[BiasDetector] Popup initializing...');
    
    // Load config
    try {
      const response = await sendMessage({ type: 'GET_CONFIG' });
      config = response?.config || response || {};
      console.log('[BiasDetector] Config loaded:', config);
    } catch (e) {
      console.error('[BiasDetector] Failed to load config:', e);
      config = {};
    }
    
    // Update UI with config
    updateUI();
    
    // Check API status
    checkAPIStatus();
    
    // Set up event listeners
    setupEventListeners();
    
    console.log('[BiasDetector] Popup ready');
  }

/**
 * Send message to background script with timeout
 */
function sendMessage(message, timeout = 30000) {
  return new Promise((resolve, reject) => {
    const timeoutId = setTimeout(() => {
      reject(new Error('Message timeout'));
    }, timeout);
    
    chrome.runtime.sendMessage(message, (response) => {
      clearTimeout(timeoutId);
      if (chrome.runtime.lastError) {
        reject(new Error(chrome.runtime.lastError.message));
      } else {
        resolve(response);
      }
    });
  });
}

/**
 * Update UI with current config
 */
function updateUI() {
  const autoAnalyzeToggle = document.getElementById('autoAnalyze');
  const youtubeToggle = document.getElementById('youtubeEnabled');
  
  if (config) {
    autoAnalyzeToggle.checked = config.autoAnalyze || false;
    youtubeToggle.checked = config.youtubeEnabled !== false;
  }
}

/**
 * Check API connection status
 */
async function checkAPIStatus() {
  const statusIndicator = document.getElementById('statusIndicator');
  const statusText = statusIndicator.querySelector('.status-text');
  
  try {
    const status = await sendMessage({ type: 'CHECK_API_STATUS' }, 5000);
    
    if (status?.available) {
      statusIndicator.classList.remove('offline');
      statusIndicator.classList.add('online');
      statusText.textContent = 'Connected';
    } else {
      statusIndicator.classList.remove('online');
      statusIndicator.classList.add('offline');
      statusText.textContent = 'Offline';
    }
  } catch (e) {
    statusIndicator.classList.remove('online');
    statusIndicator.classList.add('offline');
    statusText.textContent = 'Error';
  }
}

/**
 * Set up event listeners
 */
function setupEventListeners() {
  // Analyze page button
  const analyzeButton = document.getElementById('analyzePage');
  analyzeButton.addEventListener('click', handleAnalyzePage);
  
  // Auto-analyze toggle
  const autoAnalyzeToggle = document.getElementById('autoAnalyze');
  autoAnalyzeToggle.addEventListener('change', (e) => {
    updateConfig({ autoAnalyze: e.target.checked });
  });
  
  // YouTube toggle
  const youtubeToggle = document.getElementById('youtubeEnabled');
  youtubeToggle.addEventListener('change', (e) => {
    updateConfig({ youtubeEnabled: e.target.checked });
  });
  
  // Open options
  const optionsButton = document.getElementById('openOptions');
  optionsButton.addEventListener('click', () => {
    chrome.runtime.openOptionsPage();
  });
  
  // Open history
  const historyButton = document.getElementById('openHistory');
  historyButton.addEventListener('click', () => {
    chrome.tabs.create({ url: chrome.runtime.getURL('history.html') });
  });
}

/**
 * Handle analyze page button click
 */
async function handleAnalyzePage() {
  const button = document.getElementById('analyzePage');
  const quickResult = document.getElementById('quickResult');
  const statusText = document.querySelector('#statusIndicator .status-text');
  
  // Show loading state
  button.classList.add('loading');
  button.disabled = true;
  button.querySelector('span').textContent = 'Analyzing...';
  statusText.textContent = 'Analyzing...';
  
  // Hide previous result
  quickResult.style.display = 'none';
  
  try {
    // Get active tab
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    console.log('[BiasDetector] Analyzing tab:', tab.id, tab.url);
    
    // Check if we're on a valid page
    if (!tab.url || tab.url.startsWith('chrome://') || tab.url.startsWith('chrome-extension://')) {
      throw new Error('Cannot analyze Chrome internal pages');
    }
    
    // Extract text using scripting API
    console.log('[BiasDetector] Extracting text...');
    const extractResult = await chrome.scripting.executeScript({
      target: { tabId: tab.id },
      func: extractPageText
    });
    
    const text = extractResult?.[0]?.result;
    console.log('[BiasDetector] Extracted text length:', text?.length);
    
    if (!text || text.length < 100) {
      throw new Error(`Not enough text to analyze (found ${text?.length || 0} chars, need 100+)`);
    }
    
    // Send to background for analysis
    console.log('[BiasDetector] Sending to API...');
    const analysisResult = await sendMessage({
      type: 'ANALYZE_TEXT',
      text: text
    });
    
    console.log('[BiasDetector] Analysis result:', analysisResult);
    
    if (analysisResult?.success && analysisResult.result) {
      const result = analysisResult.result;
      
      // Display in popup
      displayQuickResult(result);
      
      // Also send to content script for page overlay with the analyzed text
      try {
        await chrome.tabs.sendMessage(tab.id, {
          type: 'SHOW_RESULT',
          result: result,
          source: tab.title,
          analyzedText: text
        });
      } catch (e) {
        console.log('[BiasDetector] Could not send to content script:', e);
      }
      
      statusText.textContent = 'Analysis Complete';
      
    } else if (analysisResult?.result?.error) {
      throw new Error(analysisResult.result.error);
    } else if (analysisResult?.error) {
      throw new Error(analysisResult.error);
    } else {
      throw new Error('Unknown error: ' + JSON.stringify(analysisResult));
    }
    
  } catch (error) {
    console.error('[BiasDetector] Analysis failed:', error);
    statusText.textContent = 'Error';
    
    // Show error in popup
    showError(error.message);
    
  } finally {
    button.classList.remove('loading');
    button.disabled = false;
    button.querySelector('span').textContent = 'Analyze This Page';
  }
}

/**
 * Function to extract text from page (injected into page context)
 */
function extractPageText() {
  // Try common article selectors
  const selectors = [
    'article',
    '[role="article"]',
    '.article-content',
    '.post-content',
    '.entry-content',
    '.article-body',
    '.story-body',
    'main article',
    'main .content',
    '#article-body',
    '.news-article',
    'main',
    '.main',
    'body'
  ];
  
  let bestElement = null;
  let bestLength = 0;
  
  for (const selector of selectors) {
    const elements = document.querySelectorAll(selector);
    for (const el of elements) {
      const text = el.innerText || el.textContent || '';
      const cleanLength = text.trim().replace(/\s+/g, ' ').length;
      if (cleanLength > bestLength) {
        bestElement = el;
        bestLength = cleanLength;
      }
    }
  }
  
  if (!bestElement) {
    bestElement = document.body;
  }
  
  // Clone and clean
  const clone = bestElement.cloneNode(true);
  
  // Remove unwanted elements
  const removeSelectors = [
    'script', 'style', 'nav', 'header', 'footer', 'aside',
    '.advertisement', '.ad', '.ads', '.social-share', '.comments',
    '.related-articles', '.sidebar', '.navigation', '.menu',
    '[role="navigation"]', '[role="complementary"]',
    'noscript', 'iframe', 'svg'
  ];
  
  removeSelectors.forEach(sel => {
    clone.querySelectorAll(sel).forEach(el => el.remove());
  });
  
  // Get text and clean
  let text = clone.innerText || clone.textContent || '';
  text = text.replace(/\s+/g, ' ').trim();
  
  // Return first 5000 chars
  return text.substring(0, 5000);
}

/**
 * Update config in storage
 */
async function updateConfig(updates) {
  config = { ...config, ...updates };
  await sendMessage({ 
    type: 'UPDATE_CONFIG', 
    config: updates 
  });
}

/**
 * Display quick result in popup
 */
function displayQuickResult(result) {
  const quickResult = document.getElementById('quickResult');
  
  if (!result) {
    quickResult.style.display = 'none';
    return;
  }
  
  // Handle error results
  if (result.error) {
    showError(result.error);
    return;
  }
  
  quickResult.style.display = 'block';
  
  // Update direction bars
  const dir = result.dir || { L: 0, C: 0, R: 0 };
  
  Object.entries(dir).forEach(([key, value]) => {
    const bar = quickResult.querySelector(`.quick-bar[data-dir="${key}"]`);
    if (bar) {
      const fill = bar.querySelector('.bar-fill');
      const valueEl = bar.querySelector('.bar-value');
      const percentage = Math.round((value || 0) * 100);
      fill.style.width = `${percentage}%`;
      valueEl.textContent = `${percentage}%`;
    }
  });
  
  // Update degree
  const deg = result.deg || { L: 0, M: 0, H: 0 };
  const dominant = Object.entries(deg).reduce((a, b) => 
    b[1] > a[1] ? b : a, ['M', 0]);
  
  const degreeEl = document.getElementById('quickDegree');
  const degreeLabels = { L: 'Low', M: 'Medium', H: 'High' };
  degreeEl.textContent = degreeLabels[dominant[0]] || '-';
  
  // Add reasoning if available
  if (result.reason || result.reasoning) {
    const existingReason = quickResult.querySelector('.quick-reason');
    if (existingReason) {
      existingReason.remove();
    }
    
    const reasonEl = document.createElement('div');
    reasonEl.className = 'quick-reason';
    reasonEl.style.cssText = 'margin-top: 12px; padding: 8px; background: rgba(0,0,0,0.1); border-radius: 4px; font-size: 11px; line-height: 1.4; max-height: 60px; overflow-y: auto;';
    reasonEl.textContent = result.reason || result.reasoning;
    quickResult.appendChild(reasonEl);
  }
}

/**
 * Show error message
 */
function showError(message) {
  const quickResult = document.getElementById('quickResult');
  quickResult.style.display = 'block';
  quickResult.innerHTML = `
    <div style="color: #ef4444; padding: 8px; text-align: center;">
      <strong>Error</strong><br>
      <span style="font-size: 12px;">${message}</span>
    </div>
  `;
}

// Listen for result updates from background
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === 'ANALYSIS_RESULT') {
    displayQuickResult(message.result);
  }
  sendResponse({ received: true });
  return false;
});
