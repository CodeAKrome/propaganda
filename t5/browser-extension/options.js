/**
 * Bias Detector - Options Page Script
 * 
 * Handles settings management and storage
 */

document.addEventListener('DOMContentLoaded', init);

let config = null;

/**
 * Initialize options page
 */
async function init() {
  // Load current config
  config = await sendMessage({ type: 'GET_CONFIG' });
  
  // Populate form
  populateForm();
  
  // Set up event listeners
  setupEventListeners();
  
  // Load cache stats
  loadCacheStats();
}

/**
 * Send message to background script
 */
function sendMessage(message) {
  return new Promise((resolve) => {
    chrome.runtime.sendMessage(message, (response) => {
      resolve(response);
    });
  });
}

/**
 * Populate form with current config
 */
function populateForm() {
  if (!config) return;
  
  // API settings
  document.getElementById('apiEndpoint').value = config.apiEndpoint || 'http://localhost:8000';
  document.getElementById('apiType').value = config.apiType || 'http';
  
  // Analysis settings
  document.getElementById('minTextLength').value = config.minTextLength || 100;
  document.getElementById('autoAnalyze').checked = config.autoAnalyze || false;
  document.getElementById('cacheEnabled').checked = config.cacheEnabled !== false;
  
  // YouTube settings
  document.getElementById('youtubeEnabled').checked = config.youtubeEnabled !== false;
  document.getElementById('youtubeBufferSeconds').value = config.youtubeBufferSeconds || 30;
  updateBufferValue(config.youtubeBufferSeconds || 30);
}

/**
 * Set up event listeners
 */
function setupEventListeners() {
  // Test connection button
  document.getElementById('testConnection').addEventListener('click', testConnection);
  
  // Buffer slider
  document.getElementById('youtubeBufferSeconds').addEventListener('input', (e) => {
    updateBufferValue(e.target.value);
  });
  
  // Save button
  document.getElementById('saveSettings').addEventListener('click', saveSettings);
  
  // Clear cache button
  document.getElementById('clearCache').addEventListener('click', clearCache);
  
  // Export/Import
  document.getElementById('exportSettings').addEventListener('click', exportSettings);
  document.getElementById('importSettings').addEventListener('click', importSettings);
  
  // Real-time validation
  document.getElementById('apiEndpoint').addEventListener('blur', validateEndpoint);
}

/**
 * Update buffer value display
 */
function updateBufferValue(value) {
  document.getElementById('bufferValue').textContent = `${value}s`;
}

/**
 * Test API connection
 */
async function testConnection() {
  const button = document.getElementById('testConnection');
  const statusEl = document.getElementById('connectionStatus');
  
  // Get current endpoint
  const endpoint = document.getElementById('apiEndpoint').value;
  const apiType = document.getElementById('apiType').value;
  
  // Update button state
  button.classList.remove('success', 'error');
  button.innerHTML = `
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" class="spinning">
      <circle cx="12" cy="12" r="10"/>
      <path d="M12 6v6l4 2"/>
    </svg>
    Testing...
  `;
  
  // Temporarily update config for test
  const testConfig = {
    apiEndpoint: endpoint,
    apiType: apiType
  };
  
  await sendMessage({ type: 'UPDATE_CONFIG', config: testConfig });
  
  // Check status
  const status = await sendMessage({ type: 'CHECK_API_STATUS' });
  
  if (status?.available) {
    button.classList.add('success');
    button.innerHTML = `
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <path d="M22 11.08V12a10 10 0 11-5.93-9.14"/>
        <path d="M22 4L12 14.01l-3-3"/>
      </svg>
      Connected
    `;
    statusEl.className = 'connection-status success';
    statusEl.textContent = `Connected to ${endpoint} (${status.device || 'unknown device'})`;
  } else {
    button.classList.add('error');
    button.innerHTML = `
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <circle cx="12" cy="12" r="10"/>
        <path d="M15 9l-6 6M9 9l6 6"/>
      </svg>
      Failed
    `;
    statusEl.className = 'connection-status error';
    statusEl.textContent = status?.error || 'Could not connect to server';
  }
  
  // Restore original config
  await sendMessage({ type: 'UPDATE_CONFIG', config: config });
}

/**
 * Validate endpoint format
 */
function validateEndpoint(e) {
  const input = e.target;
  const value = input.value.trim();
  
  // Basic URL validation
  try {
    if (value && !value.startsWith('http://') && !value.startsWith('https://')) {
      input.value = 'http://' + value;
    }
    new URL(input.value);
    input.setCustomValidity('');
  } catch {
    input.setCustomValidity('Please enter a valid URL');
  }
}

/**
 * Save settings
 */
async function saveSettings() {
  const button = document.getElementById('saveSettings');
  const statusEl = document.getElementById('saveStatus');
  
  // Collect form data
  const newConfig = {
    apiEndpoint: document.getElementById('apiEndpoint').value.trim(),
    apiType: document.getElementById('apiType').value,
    minTextLength: parseInt(document.getElementById('minTextLength').value, 10) || 100,
    autoAnalyze: document.getElementById('autoAnalyze').checked,
    cacheEnabled: document.getElementById('cacheEnabled').checked,
    youtubeEnabled: document.getElementById('youtubeEnabled').checked,
    youtubeBufferSeconds: parseInt(document.getElementById('youtubeBufferSeconds').value, 10) || 30,
    cacheTTL: 3600000 // 1 hour
  };
  
  // Update button
  button.disabled = true;
  button.innerHTML = `
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" class="spinning">
      <circle cx="12" cy="12" r="10"/>
      <path d="M12 6v6l4 2"/>
    </svg>
    Saving...
  `;
  
  // Save to storage
  const response = await sendMessage({ 
    type: 'UPDATE_CONFIG', 
    config: newConfig 
  });
  
  if (response?.success) {
    config = newConfig;
    statusEl.className = 'save-status success';
    statusEl.textContent = 'Settings saved successfully';
    
    setTimeout(() => {
      statusEl.textContent = '';
    }, 3000);
  } else {
    statusEl.className = 'save-status error';
    statusEl.textContent = 'Failed to save settings';
  }
  
  button.disabled = false;
  button.innerHTML = `
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
      <path d="M19 21H5a2 2 0 01-2-2V5a2 2 0 012-2h11l5 5v11a2 2 0 01-2 2z"/>
      <path d="M17 21v-8H7v8M7 3v5h8"/>
    </svg>
    Save Settings
  `;
}

/**
 * Load cache statistics
 */
async function loadCacheStats() {
  const stats = await sendMessage({ type: 'GET_CACHE_STATS' });
  
  if (stats) {
    document.getElementById('cacheSize').textContent = stats.size || 0;
  }
}

/**
 * Clear cache
 */
async function clearCache() {
  const button = document.getElementById('clearCache');
  
  button.disabled = true;
  button.textContent = 'Clearing...';
  
  await sendMessage({ type: 'CLEAR_CACHE' });
  
  // Update stats
  loadCacheStats();
  
  button.disabled = false;
  button.innerHTML = `
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
      <path d="M3 6h18M19 6v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6m3 0V4a2 2 0 012-2h4a2 2 0 012 2v2"/>
    </svg>
    Clear Cache
  `;
}

/**
 * Export settings to JSON file
 */
function exportSettings(e) {
  e.preventDefault();
  
  const data = JSON.stringify(config, null, 2);
  const blob = new Blob([data], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  
  const a = document.createElement('a');
  a.href = url;
  a.download = 'bias-detector-settings.json';
  a.click();
  
  URL.revokeObjectURL(url);
}

/**
 * Import settings from JSON file
 */
function importSettings(e) {
  e.preventDefault();
  
  const input = document.createElement('input');
  input.type = 'file';
  input.accept = '.json';
  
  input.onchange = async (event) => {
    const file = event.target.files[0];
    if (!file) return;
    
    try {
      const text = await file.text();
      const imported = JSON.parse(text);
      
      // Validate imported config
      if (typeof imported === 'object') {
        config = { ...config, ...imported };
        populateForm();
        
        const statusEl = document.getElementById('saveStatus');
        statusEl.className = 'save-status success';
        statusEl.textContent = 'Settings imported. Click Save to apply.';
      }
    } catch (error) {
      console.error('Import failed:', error);
      const statusEl = document.getElementById('saveStatus');
      statusEl.className = 'save-status error';
      statusEl.textContent = 'Failed to import settings. Invalid file format.';
    }
  };
  
  input.click();
}

// Add spinning animation style
const style = document.createElement('style');
style.textContent = `
  .spinning {
    animation: spin 1s linear infinite;
  }
  @keyframes spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
  }
`;
document.head.appendChild(style);