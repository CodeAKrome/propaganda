/**
 * Bias Detector - History Page Script
 * 
 * Displays and manages analysis history
 */

document.addEventListener('DOMContentLoaded', init);

let history = [];

/**
 * Initialize history page
 */
async function init() {
  await loadHistory();
  renderHistory();
  setupEventListeners();
}

/**
 * Load history from storage
 */
async function loadHistory() {
  try {
    const stored = await chrome.storage.local.get('biasHistory');
    history = stored.biasHistory || [];
    console.log('[BiasDetector] Loaded history:', history.length, 'entries');
    
    if (history.length > 0) {
      console.log('[BiasDetector] First entry:', history[0]);
      console.log('[BiasDetector] Last entry:', history[history.length - 1]);
    } else {
      console.log('[BiasDetector] No history entries found in storage');
      // Debug: check what's in storage
      const allStorage = await chrome.storage.local.get(null);
      console.log('[BiasDetector] All storage keys:', Object.keys(allStorage));
    }
  } catch (e) {
    console.error('[BiasDetector] Failed to load history:', e);
    history = [];
  }
}

/**
 * Render history list
 */
function renderHistory() {
  const list = document.getElementById('historyList');
  
  if (history.length === 0) {
    list.innerHTML = `
      <div class="history-empty">
        <div class="history-empty-icon">📊</div>
        <div class="history-empty-text">No analysis history yet.<br>Analyze some articles to see them here.</div>
      </div>
    `;
    return;
  }
  
  list.innerHTML = history.map((item, index) => {
    const dir = item.result?.dir || { L: 0, C: 0, R: 0 };
    const deg = item.result?.deg || { L: 0, M: 0, H: 0 };
    const reason = item.result?.reason || item.result?.reasoning || '';
    
    const dominantDegree = Object.entries(deg).reduce((a, b) => 
      b[1] > a[1] ? b : a, ['M', 0])[0];
    
    const degreeLabels = { L: 'Low', M: 'Medium', H: 'High' };
    const degreeClass = dominantDegree.toLowerCase();
    
    const timeAgo = formatTimeAgo(item.timestamp);
    
    return `
      <div class="history-item" data-index="${index}">
        <div class="history-item-header">
          <span class="history-item-title" title="${escapeHtml(item.title || item.source)}">${escapeHtml(item.title || item.source || 'Unknown')}</span>
          <span class="history-item-time">${timeAgo}</span>
        </div>
        <div class="history-item-url" title="${escapeHtml(item.url)}">${escapeHtml(item.url)}</div>
        <div class="history-item-result">
          <div class="history-bars">
            <div class="history-bar" data-dir="L">
              <span class="bar-label">L</span>
              <div class="bar-track"><div class="bar-fill" style="width: ${Math.round((dir.L || 0) * 100)}%"></div></div>
              <span class="bar-value">${Math.round((dir.L || 0) * 100)}%</span>
            </div>
            <div class="history-bar" data-dir="C">
              <span class="bar-label">C</span>
              <div class="bar-track"><div class="bar-fill" style="width: ${Math.round((dir.C || 0) * 100)}%"></div></div>
              <span class="bar-value">${Math.round((dir.C || 0) * 100)}%</span>
            </div>
            <div class="history-bar" data-dir="R">
              <span class="bar-label">R</span>
              <div class="bar-track"><div class="bar-fill" style="width: ${Math.round((dir.R || 0) * 100)}%"></div></div>
              <span class="bar-value">${Math.round((dir.R || 0) * 100)}%</span>
            </div>
          </div>
          <div class="history-degree">
            <div class="history-degree-label">Intensity</div>
            <div class="history-degree-value ${degreeClass}">${degreeLabels[dominantDegree]}</div>
          </div>
        </div>
        ${reason ? `<div class="history-item-reason">${escapeHtml(reason)}</div>` : ''}
        <div class="history-item-actions">
          <button class="history-item-btn" onclick="openUrl('${escapeHtml(item.url)}')">Open Page</button>
          <button class="history-item-btn" onclick="copyResult(${index})">Copy Result</button>
          <button class="history-item-btn" onclick="deleteEntry(${index})">Delete</button>
        </div>
      </div>
    `;
  }).join('');
}

/**
 * Set up event listeners
 */
function setupEventListeners() {
  document.getElementById('clearHistory').addEventListener('click', clearHistory);
  document.getElementById('exportHistory').addEventListener('click', exportHistory);
}

/**
 * Clear all history
 */
async function clearHistory() {
  if (!confirm('Are you sure you want to clear all history? This cannot be undone.')) {
    return;
  }
  
  try {
    await chrome.storage.local.set({ biasHistory: [], biasResults: {} });
    history = [];
    renderHistory();
    console.log('[BiasDetector] History cleared');
  } catch (e) {
    console.error('[BiasDetector] Failed to clear history:', e);
    alert('Failed to clear history');
  }
}

/**
 * Export history as JSON
 */
function exportHistory() {
  const data = {
    exported: new Date().toISOString(),
    entries: history
  };
  
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  
  const a = document.createElement('a');
  a.href = url;
  a.download = `bias-history-${new Date().toISOString().split('T')[0]}.json`;
  a.click();
  
  URL.revokeObjectURL(url);
}

/**
 * Open URL in new tab
 */
function openUrl(url) {
  if (url) {
    chrome.tabs.create({ url });
  }
}

/**
 * Copy result to clipboard
 */
async function copyResult(index) {
  const item = history[index];
  if (!item) return;
  
  const dir = item.result?.dir || { L: 0, C: 0, R: 0 };
  const deg = item.result?.deg || { L: 0, M: 0, H: 0 };
  const reason = item.result?.reason || item.result?.reasoning || '';
  
  const text = `Bias Analysis: ${item.title || item.source}
URL: ${item.url}
Date: ${new Date(item.timestamp).toLocaleString()}

Direction:
- Left: ${Math.round((dir.L || 0) * 100)}%
- Center: ${Math.round((dir.C || 0) * 100)}%
- Right: ${Math.round((dir.R || 0) * 100)}%

Intensity: ${deg.L > deg.M && deg.L > deg.H ? 'Low' : deg.H > deg.M ? 'High' : 'Medium'}

Reasoning: ${reason}`;
  
  try {
    await navigator.clipboard.writeText(text);
    alert('Result copied to clipboard');
  } catch (e) {
    console.error('Failed to copy:', e);
    alert('Failed to copy to clipboard');
  }
}

/**
 * Delete a single entry
 */
async function deleteEntry(index) {
  if (!confirm('Delete this entry?')) return;
  
  history.splice(index, 1);
  
  try {
    await chrome.storage.local.set({ biasHistory: history });
    renderHistory();
  } catch (e) {
    console.error('Failed to delete entry:', e);
    alert('Failed to delete entry');
  }
}

/**
 * Format timestamp as relative time
 */
function formatTimeAgo(timestamp) {
  const seconds = Math.floor((Date.now() - timestamp) / 1000);
  
  if (seconds < 60) return 'Just now';
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
  if (seconds < 604800) return `${Math.floor(seconds / 86400)}d ago`;
  
  return new Date(timestamp).toLocaleDateString();
}

/**
 * Escape HTML special characters
 */
function escapeHtml(str) {
  if (!str) return '';
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}
