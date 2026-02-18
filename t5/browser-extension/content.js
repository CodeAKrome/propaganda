/**
 * Bias Detector - Content Script
 * 
 * Captures YouTube captions in real-time and analyzes for political bias.
 * Design: Single unified caption capture system with rolling graph display.
 */

(function() {
  'use strict';

  // ========================================
  // STATE - Single source of truth
  // ========================================
  const state = {
    isYouTube: false,
    videoId: null,
    overlay: null,
    canvas: null,
    ctx: null,
    meterCanvas: null,
    meterCtx: null,
    
    // Caption capture
    captionBuffer: '',
    lastCaptionText: '',
    captionCheckInterval: null,
    lastCaptureTime: 0,
    
    // Analysis
    analysisInterval: null,
    lastAnalysisTime: 0,
    isAnalyzing: false,
    
    // Graph data
    biasHistory: [], // { time, dir: {L,C,R}, deg: {L,M,H} }
    maxHistoryPoints: 60,
    
    // Config
    config: {
      minTextLength: 100,
      analysisIntervalMs: 5000,
      captionCheckIntervalMs: 500,
      apiEndpoint: 'http://localhost:8000'
    }
  };

  // ========================================
  // INITIALIZATION
  // ========================================
  
  // Track if extension context is valid
  let extensionContextValid = true;
  let reloadNotificationShown = false;
  
  function checkExtensionContext() {
    try {
      // This will throw if extension context is invalidated
      chrome.runtime.id;
      return true;
    } catch (e) {
      console.log('[BiasDetector] Extension context invalidated');
      extensionContextValid = false;
      
      // Show notification once
      if (!reloadNotificationShown) {
        reloadNotificationShown = true;
        showReloadNotification();
      }
      
      return false;
    }
  }
  
  function init() {
    // Check if extension context is valid
    if (!checkExtensionContext()) {
      console.log('[BiasDetector] Extension context invalid, skipping init');
      return;
    }
    
    console.log('[BiasDetector] Initializing...');
    
    // Detect YouTube
    state.isYouTube = window.location.hostname.includes('youtube.com');
    
    if (state.isYouTube) {
      console.log('[BiasDetector] YouTube detected');
      waitForVideo().then(() => {
        console.log('[BiasDetector] Video found, starting caption capture');
        createOverlay();
        startCaptionCapture();
        startAnalysisLoop();
      }).catch(err => {
        console.log('[BiasDetector] Video not found:', err.message);
      });
    } else {
      // For non-YouTube pages, just listen for manual analysis
      console.log('[BiasDetector] Non-YouTube page, manual analysis only');
    }
    
    // Listen for messages from popup/background
    try {
      chrome.runtime.onMessage.addListener(handleMessage);
    } catch (e) {
      console.log('[BiasDetector] Could not add message listener:', e.message);
    }
  }

  function waitForVideo(timeout = 10000) {
    return new Promise((resolve, reject) => {
      const startTime = Date.now();
      
      function check() {
        const video = document.querySelector('video');
        if (video) {
          resolve(video);
          return;
        }
        
        if (Date.now() - startTime > timeout) {
          reject(new Error('Video not found'));
          return;
        }
        
        requestAnimationFrame(check);
      }
      
      check();
    });
  }

  // ========================================
  // CAPTION CAPTURE - Direct DOM polling
  // ========================================
  
  function startCaptionCapture() {
    console.log('[BiasDetector] Starting caption capture...');
    
    // Clear any existing interval
    if (state.captionCheckInterval) {
      clearInterval(state.captionCheckInterval);
    }
    
    // Poll for captions every 500ms
    state.captionCheckInterval = setInterval(() => {
      captureCaptions();
    }, state.config.captionCheckIntervalMs);
    
    // Also set up MutationObserver for immediate feedback
    setupCaptionObserver();
  }
  
  function captureCaptions() {
    // Check extension context first
    if (!extensionContextValid || !checkExtensionContext()) {
      console.log('[BiasDetector] Extension context invalid, stopping caption capture');
      stopCaptionCapture();
      return;
    }
    
    try {
      // Method 1: Query caption segments directly
      const segments = document.querySelectorAll('.ytp-caption-segment');
      
      if (segments.length > 0) {
        let text = '';
        segments.forEach(seg => {
          const t = seg.textContent?.trim();
          if (t) text += t + ' ';
        });
        
        text = text.trim();
        
        if (text && text !== state.lastCaptionText) {
          state.lastCaptionText = text;
          state.lastCaptureTime = Date.now();
          
          // Add to buffer
          state.captionBuffer += ' ' + text;
          
          console.log('[BiasDetector] Caption captured:', text.substring(0, 50) + '...');
          console.log('[BiasDetector] Buffer length:', state.captionBuffer.length);
          
          // Update the caption display
          updateCaptionDisplay();
        }
      }
    } catch (e) {
      console.log('[BiasDetector] Caption capture error:', e.message);
      // Check if extension context invalidated
      if (!checkExtensionContext()) {
        stopCaptionCapture();
      }
    }
  }
  
  function setupCaptionObserver() {
    const captionContainer = document.querySelector('.ytp-caption-window-container');
    
    if (captionContainer) {
      const observer = new MutationObserver((mutations) => {
        // Don't process here, just log for debugging
        const segments = document.querySelectorAll('.ytp-caption-segment');
        if (segments.length > 0) {
          console.log('[BiasDetector] MutationObserver: captions visible');
        }
      });
      
      observer.observe(captionContainer, {
        childList: true,
        subtree: true,
        characterData: true
      });
      
      console.log('[BiasDetector] MutationObserver set up on caption container');
    } else {
      console.log('[BiasDetector] Caption container not found yet, will retry...');
      // Retry after a delay
      setTimeout(setupCaptionObserver, 2000);
    }
  }
  
  function stopCaptionCapture() {
    if (state.captionCheckInterval) {
      clearInterval(state.captionCheckInterval);
      state.captionCheckInterval = null;
    }
  }

  // ========================================
  // ANALYSIS LOOP
  // ========================================
  
  function startAnalysisLoop() {
    console.log('[BiasDetector] Starting analysis loop...');
    
    if (state.analysisInterval) {
      clearInterval(state.analysisInterval);
    }
    
    state.analysisInterval = setInterval(() => {
      analyzeBuffer();
    }, state.config.analysisIntervalMs);
  }
  
  async function analyzeBuffer() {
    // Check extension context first
    if (!extensionContextValid || !checkExtensionContext()) {
      console.log('[BiasDetector] Extension context invalid, skipping analysis');
      return;
    }
    
    // Skip if already analyzing or buffer too small
    if (state.isAnalyzing) {
      console.log('[BiasDetector] Analysis already in progress, skipping');
      return;
    }
    
    const bufferLength = state.captionBuffer.trim().length;
    
    if (bufferLength < state.config.minTextLength) {
      console.log('[BiasDetector] Buffer too small:', bufferLength, '/', state.config.minTextLength);
      return;
    }
    
    state.isAnalyzing = true;
    state.lastAnalysisTime = Date.now();
    
    const textToAnalyze = state.captionBuffer.trim();
    console.log('[BiasDetector] ANALYZING CAPTIONS:', bufferLength, 'chars');
    
    try {
      const result = await callBiasAPI(textToAnalyze);
      
      // Check context again after async operation
      if (!extensionContextValid || !checkExtensionContext()) {
        console.log('[BiasDetector] Extension context invalid after API call');
        state.isAnalyzing = false;
        return;
      }
      
      if (result && !result.error) {
        console.log('[BiasDetector] Analysis result:', result);
        
        // Add to history
        state.biasHistory.push({
          time: Date.now(),
          dir: result.dir || { L: 0, C: 0, R: 0 },
          deg: result.deg || { L: 0, M: 0, H: 0 },
          reason: result.reason || ''
        });
        
        // Trim history
        if (state.biasHistory.length > state.maxHistoryPoints) {
          state.biasHistory.shift();
        }
        
        // Update UI
        updateOverlay(result);
        drawGraph();
        drawBiasMeter();
        
        // Clear buffer after successful analysis
        state.captionBuffer = '';
        updateCaptionDisplay(); // Update display to show cleared state
        console.log('[BiasDetector] Buffer cleared after analysis');
      } else {
        console.log('[BiasDetector] Analysis error:', result?.error);
        // Show error in overlay
        updateOverlayWithError(result?.error || 'Unknown error');
      }
    } catch (e) {
      console.error('[BiasDetector] Analysis failed:', e);
      updateOverlayWithError(e.message);
    }
    
    state.isAnalyzing = false;
  }
  
  async function callBiasAPI(text) {
    console.log('[BiasDetector] Calling API with', text.length, 'chars');
    
    try {
      const response = await fetch(`${state.config.apiEndpoint}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text })
      });
      
      console.log('[BiasDetector] API response status:', response.status);
      
      if (!response.ok) {
        const errorText = await response.text();
        console.error('[BiasDetector] API error response:', errorText);
        return { error: `API returned ${response.status}: ${errorText.substring(0, 100)}` };
      }
      
      const data = await response.json();
      console.log('[BiasDetector] API response data:', data);
      
      // Parse result
      let result = data.result || data;
      
      if (result.raw_output) {
        result = parseRawOutput(result.raw_output);
      }
      
      return {
        dir: result.dir || { L: 0, C: 0, R: 0 },
        deg: result.deg || { L: 0, M: 0, H: 0 },
        reason: result.reason || result.reasoning || 'Analysis complete',
        device: data.device || 'unknown'
      };
    } catch (e) {
      console.error('[BiasDetector] API call failed:', e);
      return { error: e.message };
    }
  }
  
  function updateOverlayWithError(errorMsg) {
    const reasonEl = document.getElementById('bias-reason');
    if (reasonEl) {
      reasonEl.textContent = `Error: ${errorMsg}`;
      reasonEl.style.color = '#f87171';
    }
  }
  
  function updateCaptionDisplay() {
    const captionText = document.getElementById('bias-caption-text');
    const captionCount = document.getElementById('bias-caption-count');
    const captionScroll = document.getElementById('bias-caption-scroll');
    
    if (captionText) {
      const text = state.captionBuffer.trim();
      if (text) {
        captionText.textContent = text;
        captionText.style.color = 'rgba(255, 255, 255, 0.8)';
      } else {
        captionText.textContent = 'Waiting for captions...';
        captionText.style.color = 'rgba(255, 255, 255, 0.4)';
      }
    }
    
    if (captionCount) {
      captionCount.textContent = `${state.captionBuffer.trim().length} chars`;
    }
    
    // Auto-scroll to bottom
    if (captionScroll) {
      captionScroll.scrollTop = captionScroll.scrollHeight;
    }
  }
  
  function showReloadNotification() {
    // Create a notification to tell user to reload
    const notification = document.createElement('div');
    notification.id = 'bias-detector-reload-notification';
    notification.style.cssText = `
      position: fixed;
      top: 20px;
      right: 20px;
      z-index: 2147483647;
      background: rgba(239, 68, 68, 0.95);
      color: white;
      padding: 12px 16px;
      border-radius: 8px;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
      font-size: 13px;
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
      display: flex;
      align-items: center;
      gap: 10px;
    `;
    notification.innerHTML = `
      <span>⚠️ Extension reloaded. Please refresh the page.</span>
      <button style="
        background: rgba(255,255,255,0.2);
        border: none;
        color: white;
        padding: 4px 8px;
        border-radius: 4px;
        cursor: pointer;
        font-size: 12px;
      " onclick="location.reload()">Refresh</button>
    `;
    document.body.appendChild(notification);
  }
  
  function parseRawOutput(rawOutput) {
    const result = {
      dir: { L: 0, C: 0, R: 0 },
      deg: { L: 0, M: 0, H: 0 },
      reason: ''
    };
    
    console.log('[BiasDetector] Parsing raw_output:', rawOutput);
    
    try {
      // The raw_output format is malformed JSON like:
      // "dir":"L":0.3,"C":0.6,"R":0.1,"deg":"L":0.2,"M":0.3,"H":0.5,"reason":"..."
      // Note: M and H don't have quotes around them
      
      // Find all key:value pairs (with or without quotes on key)
      // Pattern: "L":0.3 or L:0.3 or "M":0.4 or M:0.4
      const allPairs = [];
      
      // Match quoted keys like "L":0.3
      const quotedRegex = /"(L|C|R|M|H)"\s*:\s*([\d.]+)/g;
      let match;
      while ((match = quotedRegex.exec(rawOutput)) !== null) {
        allPairs.push({ key: match[1], value: parseFloat(match[2]) });
      }
      
      // Match unquoted keys like M:0.4 (but not inside strings)
      const unquotedRegex = /[,}]?\s*(M|H)\s*:\s*([\d.]+)/g;
      while ((match = unquotedRegex.exec(rawOutput)) !== null) {
        // Check if this M or H is already captured
        const alreadyCaptured = allPairs.some(p => p.key === match[1]);
        if (!alreadyCaptured) {
          allPairs.push({ key: match[1], value: parseFloat(match[2]) });
        }
      }
      
      console.log('[BiasDetector] All pairs found:', allPairs);
      
      // Separate direction and degree values
      // Direction comes first (L, C, R), then degree (L, M, H)
      const lcrValues = allPairs.filter(p => ['L', 'C', 'R'].includes(p.key));
      const mhValues = allPairs.filter(p => ['M', 'H'].includes(p.key));
      
      // First 3 L/C/R values are direction
      const dirValues = lcrValues.slice(0, 3);
      // Remaining L value (if any) and M/H are degree
      const degL = lcrValues.length > 3 ? lcrValues[3] : null;
      
      // Assign direction values
      dirValues.forEach(v => {
        result.dir[v.key] = v.value;
      });
      
      // Assign degree values
      if (degL) result.deg.L = degL.value;
      mhValues.forEach(v => {
        result.deg[v.key] = v.value;
      });
      
      // Extract reason
      const reasonMatch = rawOutput.match(/"reason"\s*:\s*"([^"]+)"/);
      if (reasonMatch) result.reason = reasonMatch[1];
      
      console.log('[BiasDetector] Parsed result:', result);
    } catch (e) {
      console.error('[BiasDetector] Parse error:', e);
    }
    
    return result;
  }

  // ========================================
  // OVERLAY UI
  // ========================================
  
  function createOverlay() {
    // Remove existing overlay
    const existing = document.getElementById('bias-detector-overlay');
    if (existing) existing.remove();
    
    // Create overlay
    const overlay = document.createElement('div');
    overlay.id = 'bias-detector-overlay';
    overlay.innerHTML = `
      <div class="bias-detector-inner">
        <button class="bias-detector-close" title="Close">×</button>
        
        <div class="bias-detector-header">
          <div class="bias-detector-icon">B</div>
          <div class="bias-detector-title">Bias Detector</div>
        </div>
        
        <div class="bias-detector-content">
          <!-- Direction bars -->
          <div class="bias-detector-direction">
            <div class="bias-detector-label">Political Leaning</div>
            <div class="bias-detector-bars">
              <div class="bias-bar bias-bar-left">
                <span class="bias-bar-label">L</span>
                <div class="bias-bar-track">
                  <div class="bias-bar-fill" id="bias-bar-left"></div>
                </div>
                <span class="bias-bar-value" id="bias-val-left">0%</span>
              </div>
              <div class="bias-bar bias-bar-center">
                <span class="bias-bar-label">C</span>
                <div class="bias-bar-track">
                  <div class="bias-bar-fill" id="bias-bar-center"></div>
                </div>
                <span class="bias-bar-value" id="bias-val-center">0%</span>
              </div>
              <div class="bias-bar bias-bar-right">
                <span class="bias-bar-label">R</span>
                <div class="bias-bar-track">
                  <div class="bias-bar-fill" id="bias-bar-right"></div>
                </div>
                <span class="bias-bar-value" id="bias-val-right">0%</span>
              </div>
            </div>
          </div>
          
          <!-- Degree meter -->
          <div class="bias-detector-degree">
            <div class="bias-detector-label">Intensity</div>
            <div class="bias-detector-meter">
              <div class="bias-meter-segment bias-meter-low" id="bias-deg-low">Low</div>
              <div class="bias-meter-segment bias-meter-medium" id="bias-deg-medium">Med</div>
              <div class="bias-meter-segment bias-meter-high" id="bias-deg-high">High</div>
            </div>
          </div>
          
          <!-- Reasoning -->
          <div class="bias-detector-reason">
            <div class="bias-detector-label">Analysis</div>
            <div class="bias-detector-reason-text" id="bias-reason">
              Waiting for captions...
            </div>
          </div>
          
          <!-- Graph -->
          <div class="bias-graph-container">
            <div class="bias-graph-header">
              <span class="bias-graph-title">Bias Over Time</span>
              <span class="bias-graph-samples" id="bias-samples">0 samples</span>
            </div>
            <div class="bias-graph-wrapper">
              <canvas id="biasGraphCanvas" width="288" height="100"></canvas>
            </div>
            <div class="bias-graph-legend">
              <div class="bias-graph-legend-item">
                <div class="bias-graph-legend-line left"></div>
                <span>Left</span>
              </div>
              <div class="bias-graph-legend-item">
                <div class="bias-graph-legend-line center"></div>
                <span>Center</span>
              </div>
              <div class="bias-graph-legend-item">
                <div class="bias-graph-legend-line right"></div>
                <span>Right</span>
              </div>
              <div class="bias-graph-legend-item">
                <span style="font-size: 9px; color: rgba(255,255,255,0.3);">• Dot = current</span>
              </div>
            </div>
          </div>
          
          <!-- Current Bias Meter -->
          <div class="bias-meter-container">
            <div class="bias-meter-header">
              <span class="bias-meter-title">Current Bias</span>
            </div>
            <div class="bias-meter-wrapper">
              <canvas id="biasMeterCanvas" width="288" height="80"></canvas>
            </div>
          </div>
          
          <!-- Caption Text View -->
          <div class="bias-caption-container">
            <div class="bias-caption-header">
              <span class="bias-caption-title">Captured Text</span>
              <span class="bias-caption-count" id="bias-caption-count">0 chars</span>
            </div>
            <div class="bias-caption-scroll" id="bias-caption-scroll">
              <div class="bias-caption-text" id="bias-caption-text">
                Waiting for captions...
              </div>
            </div>
          </div>
        </div>
        
        <div class="bias-detector-footer">
          <span class="bias-detector-source" id="bias-source">YouTube Captions</span>
        </div>
      </div>
    `;
    
    document.body.appendChild(overlay);
    state.overlay = overlay;
    
    // Get canvases
    state.canvas = document.getElementById('biasGraphCanvas');
    state.ctx = state.canvas.getContext('2d');
    state.meterCanvas = document.getElementById('biasMeterCanvas');
    state.meterCtx = state.meterCanvas.getContext('2d');
    
    // Close button
    overlay.querySelector('.bias-detector-close').addEventListener('click', () => {
      overlay.classList.add('bias-detector-hidden');
    });
    
    console.log('[BiasDetector] Overlay created');
  }
  
  function updateOverlay(result) {
    if (!state.overlay) return;
    
    // Update direction bars
    const leftBar = document.getElementById('bias-bar-left');
    const centerBar = document.getElementById('bias-bar-center');
    const rightBar = document.getElementById('bias-bar-right');
    
    const leftVal = document.getElementById('bias-val-left');
    const centerVal = document.getElementById('bias-val-center');
    const rightVal = document.getElementById('bias-val-right');
    
    if (leftBar && result.dir) {
      leftBar.style.width = `${(result.dir.L * 100).toFixed(0)}%`;
      leftVal.textContent = `${(result.dir.L * 100).toFixed(0)}%`;
    }
    
    if (centerBar && result.dir) {
      centerBar.style.width = `${(result.dir.C * 100).toFixed(0)}%`;
      centerVal.textContent = `${(result.dir.C * 100).toFixed(0)}%`;
    }
    
    if (rightBar && result.dir) {
      rightBar.style.width = `${(result.dir.R * 100).toFixed(0)}%`;
      rightVal.textContent = `${(result.dir.R * 100).toFixed(0)}%`;
    }
    
    // Update degree meter
    const lowSeg = document.getElementById('bias-deg-low');
    const medSeg = document.getElementById('bias-deg-medium');
    const highSeg = document.getElementById('bias-deg-high');
    
    if (lowSeg) lowSeg.classList.remove('bias-meter-active');
    if (medSeg) medSeg.classList.remove('bias-meter-active');
    if (highSeg) highSeg.classList.remove('bias-meter-active');
    
    if (result.deg) {
      const maxDeg = Math.max(result.deg.L, result.deg.M, result.deg.H);
      if (maxDeg === result.deg.L && lowSeg) lowSeg.classList.add('bias-meter-active');
      else if (maxDeg === result.deg.M && medSeg) medSeg.classList.add('bias-meter-active');
      else if (maxDeg === result.deg.H && highSeg) highSeg.classList.add('bias-meter-active');
    }
    
    // Update reason
    const reasonEl = document.getElementById('bias-reason');
    if (reasonEl) {
      reasonEl.textContent = result.reason || 'Analysis complete';
    }
    
    // Update sample count
    const samplesEl = document.getElementById('bias-samples');
    if (samplesEl) {
      samplesEl.textContent = `${state.biasHistory.length} samples`;
    }
  }

  // ========================================
  // GRAPH DRAWING
  // ========================================
  
  function drawGraph() {
    if (!state.canvas || !state.ctx) return;
    
    const ctx = state.ctx;
    const width = state.canvas.width;
    const height = state.canvas.height;
    
    // Clear the canvas completely (no ghosting)
    ctx.clearRect(0, 0, width, height);
    
    // Fill with background color
    ctx.fillStyle = 'rgba(0, 0, 0, 0.2)';
    ctx.fillRect(0, 0, width, height);
    
    // Draw center line (neutral)
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(0, height / 2);
    ctx.lineTo(width, height / 2);
    ctx.stroke();
    ctx.setLineDash([]);
    
    // Draw left/right zone labels
    ctx.fillStyle = 'rgba(59, 130, 246, 0.3)'; // Blue for left
    ctx.fillRect(0, 0, width, height / 2);
    
    ctx.fillStyle = 'rgba(239, 68, 68, 0.3)'; // Red for right
    ctx.fillRect(0, height / 2, width, height / 2);
    
    // Draw Y-axis labels
    ctx.fillStyle = 'rgba(255, 255, 255, 0.5)';
    ctx.font = '9px sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText('L', 4, 12);
    ctx.fillText('R', 4, height - 4);
    ctx.fillText('C', 4, height / 2 + 3);
    
    // Draw data
    const history = state.biasHistory;
    if (history.length < 1) {
      // Draw "waiting" text
      ctx.fillStyle = 'rgba(255, 255, 255, 0.3)';
      ctx.font = '12px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('Waiting for data...', width / 2, height / 2);
      return;
    }
    
    const xStep = width / (state.maxHistoryPoints - 1);
    
    // Calculate bias score: -1 (left) to +1 (right)
    // Formula: R - L (positive = right, negative = left)
    const getBiasScore = (point) => {
      const L = point.dir?.L || 0;
      const R = point.dir?.R || 0;
      return R - L; // -1 to +1
    };
    
    // Get color based on bias score
    const getBiasColor = (score) => {
      if (score < -0.2) return '#3b82f6'; // Blue (left)
      if (score > 0.2) return '#ef4444';  // Red (right)
      return '#6b7280'; // Gray (center)
    };
    
    // Draw bias line with gradient coloring
    ctx.lineWidth = 3;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    
    for (let i = 1; i < history.length; i++) {
      const prevPoint = history[i - 1];
      const point = history[i];
      
      const prevScore = getBiasScore(prevPoint);
      const score = getBiasScore(point);
      
      const x1 = (i - 1) * xStep;
      const x2 = i * xStep;
      
      // Y position: 0 = left (top), 1 = right (bottom)
      const y1 = (prevScore + 1) / 2 * height; // Convert -1..+1 to 0..height
      const y2 = (score + 1) / 2 * height;
      
      // Draw line segment
      ctx.strokeStyle = getBiasColor(score);
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.stroke();
    }
    
    // Draw dots at each data point
    history.forEach((point, i) => {
      const score = getBiasScore(point);
      const x = i * xStep;
      const y = (score + 1) / 2 * height;
      
      ctx.fillStyle = getBiasColor(score);
      ctx.beginPath();
      ctx.arc(x, y, 3, 0, Math.PI * 2);
      ctx.fill();
    });
    
    // Draw current bias indicator (larger dot at end)
    if (history.length > 0) {
      const lastPoint = history[history.length - 1];
      const lastScore = getBiasScore(lastPoint);
      const lastX = (history.length - 1) * xStep;
      const lastY = (lastScore + 1) / 2 * height;
      
      // Outer glow
      ctx.fillStyle = getBiasColor(lastScore);
      ctx.beginPath();
      ctx.arc(lastX, lastY, 6, 0, Math.PI * 2);
      ctx.fill();
      
      // Inner dot
      ctx.fillStyle = '#fff';
      ctx.beginPath();
      ctx.arc(lastX, lastY, 3, 0, Math.PI * 2);
      ctx.fill();
      
      // Draw bias label at end
      ctx.fillStyle = '#fff';
      ctx.font = 'bold 10px sans-serif';
      ctx.textAlign = 'right';
      let label = 'C';
      if (lastScore < -0.2) label = 'L';
      else if (lastScore > 0.2) label = 'R';
      ctx.fillText(label, width - 4, lastY + 4);
    }
  }
  
  // ========================================
  // BIAS METER DRAWING (Analog Gauge)
  // ========================================
  
  function drawBiasMeter() {
    if (!state.meterCanvas || !state.meterCtx) return;
    
    const ctx = state.meterCtx;
    const width = state.meterCanvas.width;
    const height = state.meterCanvas.height;
    
    // Clear
    ctx.clearRect(0, 0, width, height);
    
    // Background
    ctx.fillStyle = 'rgba(0, 0, 0, 0.2)';
    ctx.fillRect(0, 0, width, height);
    
    // Get latest bias data
    const history = state.biasHistory;
    if (history.length === 0) {
      ctx.fillStyle = 'rgba(255, 255, 255, 0.3)';
      ctx.font = '11px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('No data yet', width / 2, height / 2 + 4);
      return;
    }
    
    const latest = history[history.length - 1];
    const L = latest.dir?.L || 0;
    const C = latest.dir?.C || 0;
    const R = latest.dir?.R || 0;
    
    // Meter settings
    const centerX = width / 2;
    const centerY = height - 10;
    const radius = Math.min(width / 2 - 20, height - 25);
    
    // Draw arc background (semi-circle on top)
    // Clockwise from PI (left) through 3PI/2 (top) to 0 (right)
    ctx.beginPath();
    ctx.arc(centerX, centerY, radius, Math.PI, 0, true);
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
    ctx.lineWidth = 8;
    ctx.stroke();
    
    // Draw colored zones
    // Left zone (blue) - from PI to 5PI/4
    ctx.beginPath();
    ctx.arc(centerX, centerY, radius, Math.PI, Math.PI * 1.25, true);
    ctx.strokeStyle = 'rgba(59, 130, 246, 0.6)';
    ctx.lineWidth = 8;
    ctx.stroke();
    
    // Center zone (gray) - from 5PI/4 to 3PI/4
    ctx.beginPath();
    ctx.arc(centerX, centerY, radius, Math.PI * 1.25, Math.PI * 0.75, true);
    ctx.strokeStyle = 'rgba(107, 114, 128, 0.6)';
    ctx.lineWidth = 8;
    ctx.stroke();
    
    // Right zone (red) - from 3PI/4 to 0
    ctx.beginPath();
    ctx.arc(centerX, centerY, radius, Math.PI * 0.75, 0, true);
    ctx.strokeStyle = 'rgba(239, 68, 68, 0.6)';
    ctx.lineWidth = 8;
    ctx.stroke();
    
    // Draw tick marks
    for (let i = 0; i <= 10; i++) {
      // Angle goes from PI (left) to 0 (right) clockwise
      const angle = Math.PI - (i / 10) * Math.PI;
      const innerR = radius - 12;
      const outerR = radius - 4;
      
      const x1 = centerX + Math.cos(angle) * innerR;
      const y1 = centerY + Math.sin(angle) * innerR;
      const x2 = centerX + Math.cos(angle) * outerR;
      const y2 = centerY + Math.sin(angle) * outerR;
      
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
      ctx.lineWidth = i % 5 === 0 ? 2 : 1;
      ctx.stroke();
    }
    
    // Draw labels
    ctx.fillStyle = 'rgba(255, 255, 255, 0.7)';
    ctx.font = 'bold 10px sans-serif';
    ctx.textAlign = 'center';
    
    // Left label (at left end of arc)
    const leftLabelX = centerX + Math.cos(Math.PI) * (radius + 12);
    const leftLabelY = centerY + Math.sin(Math.PI) * (radius + 12);
    ctx.fillText('L', leftLabelX, leftLabelY);
    
    // Center label (at top of arc)
    const centerLabelX = centerX + Math.cos(-Math.PI / 2) * (radius + 12);
    const centerLabelY = centerY + Math.sin(-Math.PI / 2) * (radius + 12);
    ctx.fillText('C', centerLabelX, centerLabelY);
    
    // Right label (at right end of arc)
    const rightLabelX = centerX + Math.cos(0) * (radius + 12);
    const rightLabelY = centerY + Math.sin(0) * (radius + 12);
    ctx.fillText('R', rightLabelX, rightLabelY);
    
    // Calculate needle angle
    // Bias score: -1 (left) to +1 (right)
    // Angle: PI (left) to 0 (right), with -PI/2 being center (top)
    const biasScore = R - L;
    // Map biasScore from [-1, 1] to angle [PI, 0] through -PI/2
    // Using: angle = PI - (biasScore + 1) * PI / 2
    // For biasScore = -1: angle = PI - 0 = PI (left)
    // For biasScore = 0: angle = PI - PI/2 = PI/2... but we want -PI/2 (up)
    // 
    // Correct mapping: angle = PI + (biasScore + 1) * PI / 2, then normalize
    // For biasScore = -1: angle = PI + 0 = PI (left) ✓
    // For biasScore = 0: angle = PI + PI/2 = 3PI/2 = -PI/2 (up) ✓
    // For biasScore = 1: angle = PI + PI = 2PI = 0 (right) ✓
    let needleAngle = Math.PI + (biasScore + 1) * Math.PI / 2;
    if (needleAngle > Math.PI * 2) needleAngle -= Math.PI * 2;
    
    // Get needle color based on bias
    const getNeedleColor = (score) => {
      if (score < -0.2) return '#3b82f6'; // Blue (left)
      if (score > 0.2) return '#ef4444';  // Red (right)
      return '#6b7280'; // Gray (center)
    };
    
    // Draw needle
    const needleLength = radius - 15;
    const needleX = centerX + Math.cos(needleAngle) * needleLength;
    const needleY = centerY + Math.sin(needleAngle) * needleLength;
    
    // Needle shadow
    ctx.beginPath();
    ctx.moveTo(centerX, centerY);
    ctx.lineTo(needleX + 1, needleY + 1);
    ctx.strokeStyle = 'rgba(0, 0, 0, 0.3)';
    ctx.lineWidth = 3;
    ctx.stroke();
    
    // Needle
    ctx.beginPath();
    ctx.moveTo(centerX, centerY);
    ctx.lineTo(needleX, needleY);
    ctx.strokeStyle = getNeedleColor(biasScore);
    ctx.lineWidth = 2;
    ctx.lineCap = 'round';
    ctx.stroke();
    
    // Center dot
    ctx.beginPath();
    ctx.arc(centerX, centerY, 5, 0, Math.PI * 2);
    ctx.fillStyle = '#fff';
    ctx.fill();
    
    // Draw percentage values at top
    ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
    ctx.font = 'bold 10px sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText(`L: ${(L * 100).toFixed(0)}%`, 8, 14);
    ctx.textAlign = 'center';
    ctx.fillText(`C: ${(C * 100).toFixed(0)}%`, width / 2, 14);
    ctx.textAlign = 'right';
    ctx.fillText(`R: ${(R * 100).toFixed(0)}%`, width - 8, 14);
  }

  // ========================================
  // MESSAGE HANDLING
  // ========================================
  
  function handleMessage(message, sender, sendResponse) {
    // Check extension context
    if (!extensionContextValid || !checkExtensionContext()) {
      console.log('[BiasDetector] Extension context invalid, cannot handle message');
      try {
        sendResponse({ error: 'Extension context invalidated. Please reload the page.' });
      } catch (e) {
        // Ignore - can't send response if context is invalid
      }
      return false;
    }
    
    console.log('[BiasDetector] Message received:', message.type);
    
    switch (message.type) {
      case 'ANALYZE_TEXT':
        // Manual analysis from popup
        callBiasAPI(message.text)
          .then(result => {
            if (checkExtensionContext()) {
              sendResponse({ success: true, result });
            }
          })
          .catch(error => {
            if (checkExtensionContext()) {
              sendResponse({ success: false, error: error.message });
            }
          });
        return true;
        
      case 'GET_STATE':
        sendResponse({
          isYouTube: state.isYouTube,
          bufferLength: state.captionBuffer.length,
          historyLength: state.biasHistory.length,
          isAnalyzing: state.isAnalyzing
        });
        return false;
        
      case 'TOGGLE_OVERLAY':
        if (state.overlay) {
          state.overlay.classList.toggle('bias-detector-hidden');
        }
        sendResponse({ success: true });
        return false;
        
      case 'CLEAR_HISTORY':
        state.biasHistory = [];
        drawGraph();
        sendResponse({ success: true });
        return false;
        
      default:
        sendResponse({ error: 'Unknown message type' });
        return false;
    }
  }

  // ========================================
  // START
  // ========================================
  
  // Wait for DOM ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
  
})();
