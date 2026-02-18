/**
 * Automated test for YouTube caption capture
 * 
 * Usage: node test-youtube-captions.js
 * 
 * Prerequisites:
 * - Puppeteer installed: npm install puppeteer
 * - Extension built and in browser-extension/ directory
 * - Local LLM server running on localhost:8000
 */

const puppeteer = require('puppeteer');
const path = require('path');
const fs = require('fs');

// Helper for waiting
const wait = (ms) => new Promise(resolve => setTimeout(resolve, ms));

async function testYouTubeCaptions() {
  console.log('=== YouTube Caption Capture Test ===\n');
  
  const extensionPath = path.resolve(__dirname);
  console.log('Extension path:', extensionPath);
  
  // Check if extension files exist
  const manifestPath = path.join(extensionPath, 'manifest.json');
  if (!fs.existsSync(manifestPath)) {
    console.error('ERROR: manifest.json not found at', manifestPath);
    process.exit(1);
  }
  
  console.log('Launching browser with extension...');
  
  const browser = await puppeteer.launch({
    headless: false, // Must be false for extension testing
    args: [
      `--disable-extensions-except=${extensionPath}`,
      `--load-extension=${extensionPath}`,
      '--no-sandbox',
      '--disable-setuid-sandbox',
      '--disable-web-security',
      '--allow-running-insecure-content',
    ],
    defaultViewport: null,
  });
  
  const page = await browser.newPage();
  
  // Enable console log from the page
  page.on('console', msg => {
    const text = msg.text();
    if (text.includes('[BiasDetector]')) {
      console.log('[PAGE]', text);
    }
  });
  
  // Log errors
  page.on('pageerror', error => {
    console.error('[PAGE ERROR]', error.message);
  });
  
  try {
    // Navigate to a simple test page first
    console.log('\n--- Testing extension loading ---');
    await page.goto('https://example.com', { waitUntil: 'networkidle2', timeout: 30000 });
    await wait(2000);
    
    // Check if the extension's overlay exists
    const overlayExists = await page.evaluate(() => {
      return document.getElementById('bias-detector-overlay') !== null;
    });
    
    if (overlayExists) {
      console.log('SUCCESS: Extension overlay injected into page');
    } else {
      console.log('FAILED: Extension overlay not found');
    }
    
    // Now try YouTube (may not work due to bot detection)
    console.log('\n--- Attempting YouTube test ---');
    try {
      await page.goto('https://www.youtube.com/watch?v=arj7oStkKLM', { waitUntil: 'domcontentloaded', timeout: 15000 });
      await wait(5000);
      
      // Check for video element
      const videoStatus = await page.evaluate(() => {
        const video = document.querySelector('video');
        const overlay = document.getElementById('bias-detector-overlay');
        
        return {
          hasVideo: !!video,
          videoPaused: video ? video.paused : null,
          videoDuration: video ? video.duration : null,
          hasOverlay: !!overlay,
          overlayHidden: overlay ? overlay.classList.contains('bias-detector-hidden') : null
        };
      });
      
      console.log('YouTube page status:', JSON.stringify(videoStatus, null, 2));
      
      if (videoStatus.hasVideo && videoStatus.hasOverlay) {
        console.log('SUCCESS: Video and overlay found on YouTube');
      } else {
        console.log('WARNING: Video or overlay not found - may be bot detection');
      }
    } catch (ytError) {
      console.log('YouTube test skipped:', ytError.message);
    }
    
    // Wait and monitor for caption capture
    console.log('\n--- Monitoring caption capture (30 seconds) ---');
    await wait(5000);
    
    // Check for caption track status
    const captionStatus = await page.evaluate(() => {
      const video = document.querySelector('video');
      if (!video) return { error: 'No video element' };
      
      const tracks = video.textTracks;
      const trackInfo = [];
      
      if (tracks) {
        for (let i = 0; i < tracks.length; i++) {
          const track = tracks[i];
          trackInfo.push({
            kind: track.kind,
            label: track.label,
            language: track.language,
            mode: track.mode,
            cueCount: track.cues ? track.cues.length : 0
          });
        }
      }
      
      return {
        trackCount: tracks ? tracks.length : 0,
        tracks: trackInfo
      };
    });
    
    console.log('\nCaption track status:', JSON.stringify(captionStatus, null, 2));
    
    // Wait for analysis to happen
    console.log('\n--- Waiting for bias analysis ---');
    await wait(20000);
    
    // Check for overlay
    console.log('\n--- Checking for bias overlay ---');
    const overlayStatus = await page.evaluate(() => {
      const overlay = document.getElementById('bias-detector-overlay');
      if (!overlay) return { found: false };
      
      const isHidden = overlay.classList.contains('bias-detector-hidden');
      const resultVisible = overlay.querySelector('.bias-detector-result')?.style.display !== 'none';
      
      // Get bias values
      const leftBar = overlay.querySelector('.bias-bar-left .bias-bar-value')?.textContent || '0%';
      const centerBar = overlay.querySelector('.bias-bar-center .bias-bar-value')?.textContent || '0%';
      const rightBar = overlay.querySelector('.bias-bar-right .bias-bar-value')?.textContent || '0%';
      
      // Get rolling text entries
      const rollingText = overlay.querySelectorAll('.bias-rolling-segment');
      
      // Get graph data points
      const graphCanvas = overlay.querySelector('#biasGraphCanvas');
      
      return {
        found: true,
        visible: !isHidden,
        resultVisible,
        bias: { left: leftBar, center: centerBar, right: rightBar },
        rollingTextCount: rollingText.length,
        graphCanvasExists: !!graphCanvas
      };
    });
    
    console.log('\nOverlay status:', JSON.stringify(overlayStatus, null, 2));
    
    // Check console logs for caption capture
    console.log('\n--- Test Results ---');
    
    if (captionStatus.trackCount > 0) {
      console.log('SUCCESS: Caption tracks found');
    } else {
      console.log('WARNING: No caption tracks found - video may not have captions');
    }
    
    if (overlayStatus.found) {
      console.log('SUCCESS: Bias overlay found');
      if (overlayStatus.resultVisible) {
        console.log('SUCCESS: Bias analysis displayed');
        console.log('Bias values:', overlayStatus.bias);
      } else {
        console.log('WARNING: Bias analysis not yet displayed');
      }
      
      if (overlayStatus.rollingTextCount > 0) {
        console.log('SUCCESS: Rolling text entries found:', overlayStatus.rollingTextCount);
      } else {
        console.log('WARNING: No rolling text entries yet');
      }
    } else {
      console.log('FAILED: Bias overlay not found');
    }
    
    // Keep browser open for manual inspection
    console.log('\n--- Keeping browser open for 30 seconds for manual inspection ---');
    await wait(30000);
    
  } catch (error) {
    console.error('\nERROR:', error.message);
    console.error(error.stack);
  } finally {
    console.log('\n--- Closing browser ---');
    await browser.close();
  }
}

// Run the test
testYouTubeCaptions().catch(console.error);