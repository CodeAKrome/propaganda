/**
 * Simple test script to diagnose YouTube caption capture
 * Run this in the browser console on a YouTube video page
 */

(function() {
  'use strict';
  
  console.log('=== YouTube Caption Capture Test ===');
  
  // Test 1: Check if video exists
  const video = document.querySelector('video');
  console.log('1. Video element:', video ? 'FOUND' : 'NOT FOUND');
  if (video) {
    console.log('   - Paused:', video.paused);
    console.log('   - Current time:', video.currentTime);
  }
  
  // Test 2: Check caption container
  const captionContainer = document.querySelector('.ytp-caption-window-container');
  console.log('2. Caption container:', captionContainer ? 'FOUND' : 'NOT FOUND');
  if (captionContainer) {
    console.log('   - innerHTML:', captionContainer.innerHTML?.substring(0, 200));
    console.log('   - textContent:', captionContainer.textContent?.trim());
    console.log('   - children:', captionContainer.children.length);
  }
  
  // Test 3: Check caption segments
  const segments = document.querySelectorAll('.ytp-caption-segment');
  console.log('3. Caption segments:', segments.length, 'found');
  if (segments.length > 0) {
    segments.forEach((seg, i) => {
      console.log(`   - Segment ${i}:`, seg.textContent?.trim());
    });
  }
  
  // Test 4: Check text tracks
  if (video && video.textTracks) {
    console.log('4. Text tracks:', video.textTracks.length);
    for (let i = 0; i < video.textTracks.length; i++) {
      const track = video.textTracks[i];
      console.log(`   - Track ${i}:`, {
        kind: track.kind,
        label: track.label,
        mode: track.mode,
        cueCount: track.cues?.length || 0,
        activeCueCount: track.activeCues?.length || 0
      });
    }
  }
  
  // Test 5: Check CC button
  const ccButton = document.querySelector('.ytp-subtitles-button');
  console.log('5. CC button:', ccButton ? 'FOUND' : 'NOT FOUND');
  if (ccButton) {
    console.log('   - aria-pressed:', ccButton.getAttribute('aria-pressed'));
    console.log('   - Enabled:', ccButton.getAttribute('aria-pressed') === 'true');
  }
  
  // Test 6: Watch for caption changes
  console.log('6. Setting up caption watcher for 10 seconds...');
  let captureCount = 0;
  
  const interval = setInterval(() => {
    const segs = document.querySelectorAll('.ytp-caption-segment');
    if (segs.length > 0) {
      let text = '';
      segs.forEach(seg => {
        const t = seg.textContent?.trim();
        if (t) text += t + ' ';
      });
      if (text.trim()) {
        captureCount++;
        console.log(`   [${captureCount}] Captured:`, text.trim().substring(0, 100));
      }
    }
  }, 500);
  
  setTimeout(() => {
    clearInterval(interval);
    console.log('=== Test Complete ===');
    console.log('Total captures:', captureCount);
    if (captureCount === 0) {
      console.log('WARNING: No captions captured. Possible issues:');
      console.log('  - Captions not enabled (click CC button)');
      console.log('  - Video has no captions');
      console.log('  - Video is paused');
    }
  }, 10000);
})();
