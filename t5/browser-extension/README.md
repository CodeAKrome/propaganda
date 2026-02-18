# Bias Detector Browser Extension

A Chrome/Firefox browser extension that detects political bias in news articles and YouTube captions in real-time using a fine-tuned T5 model.

## Features

- **Web Page Analysis**: Extract and analyze article text from any news page
- **YouTube Caption Analysis**: Real-time bias detection from video captions
- **Context Menu Integration**: Right-click to analyze selected text
- **Configurable API**: Connect to local MCP server or remote HTTP API
- **Result Caching**: Avoid re-analyzing identical content
- **Privacy-First**: All analysis happens locally via your own server

## Installation

### Prerequisites

1. **Bias Detection Server**: The extension requires a running bias detection API
   ```bash
   # Option 1: MCP Server (recommended)
   pip install bias-mcp-server
   bias-mcp-server
   # Runs on http://localhost:8000

   # Option 2: HTTP Server
   python server_mps.py
   # Runs on http://localhost:8000
   ```

### Load Extension (Development)

#### Chrome/Edge
1. Open `chrome://extensions/`
2. Enable "Developer mode" (top right)
3. Click "Load unpacked"
4. Select the `browser-extension` folder

#### Firefox
1. Open `about:debugging#/runtime/this-firefox`
2. Click "Load Temporary Add-on"
3. Select `manifest.json` from the `browser-extension` folder

### Generate Icons

The extension requires PNG icons. Generate them from the SVG:

```bash
# Using ImageMagick
cd browser-extension/icons
convert icon16.svg icon16.png
convert -background none -resize 32x32 icon16.svg icon32.png
convert -background none -resize 48x48 icon16.svg icon48.png
convert -background none -resize 128x128 icon16.svg icon128.png
```

Or use an online SVG to PNG converter.

## Usage

### Analyze a Web Page
1. Navigate to a news article
2. Click the extension icon in the toolbar
3. Click "Analyze This Page"
4. View results in the overlay

### Analyze Selected Text
1. Highlight text on any page
2. Right-click and select "Analyze bias"
3. View results in the overlay

### YouTube Caption Analysis
1. Navigate to a YouTube video with captions
2. Enable captions (CC button)
3. The extension automatically analyzes captions in real-time
4. Results update every 10-30 seconds

## Configuration

Access settings by clicking the gear icon in the popup or right-clicking the extension icon > Options.

| Setting | Description | Default |
|---------|-------------|---------|
| API Endpoint | URL of bias detection server | `http://localhost:8000` |
| API Type | Protocol (HTTP or MCP) | `HTTP` |
| Minimum Text Length | Characters required for analysis | `100` |
| Auto-analyze | Automatically analyze news pages | `Off` |
| Enable Caching | Cache analysis results | `On` |
| YouTube Analysis | Enable caption monitoring | `On` |
| Caption Buffer | Seconds of captions to accumulate | `30s` |

## Architecture

```
browser-extension/
  manifest.json       # Extension configuration (Manifest V3)
  background.js       # Service worker for API communication
  content.js          # Content script for page interaction
  overlay.css         # Styles for bias overlay
  popup.html/js/css   # Toolbar popup UI
  options.html/js/css # Settings page
  icons/              # Extension icons
```

### Data Flow

```
User Action
    |
    v
Content Script (content.js)
    | Extract text / captions
    v
Background Service Worker (background.js)
    | API call + caching
    v
Bias Detection Server (localhost:8000)
    | T5 model inference
    v
Response with bias scores
    |
    v
Overlay Display (content.js)
```

## API Compatibility

The extension supports two API formats:

### HTTP REST API
```json
POST /predict
{ "text": "Article text..." }

Response:
{
  "result": {
    "dir": { "L": 0.2, "C": 0.6, "R": 0.2 },
    "deg": { "L": 0.1, "M": 0.8, "H": 0.1 },
    "reason": "The article maintains..."
  },
  "device": "mps"
}
```

### MCP Server
```json
POST /analyze_bias
{ "text": "Article text..." }

Response:
{
  "direction": { "L": 0.2, "C": 0.6, "R": 0.2 },
  "degree": { "L": 0.1, "M": 0.8, "H": 0.1 },
  "reasoning": "The article maintains..."
}
```

## Troubleshooting

### "Offline" Status
- Ensure the bias detection server is running
- Check the API endpoint in settings
- Verify CORS is enabled on the server

### No Results on YouTube
- Enable captions (CC) on the video
- Wait for caption data to accumulate
- Check if video has auto-generated captions

### Extension Not Loading
- Check for errors in `chrome://extensions/`
- Verify all files are present
- Ensure icons are PNG format (not SVG)

## Development

### Build for Distribution

```bash
# Create ZIP for Chrome Web Store
cd browser-extension
zip -r ../bias-detector-extension.zip .

# Create XPI for Firefox
# Use web-ext tool
npm install -g web-ext
web-ext build
```

### Testing

1. Load extension in developer mode
2. Open a news article (e.g., CNN, Fox News, BBC)
3. Click "Analyze This Page"
4. Verify overlay appears with results

## Privacy

- **No data collection**: The extension does not send data to third parties
- **Local processing**: All analysis runs on your local server
- **No tracking**: No analytics or telemetry
- **Open source**: Full source code available for audit

## License

MIT License - See LICENSE file for details.

## Related Projects

- [bias-detector](../) - Python package for bias detection
- [mcp_bias_server](../mcp_bias_server/) - MCP server implementation
- [docker-inference](../docker-inference/) - Docker containers for inference