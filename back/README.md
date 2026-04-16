# Back - API Server

Express.js backend API server for the propaganda news analysis system.

## Quick Reference

| Task | Command |
|------|---------|
| Start server | `./RUNME.sh` |
| Run with node | `node server.js` |
| Install deps | `npm install` |

## Shell Scripts

### RUNME.sh
Starts the Express API server.

```bash
./RUNME.sh

# Runs: node server.js
# Default port: 3000 (or PORT env var)
```

## Running

```bash
# Install dependencies
cd back
npm install

# Start server
node server.js
# or
./RUNME.sh
```

## API Endpoints

The server provides REST endpoints for:
- Article search
- Entity retrieval
- Bias data access
- Report generation

See `server.js` for full endpoint documentation.

## Configuration

### Environment Variables
```bash
export PORT=3000
export MONGO_URI="mongodb://localhost:27017"
```

## See Also

- Frontend: [../front/README.md](../front/README.md)
- Main README: [../README.md](../README.md)