---
description: Open the RailDrishti portal — starts backend + frontend servers and opens in browser
---
// turbo-all

## Steps

1. Kill any existing processes on port 8000 to avoid conflicts:
```
Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess | ForEach-Object { Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue }
```

2. Build the frontend (needed because Vite dev server doesn't work with `#` in folder path):
```
npm run build
```
Run from: `railway_ai_system/frontend`

3. Start the FastAPI backend server (this also serves the built frontend):
```
conda activate hack4delhi; uvicorn api.main:app --reload --port 8000
```
Run from: `railway_ai_system`
Wait for "Application startup complete" in the output.

4. Open the portal in the browser at http://localhost:8000/
