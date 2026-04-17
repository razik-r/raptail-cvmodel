# Smart Checkout System

A real-time self-checkout demo that combines:
- React + Vite frontend
- Express + Socket.IO backend
- SQLite inventory database

## Run locally (demo mode)

```bash
npm install
npm run dev
```

- Frontend: `http://localhost:5173`
- Backend API + Socket server: `http://localhost:3000`

## Deploy as a demonstrable product

This repository now supports a single-process deployment: the backend serves API endpoints, WebSocket updates, product images, and the built frontend UI.

### 1) Build frontend assets

```bash
npm run build
```

### 2) Start production server

```bash
npm run start
```

Then open:
- `http://<your-host>:3000`

## Environment variables

Create `.env` from `.env.example` when needed.

| Variable | Default | Purpose |
|---|---|---|
| `PORT` | `3000` | HTTP server port |
| `FRONTEND_ORIGIN` | `http://localhost:5173` | CORS + Socket.IO origin for development |
| `DB_PATH` | `src/server/inventory.db` | SQLite file location |
| `IMAGE_DIR` | `src/server/images` | Product image directory |
| `STATIC_DIR` | `dist` | Built frontend assets directory |
| `VITE_API_BASE_URL` | `http://localhost:3000` | Frontend API/Socket base URL |

## Docker deployment

### Build image

```bash
docker build -t smart-checkout-demo .
```

### Run container

```bash
docker run --rm -p 3000:3000 smart-checkout-demo
```

Open `http://localhost:3000`.

## API quick check

```bash
curl http://localhost:3000/health
curl http://localhost:3000/api/cart
curl http://localhost:3000/api/products
```
