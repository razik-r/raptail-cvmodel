import express from 'express';
import bodyParser from 'body-parser';
import cors from 'cors';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { Server as SocketIO } from 'socket.io';
import { createServer } from 'http';
import { createRequire } from 'module';

const require = createRequire(import.meta.url);
const sqlite3 = require('sqlite3').verbose();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const port = Number(process.env.PORT || 3000);
const frontendOrigin = process.env.FRONTEND_ORIGIN || 'http://localhost:5173';

const server = createServer(app);
const io = new SocketIO(server, {
  cors: {
    origin: frontendOrigin,
    methods: ['GET', 'POST', 'DELETE'],
    credentials: true,
  },
  transports: ['websocket', 'polling'],
});

let cart = [];
let orders = [];

app.use(
  cors({
    origin: frontendOrigin,
    methods: ['GET', 'POST', 'DELETE'],
    credentials: true,
  }),
);
app.use(bodyParser.urlencoded({ extended: false }));
app.use(bodyParser.json());

const dbPath = process.env.DB_PATH || path.join(__dirname, 'inventory.db');
const db = new sqlite3.Database(dbPath, (err) => {
  if (err) {
    console.error('Failed to connect to SQLite database:', err.message);
  } else {
    console.log(`Connected to inventory database at ${dbPath}`);
  }
});

const imageDirectory = process.env.IMAGE_DIR || path.join(__dirname, 'images');
app.use('/images', express.static(imageDirectory));

app.get('/health', (_req, res) => {
  res.json({ status: 'ok' });
});

app.get('/api/products', (_req, res) => {
  db.all('SELECT id, name, price, image_path as imageUrl FROM inventory', [], (err, rows) => {
    if (err) {
      res.status(500).json({ error: err.message });
      return;
    }

    res.json(rows);
  });
});

app.get('/api/cart', (_req, res) => {
  res.json(cart);
});

app.post('/api/cart', (req, res) => {
  const product = req.body;

  if (!product || typeof product !== 'object') {
    res.status(400).json({ error: 'Invalid product payload.' });
    return;
  }

  cart.push(product);
  io.emit('cartUpdate', cart);

  res.status(201).json({ message: 'Product added to cart.' });
});

app.delete('/api/cart', (_req, res) => {
  cart = [];
  io.emit('cartUpdate', cart);

  res.json({ message: 'Cart cleared.' });
});

app.post('/api/checkout', (req, res) => {
  const order = req.body || { items: cart };
  orders.push(order);

  cart = [];
  io.emit('cartUpdate', cart);

  res.json({ success: true, message: 'Checkout completed for demo flow.' });
});

app.get('/api/checkout', (_req, res) => {
  res.json(orders);
});

// Backward-compatible aliases
app.get('/product', (_req, res) => res.json(cart));
app.post('/api/product', (req, res) => {
  cart.push(req.body);
  io.emit('cartUpdate', cart);
  res.status(201).json({ message: 'Product added to cart.' });
});

io.on('connection', (socket) => {
  console.log('Client connected:', socket.id);

  socket.on('disconnect', () => {
    console.log('Client disconnected:', socket.id);
  });
});

const staticDir = process.env.STATIC_DIR || path.resolve(__dirname, '../../dist');
if (fs.existsSync(staticDir)) {
  app.use(express.static(staticDir));

  app.get('*', (req, res, next) => {
    if (req.path.startsWith('/api') || req.path.startsWith('/images')) {
      next();
      return;
    }

    res.sendFile(path.join(staticDir, 'index.html'));
  });
}

server.listen(port, () => {
  console.log(`Server listening on port ${port}`);
});
