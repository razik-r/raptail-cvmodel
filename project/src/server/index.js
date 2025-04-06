import express from 'express';
import bodyParser from 'body-parser';
import cors from 'cors';

import { Server as SocketIO } from 'socket.io';
import { createServer } from 'http'; 
import { createRequire } from 'module';

const require = createRequire(import.meta.url);

const sqlite3 = require('sqlite3').verbose();


const app = express();
const port = process.env.PORT || 3000;

const server = createServer(app); 
const io = new SocketIO(server, {
    cors: {
      origin: "http://localhost:5173",
      methods: ["GET", "POST"],
      credentials: true
    },
    transports: ['websocket', 'polling']
  });


let products = [];
let orders = [];

app.use(cors(
    {
        origin: 'http://localhost:5173',
        methods: ['GET', 'POST']
      }
));
app.use(bodyParser.urlencoded({ extended: false }));
app.use(bodyParser.json());

const dbPath = 'C:\\Users\\razikr\\raptail-cvmodel\\my_model\\inventory.db'; 

const db = new sqlite3.Database(dbPath); // Your SQLite file
app.use('/images', express.static('images'));

// Example product endpoint
app.get('/product', (req, res) => {
    // Query should include image_path as imageUrl
    db.all("SELECT id, name, price, image_path as imageUrl FROM inventory", [], (err, rows) => {
        if (err) {

            res.status(500).json({error: err.message});
            return;
        }
        res.json(rows);
    });
});


app.get('/', (req, res) => {
    res.send("API deployment successful");
});



app.post('/api/product', (req, res) => {
    const product = req.body;

    // output the product to the console for debugging
    console.log(product);
    products.push(product);

    res.send('Product is added to the database');

    io.emit('cartUpdate', products);

   
});


// Connection handler
io.on('connection', (socket) => {
    console.log('Client connected:', socket.id);
    
    socket.on('disconnect', () => {
      console.log('Client disconnected:', socket.id);
    });
  });


app.get('/product', (req, res) => {
    res.json(products);
});

app.get('/product/:id', (req, res) => {
    const id = req.params.id;

    for (let product of products) {
        if (product.id === id) {
            res.json(product);
            return;
        }
    }

    res.status(404).send('Product not found');
});

app.delete('/product/:id', (req, res) => {
    const id = req.params.id;
    products = products.filter(i => i.id !== id);
    res.send('Product is deleted');
});

app.post('/product/:id', (req, res) => {
    const id = req.params.id;
    const newProduct = req.body;

    for (let i = 0; i < products.length; i++) {
        if (products[i].id === id) {
            products[i] = newProduct;
        }
    }

    res.send('Product is edited');
});

app.post('/checkout', (req, res) => {
    const order = req.body;
    orders.push(order);
    res.redirect(302, 'https://assettracker.cf');
});

app.get('/checkout', (req, res) => {
    res.json(orders);
});


  
server.listen(port, () => console.log(`Server listening on port ${port}!`));
