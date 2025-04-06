import { useEffect, useState } from 'react';
import { io } from 'socket.io-client';
import { ShoppingCart, Package, CreditCard, Trash2 } from 'lucide-react';

interface CartItem {
  id: number;
  name: string;
  price: number;
  quantity?: number;
  payable?: number;
  imageUrl?: string;  

}

const socket = io('http://localhost:3000');
socket.on('connect', () => console.log('Connected to WebSocket'));

function App() {
  // State to store cart items and total amount
  const [items, setItems] = useState<CartItem[]>([]);
  const [total, setTotal] = useState<number>(0);

  // Function to group items by name and sum quantities with proper typing
  const groupItems = (cartItems: CartItem[]): CartItem[] => {
    if (!Array.isArray(cartItems)) {
      console.error('Expected array but received:', cartItems);
      return [];
    }

    const grouped: Record<string, CartItem> = {};
    
    cartItems.forEach(item => {
      if (!item || typeof item !== 'object') {
        console.error('Invalid item format:', item);
        return;
      }

      if (!grouped[item.name]) {
        grouped[item.name] = {
          ...item,
          quantity: 1,
          payable: item.price
        };
      } else {
        grouped[item.name].quantity = (grouped[item.name].quantity || 0) + 1;
        grouped[item.name].payable = (grouped[item.name].payable || 0) + item.price;
      }
    });
    
    return Object.values(grouped);
  };

  // Fetch initial cart data from the backend
  useEffect(() => {
    const fetchItems = async () => {
      try {
        const response = await fetch('http://localhost:3000/api/product');
        if (!response.ok) {
          throw new Error('Failed to fetch items');
        }
        const data = await response.json();

        setItems(groupItems(data)); // Update the state with the received items
      } catch (error) {
        console.error('Error fetching items:', error);
      }
    };

    fetchItems();

    // Listen for real-time cart updates via socket.io
    socket.on('cartUpdate', (updatedCart) => {
      setItems(groupItems(updatedCart));  
    });

    // Cleanup socket listener on component unmount
    
  }, []);

  // Recalculate the total whenever the items change
  useEffect(() => {
    const newTotal = items.reduce((sum, item) => sum + item.payable, 0);
    setTotal(newTotal);
  }, [items]);

  // Handle checkout
  const handleCheckout = async () => {
    try {
      const response = await fetch('http://localhost:3000/api/checkout', {
        method: 'POST',
      });
      if (!response.ok) {
        throw new Error('Checkout failed');
      }
      const data = await response.json();
      if (data.success) {
        alert('Order placed successfully!');
        setItems([]); // Clear the cart after successful checkout
      }
    } catch (error) {
      console.error('Checkout failed:', error);
      alert('Checkout failed. Please try again.');
    }
  };

  // Handle clearing the cart
  const handleClearCart = async () => {
    try {
      const response = await fetch('http://localhost:3000/product', {
        method: 'GET',
      });
      if (!response.ok) {
        throw new Error('Failed to clear cart');
      }
      setItems([]); // Clear the cart in the state
    } catch (error) {
      console.error('Failed to clear cart:', error);
      alert('Failed to clear cart. Please try again.');
    }
  };

  return (
    <div className=" min-h-screen bg-white">
      <header className="bg-slate-950 shadow rounded-b-xl">
        <div className="max-w-7xl mx-auto px-4 py-6 sm:px-6 lg:px-8 flex justify-between items-center">
          <div className="flex items-center space-x-3">
            <Package className="h-8 w-8 text-indigo-600" />
            <h1 className="text-3xl font-bold text-gray-200">Raptail AI</h1>
          </div>
            
          <div className="flex items-center space-x-4">
            <ShoppingCart className="h-6 w-6 text-white" />
            <span className=" text-white text-lg font-semibold">{  items.reduce((total, item) => total + (item.quantity || 1), 0)} items</span>

           
          </div>
       
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4  lg:px-8">
     

        <div className="  min-h-full mt-8  rounded-lg p-6">
          <div className="flex   justify-between items-center  bg-slate-950 border border-gray-900  mb-2   rounded-2xl p-8">
            <h2 className="text-2xl font-semibold text-white">Shopping Cart</h2>
            {items.length > 0 && (
              <button onClick={handleClearCart} className="flex items-center space-x-2 text-red-600 hover:text-red-700">
                <Trash2 className="h-5 w-5" />
                <span>Clear Cart</span>
              </button>
            )}
          </div>

          {items.length === 0 ? (
           
             <div className="text-center py-12  border border-gray-200    rounded-2xl">
              <Package className="mx-auto h-12 w-12 text-gray-300" />
              <p className="mt-2 text-gray-400">Your cart is empty</p>
            </div>   
          ) : (
            <>
              <div className=" border border-gray-200 bg-white shadow  rounded-2xl  ">
                {items.map((item) => (
                  <div key={item.name} className="  border-t border-gray-300 rounded-2xl p-5  flex items-center justify-between">
                    <div className="p-2 gap-2 flex ">
                      <div className="flex size-14 bg-gray-300 rounded-[5px] border ">
                      <img 
  src={`http://localhost:3000/images/${item.imageUrl || 'default.png'}`}
  alt={item.name}
  className="size-14 object-cover rounded-[5px] border"
  onError={(e) => {
    (e.target as HTMLImageElement).src = 'http://localhost:3000/images/default.png';
  }}
/>                                      
                      </div>
                      <div className="flex-col  ">
                      <h3 className="text-lg font-medium text-gray-900 ">{item.name}</h3>
                      <p className="text-sm text-gray-500">
                      <span >&#x20B9;</span>{item.price.toFixed(2)} per item
                      </p>
                      </div>
                    </div>
                    <div className="flex items-center gap-3.5 place-items-start  ">
                      
                      <div className="text-gray-600 border   border-gray-200  rounded-xl text-nowrap  p-1.5">Qty: {item.quantity}</div>
                      <div className="text-lg font-medium text-gray-900 w-20">
                      <span >&#x20B9;</span>{item.payable.toFixed(2)}
                      </div>
                    </div>
                  </div>
                ))}
              </div>

              <div className="border-t border-gray-200 pt-6 mt-3">
                <div className="bg-white shadow border border-gray-200 rounded-2xl p-5">
                <div className="flex justify-between items-center">
                  <span className="text-xl font-semibold text-gray-900">Total</span>
                  <span className="text-2xl font-bold text-indigo-600">
                  <span >&#x20B9;</span>{total.toFixed(2)}
                  </span>
                </div>
                

                <button
                  onClick={handleCheckout}
                  className="mt-6 w-full flex items-center justify-center px-6 py-3 border border-transparent rounded-md shadow-sm text-base font-medium text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
                >
                  <CreditCard className="h-5 w-5 mr-2" />
                  Proceed to Checkout
                </button>
              </div>
              </div>
            </>
          )}
        </div>
      </main>
    </div>
  );
}

export default App;