import { create } from 'zustand';

interface CartItem {
  id: number;
  name: string;
  price: number;
  units: string;
  quantity: number;
  payable: number;
}

interface CartStore {
  items: CartItem[];
  setItems: (items: CartItem[]) => void;
  clearCart: () => void;
}

export const useCartStore = create<CartStore>((set) => ({
  items: [],
  setItems: (items) => set({ items }),
  clearCart: () => set({ items: [] }),
}));