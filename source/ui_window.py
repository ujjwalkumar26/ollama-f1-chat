import tkinter as tk
from tkinter import scrolledtext, ttk
import queue


class ChatUI:
    def __init__(self):
        self.running = True
        self.message_queue = queue.Queue()
        self.pending_input = queue.Queue()
        
        # Create window
        self.root = tk.Tk()
        self.root.title("F1 Data Assistant")
        self.root.geometry("700x600")
        
        # Chat display
        self.chat_area = scrolledtext.ScrolledText(
            self.root, 
            wrap=tk.WORD, 
            state='disabled',
            font=('Arial', 10),
            bg='#1e1e1e',
            fg='#ffffff'
        )
        self.chat_area.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        # Configure tags for colors
        self.chat_area.tag_config('user', foreground='#58a6ff')
        self.chat_area.tag_config('assistant', foreground='#7ee787')
        self.chat_area.tag_config('system', foreground='#a5a5a5')
        self.chat_area.tag_config('error', foreground='#ff6b6b')
        
        # Input frame
        input_frame = ttk.Frame(self.root)
        input_frame.pack(padx=10, pady=(0, 10), fill=tk.X)
        
        self.input_field = ttk.Entry(input_frame, font=('Arial', 10))
        self.input_field.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        self.input_field.bind('<Return>', self._on_send)
        self.input_field.focus()
        
        send_btn = ttk.Button(input_frame, text="Send", command=self._on_send)
        send_btn.pack(side=tk.RIGHT)
        
        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        
        # Start UI update loop
        self._update_ui()
    
    def _on_send(self, event=None):
        user_input = self.input_field.get().strip()
        if user_input:
            self.pending_input.put(user_input)
            self.input_field.delete(0, tk.END)
    
    def _on_close(self):
        self.running = False
        self.root.destroy()
    
    def _update_ui(self):
        """Process pending messages"""
        while not self.message_queue.empty():
            try:
                text, msg_type = self.message_queue.get_nowait()
                self._display_message(text, msg_type)
            except queue.Empty:
                break
        
        if self.running:
            self.root.after(100, self._update_ui)
    
    def _display_message(self, text, msg_type):
        self.chat_area.config(state='normal')
        
        prefix = {
            'user': '👤 You: ',
            'assistant': '🤖 Assistant: ',
            'system': '🔧 System: ',
            'error': '❌ Error: '
        }.get(msg_type, '')
        
        self.chat_area.insert(tk.END, prefix + text + '\n\n', msg_type)
        self.chat_area.see(tk.END)
        self.chat_area.config(state='disabled')
    
    def add_message(self, text, msg_type='system'):
        """Add message to display queue (thread-safe)"""
        self.message_queue.put((text, msg_type))
    
    def has_pending_message(self):
        """Check if user has sent a message"""
        return not self.pending_input.empty()
    
    def get_pending_message(self):
        """Get user's message"""
        try:
            return self.pending_input.get_nowait()
        except queue.Empty:
            return None
    
    def run(self):
        """Start the UI main loop (must be called from main thread)"""
        self.root.mainloop()