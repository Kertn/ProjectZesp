import os
import webbrowser
import tkinter as tk
from tkinter import scrolledtext, messagebox
from tkinter import ttk
import threading
from functools import partial
from datetime import datetime

# ====== Correct and Stable Imports ======
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# ========  CONFIGURATION  ========
os.environ['OPENAI_API_KEY'] = ''  # Add your key here

# ========  LOAD RAG COMPONENTS  ========
try:
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local('faiss_index', embeddings, allow_dangerous_deserialization=True)
    llm = ChatOpenAI(model_name="gpt-5-mini-2025-08-07", temperature=0.2)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 20})

    prompt_template = """Use the following pieces of context from documents to answer the question. 
If the question refers to a filename or document, summarize its key content based on all available context. 
If context is limited, describe what is available and note any gaps. 
Do not make up information.

Context: {context}

Question: {question}

Answer:"""

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
except Exception as e:
    messagebox.showerror("Initialization Error", f"Failed to initialize RAG model:\n{e}")
    raise

# ========= RAG QUERY FUNCTION ==========
def ask_question(query):
    """Ask a question using the RAG chain and return response + sources."""
    try:
        result = qa_chain({"query": query})
        response = result["result"]
        sources = [
            doc.metadata.get("source", "unknown") for doc in result.get("source_documents", [])
        ]
        return response, sources
    except Exception as e:
        return f"Error: {e}", []

# ========= HISTORY HANDLING ==========
HISTORY_FILE = "chat_history.txt"

def load_history():
    """Load chat history from the text file."""
    if not os.path.exists(HISTORY_FILE):
        return []
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        return f.read().split("\n---\n")

def save_entry(question, answer, sources):
    """Save a single conversation entry to history."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sources_str = "\n".join(sources) if sources else "No sources"
    entry = (
        f"[{timestamp}]\n"
        f"Q: {question}\n"
        f"A: {answer}\n"
        f"Sources:\n{sources_str}\n---\n"
    )
    with open(HISTORY_FILE, "a", encoding="utf-8") as f:
        f.write(entry)

# ========= TKINTER GUI ==========
class RAGApp:
    def __init__(self, root):
        self.root = root
        self.root.title("RAG Assistant - Local Chat")
        self.root.geometry("900x600")

        # For unique link tags
        self._link_counter = 0

        # --- Frames ---
        self.history_frame = tk.Frame(self.root, width=200)
        self.history_frame.pack(side=tk.LEFT, fill=tk.Y)

        self.chat_frame = tk.Frame(self.root)
        self.chat_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # --- History List ---
        tk.Label(self.history_frame, text="Chat History", font=("Helvetica", 12, "bold")).pack(pady=5)
        self.history_text = scrolledtext.ScrolledText(self.history_frame, wrap=tk.WORD, width=30)
        self.history_text.pack(fill=tk.BOTH, expand=True, padx=5)
        self.load_history_into_ui()

        # --- Chat Display ---
        self.chat_display = scrolledtext.ScrolledText(self.chat_frame, wrap=tk.WORD, height=25)
        self.chat_display.pack(fill=tk.BOTH, padx=10, pady=5, expand=True)
        self.chat_display.config(state=tk.DISABLED)

        # --- Progress area (initially hidden) ---
        self.progress_frame = tk.Frame(self.chat_frame)
        self.progress_label = tk.Label(self.progress_frame, text="Generating answer…")
        self.progress_label.pack(side=tk.LEFT, padx=6)
        self.progress = ttk.Progressbar(self.progress_frame, mode="indeterminate", length=220)
        self.progress.pack(side=tk.LEFT, padx=6)

        # --- Input Field ---
        self.input_field = tk.Entry(self.chat_frame, font=("Helvetica", 12))
        self.input_field.pack(fill=tk.X, padx=10, pady=5)
        self.input_field.bind("<Return>", self.send_message)

        # --- Buttons ---
        button_frame = tk.Frame(self.chat_frame)
        button_frame.pack(fill=tk.X, pady=5)

        self.send_button = tk.Button(button_frame, text="Send", command=self.send_message, bg="#4CAF50", fg="white")
        self.send_button.pack(side=tk.LEFT, padx=5)

        self.new_chat_button = tk.Button(button_frame, text="New Chat", command=self.start_new_chat, bg="#f44336", fg="white")
        self.new_chat_button.pack(side=tk.LEFT, padx=5)

    # --- Load existing history into sidebar ---
    def load_history_into_ui(self):
        entries = load_history()
        if entries:
            self.history_text.config(state=tk.NORMAL)
            for entry in entries:
                if entry.strip():
                    self.history_text.insert(tk.END, entry + "\n---\n")
            self.history_text.config(state=tk.DISABLED)

    # --- Clear chat area ---
    def start_new_chat(self):
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.delete("1.0", tk.END)
        self.chat_display.config(state=tk.DISABLED)
        self.input_field.delete(0, tk.END)

    # --- Send message and get answer (non-blocking) ---
    def send_message(self, event=None):
        query = self.input_field.get().strip()
        if not query:
            return

        self.display_message("You: " + query + "\n")
        self.input_field.delete(0, tk.END)

        # Show spinner and disable inputs
        self._start_progress()

        # Run heavy work in a background thread
        t = threading.Thread(target=self._run_query, args=(query,), daemon=True)
        t.start()

    def _run_query(self, query):
        try:
            response, sources = ask_question(query)
        except Exception as e:
            response, sources = f"Error: {e}", []
        finally:
            # Ensure UI update happens on main thread
            self.root.after(0, lambda: self._finish_query(query, response, sources))

    def _finish_query(self, query, response, sources):
        try:
            # Enable once, do all UI updates, then disable
            self.chat_display.config(state=tk.NORMAL)

            self.chat_display.insert(tk.END, "AI: " + response + "\n\n")
            if sources:
                self.chat_display.insert(tk.END, "Sources:\n")
                for src in sources:
                    self.make_link(src)  # assumes widget is already NORMAL
                self.chat_display.insert(tk.END, "\n")

            self.chat_display.config(state=tk.DISABLED)
            self.chat_display.see(tk.END)

            save_entry(query, response, sources)
            self.update_history()
        except Exception as ex:
            messagebox.showerror("Render error", f"Failed to render response/sources:\n{ex}")
        finally:
            self._stop_progress()

    # --- Display message in main text box ---
    def display_message(self, message):
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.insert(tk.END, message)
        self.chat_display.config(state=tk.DISABLED)
        self.chat_display.see(tk.END)

    # --- Make clickable links in sources (assumes Text is NORMAL) ---
    def _open_link(self, url):
        if url.startswith("http://") or url.startswith("https://"):
            webbrowser.open(url)
        else:
            messagebox.showinfo("Local Source", f"Local file or non-HTTP path:\n{url}")

    def make_link(self, link):
        start_idx = self.chat_display.index(tk.END)
        self.chat_display.insert(tk.END, link + "\n")
        end_idx = self.chat_display.index(tk.END)

        tag = f"src_link_{self._link_counter}"
        self._link_counter += 1

        self.chat_display.tag_add(tag, start_idx, end_idx)
        self.chat_display.tag_config(tag, foreground="blue", underline=True)
        self.chat_display.tag_bind(tag, "<Button-1>", lambda e, url=link: self._open_link(url))

    # --- Update sidebar history ---
    def update_history(self):
        self.history_text.config(state=tk.NORMAL)
        self.history_text.delete("1.0", tk.END)
        self.load_history_into_ui()
        self.history_text.config(state=tk.DISABLED)

    # --- Progress helpers ---
    def _start_progress(self):
        try:
            self.progress_frame.pack(fill=tk.X, padx=10, pady=5)
            self.progress.start(12)  # smaller = faster animation
        except Exception:
            pass
        self.send_button.config(state=tk.DISABLED)
        self.input_field.config(state=tk.DISABLED)
        self.root.config(cursor="watch")

    def _stop_progress(self):
        try:
            self.progress.stop()
            self.progress_frame.pack_forget()
        except Exception:
            pass
        self.send_button.config(state=tk.NORMAL)
        self.input_field.config(state=tk.NORMAL)
        self.input_field.focus_set()
        self.root.config(cursor="")


# ========= MAIN ==========
if __name__ == "__main__":
    root = tk.Tk()
    app = RAGApp(root)
    root.mainloop()