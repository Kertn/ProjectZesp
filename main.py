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
    try:
        messagebox.showerror("Initialization Error", f"Failed to initialize RAG model:\n{e}")
    except Exception:
        pass
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
    # --- Chat bubble config ---
    BUBBLE_MAX_WIDTH_CHARS = 70  # wrap width inside bubbles
    PREVIEW_LINES = 4
    PREVIEW_SPACING_BLANK_LINES = 2

    # Theme palettes (light/dark)
    THEMES = {
        "dark": {
            "BG": "#1e1e1e",
            "PANEL_BG": "#191919",
            "TEXT_BG": "#1f1f1f",
            "ENTRY_BG": "#252526",
            "BUTTON_BG": "#2d2d30",
            "BUTTON_ACTIVE_BG": "#3a3d41",
            "FG": "#e6e6e6",
            "SUBTLE_FG": "#cfcfcf",
            "ACCENT": "#4ea1ff",
            "LINK": "#61dafb",
            "BUBBLE_AI_BG": "#2a2a2a",
            "BUBBLE_AI_FG": "#f2f2f2",
            "BUBBLE_USER_BG": "#0e7ddc",
            "BUBBLE_USER_FG": "#ffffff",
            "CHIP_BG": "#133a5c",
            "CHIP_FG": "#d6ecff",
            "DIVIDER": "#2c2c2c",
        },
        "light": {
            "BG": "#f6f7fb",
            "PANEL_BG": "#eef0f5",
            "TEXT_BG": "#ffffff",
            "ENTRY_BG": "#ffffff",
            "BUTTON_BG": "#f0f2f7",
            "BUTTON_ACTIVE_BG": "#e6e9f2",
            "FG": "#1f1f1f",
            "SUBTLE_FG": "#3e3e3e",
            "ACCENT": "#006adc",
            "LINK": "#006adc",
            "BUBBLE_AI_BG": "#f1f5ff",
            "BUBBLE_AI_FG": "#0f172a",
            "BUBBLE_USER_BG": "#006adc",
            "BUBBLE_USER_FG": "#ffffff",
            "CHIP_BG": "#eaf3ff",
            "CHIP_FG": "#04366e",
            "DIVIDER": "#dde3ee",
        },
    }

    def __init__(self, root, model_name: str = "Chat Model"):
        self.root = root
        self.model_name = model_name
        self.theme_name = "dark"
        self.t = self.THEMES[self.theme_name]

        self.root.title("RAG Assistant - Local Chat")
        self.root.geometry("980x680")
        self.root.configure(bg=self.t["BG"])
        self.root.minsize(780, 520)

        # For unique link tags
        self._link_counter = 0

        # Track popups to avoid multiple windows per entry
        self.entry_windows = []

        # tk variables (for toggles)
        self.theme_var = tk.StringVar(value=self.theme_name)

        # --- ttk style ---
        try:
            style = ttk.Style()
            style.theme_use("clam")
            style.configure(
                "Dark.Horizontal.TProgressbar",
                troughcolor=self.t["BUTTON_BG"],
                bordercolor=self.t["BUTTON_BG"],
                background=self.t["ACCENT"],
                lightcolor=self.t["ACCENT"],
                darkcolor=self.t["ACCENT"],
            )
        except Exception:
            pass

        # --- Root layout (header | body) ---
        self._create_header()

        body = tk.Frame(self.root, bg=self.t["BG"])
        body.pack(fill=tk.BOTH, expand=True)

        # --- Left History Panel ---
        self.history_frame = tk.Frame(body, width=260, bg=self.t["PANEL_BG"], highlightthickness=1,
                                      highlightbackground=self.t["DIVIDER"])
        self.history_frame.pack(side=tk.LEFT, fill=tk.Y)

        hist_header = tk.Frame(self.history_frame, bg=self.t["PANEL_BG"])
        hist_header.pack(fill=tk.X, padx=10, pady=(10, 6))

        hist_label = tk.Label(hist_header, text="Chat History", font=("Segoe UI", 11, "bold"),
                              bg=self.t["PANEL_BG"], fg=self.t["FG"])
        hist_label.pack(side=tk.LEFT)

        self.history_text = scrolledtext.ScrolledText(
            self.history_frame, wrap=tk.WORD, width=32, bg=self.t["TEXT_BG"], fg=self.t["FG"],
            insertbackground=self.t["FG"], relief=tk.FLAT, font=("Segoe UI", 10)
        )
        self.history_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        self.history_text.config(state=tk.DISABLED)

        self.history_entries = []
        self.load_history_into_ui()

        # --- Right Chat Panel ---
        chat_side = tk.Frame(body, bg=self.t["BG"])
        chat_side.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Chat display
        self.chat_display = scrolledtext.ScrolledText(
            chat_side, wrap=tk.WORD, height=25, bg=self.t["TEXT_BG"], fg=self.t["FG"],
            insertbackground=self.t["FG"], relief=tk.FLAT, font=("Segoe UI", 11)
        )
        self.chat_display.pack(fill=tk.BOTH, padx=12, pady=(12, 8), expand=True)
        self.chat_display.config(state=tk.DISABLED)

        self._configure_text_tags()

        # Progress overlay (appears above input area)
        self.progress_frame = tk.Frame(chat_side, bg=self.t["BG"])
        self.progress_label = tk.Label(self.progress_frame, text="Thinking…",
                                       bg=self.t["BG"], fg=self.t["SUBTLE_FG"],
                                       font=("Segoe UI", 10))
        self.progress_label.pack(side=tk.LEFT, padx=6)
        self.progress = ttk.Progressbar(self.progress_frame, mode="indeterminate", length=220,
                                        style="Dark.Horizontal.TProgressbar")
        self.progress.pack(side=tk.LEFT, padx=6)

        # Input row
        input_row = tk.Frame(chat_side, bg=self.t["BG"])
        input_row.pack(fill=tk.X, padx=12, pady=(0, 12))

        self.input_field = tk.Entry(input_row, font=("Segoe UI", 11),
                                    bg=self.t["ENTRY_BG"], fg=self.t["FG"], insertbackground=self.t["FG"],
                                    relief=tk.FLAT)
        self.input_field.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8), ipady=8)
        self._add_placeholder(self.input_field, "Type your question…")

        # Buttons
        self.send_button = tk.Button(input_row, text="Send", command=self.send_message,
                                     bg=self.t["BUTTON_BG"], fg=self.t["FG"],
                                     activebackground=self.t["BUTTON_ACTIVE_BG"],
                                     activeforeground=self.t["FG"], relief=tk.FLAT, padx=16, pady=8)
        self.send_button.pack(side=tk.LEFT, padx=(0, 6))

        self.new_chat_button = tk.Button(input_row, text="New Chat", command=self.start_new_chat,
                                         bg=self.t["BUTTON_BG"], fg=self.t["FG"],
                                         activebackground=self.t["BUTTON_ACTIVE_BG"],
                                         activeforeground=self.t["FG"], relief=tk.FLAT, padx=12, pady=8)
        self.new_chat_button.pack(side=tk.LEFT)

        # Hover effects for buttons
        self._add_hover(self.send_button)
        self._add_hover(self.new_chat_button)

        # Bindings
        self.input_field.bind("<Return>", self.send_message)
        self.input_field.bind("<Control-Return>", self.send_message)  # alternative
        self.history_text.tag_bind("hist_click", "<Enter>", lambda e: self.history_text.config(cursor="hand2"))
        self.history_text.tag_bind("hist_click", "<Leave>", lambda e: self.history_text.config(cursor=""))

        self.input_field.focus_set()

    # ---- Header with model badge, toggle and actions ----
    def _create_header(self):
        header = tk.Frame(self.root, bg=self.t["PANEL_BG"], height=56, highlightthickness=1,
                          highlightbackground=self.t["DIVIDER"])
        header.pack(fill=tk.X, side=tk.TOP)

        title = tk.Label(header, text="RAG Assistant", font=("Segoe UI", 13, "bold"),
                         bg=self.t["PANEL_BG"], fg=self.t["FG"])
        title.pack(side=tk.LEFT, padx=12)

        model_badge = tk.Label(header, text=f"Model: {self.model_name}", font=("Segoe UI", 9),
                               bg=self.t["BUTTON_BG"], fg=self.t["SUBTLE_FG"], padx=10, pady=4)
        model_badge.pack(side=tk.LEFT, padx=(8, 0))

        spacer = tk.Frame(header, bg=self.t["PANEL_BG"])
        spacer.pack(side=tk.LEFT, expand=True, fill=tk.X)

        # Theme toggle
        toggle = tk.Button(header, text="Toggle Theme", command=self._toggle_theme,
                           bg=self.t["BUTTON_BG"], fg=self.t["FG"],
                           activebackground=self.t["BUTTON_ACTIVE_BG"],
                           activeforeground=self.t["FG"], relief=tk.FLAT, padx=10, pady=6)
        toggle.pack(side=tk.RIGHT, padx=(0, 8))
        self._add_hover(toggle)

        clear_btn = tk.Button(header, text="Clear History", command=self._clear_history_confirm,
                              bg=self.t["BUTTON_BG"], fg=self.t["FG"],
                              activebackground=self.t["BUTTON_ACTIVE_BG"],
                              activeforeground=self.t["FG"], relief=tk.FLAT, padx=10, pady=6)
        clear_btn.pack(side=tk.RIGHT, padx=(0, 8))
        self._add_hover(clear_btn)

        # Simple Help/about
        help_btn = tk.Button(header, text="Help", command=self._show_help,
                             bg=self.t["BUTTON_BG"], fg=self.t["FG"],
                             activebackground=self.t["BUTTON_ACTIVE_BG"],
                             activeforeground=self.t["FG"], relief=tk.FLAT, padx=10, pady=6)
        help_btn.pack(side=tk.RIGHT, padx=(0, 8))
        self._add_hover(help_btn)

    # ---- Text widget styling (chat bubbles, metadata, links) ----
    def _configure_text_tags(self):
        # General paragraph spacing
        self.chat_display.tag_config("space_before", spacing1=6)
        self.chat_display.tag_config("space_after", spacing3=10)

        # AI bubble (left)
        self.chat_display.tag_config(
            "ai_bubble",
            background=self.t["BUBBLE_AI_BG"],
            foreground=self.t["BUBBLE_AI_FG"],
            lmargin1=14, lmargin2=14, rmargin=140,  # left margin big, right margin small
            spacing1=6, spacing3=10,
            font=("Segoe UI", 11)
        )
        # User bubble (right-ish)
        self.chat_display.tag_config(
            "user_bubble",
            background=self.t["BUBBLE_USER_BG"],
            foreground=self.t["BUBBLE_USER_FG"],
            lmargin1=140, lmargin2=140, rmargin=14,  # right margin big
            spacing1=6, spacing3=10,
            font=("Segoe UI", 11, "bold")
        )

        # Small meta text (e.g., "Sources")
        self.chat_display.tag_config("meta", foreground=self.t["SUBTLE_FG"], font=("Segoe UI", 9), spacing1=4)

        # Source chip style
        self.chat_display.tag_config(
            "chip",
            background=self.t["CHIP_BG"],
            foreground=self.t["CHIP_FG"],
            lmargin1=34, lmargin2=34, rmargin=14,
            spacing1=2, spacing3=6,
            font=("Segoe UI", 9)
        )
        # Clickable area uses this tag name prefix: "src_link_..."

    # ---- Utility: Hover effect for flat buttons ----
    def _add_hover(self, btn):
        def on_enter(e): btn.config(bg=self.t["BUTTON_ACTIVE_BG"])
        def on_leave(e): btn.config(bg=self.t["BUTTON_BG"])
        btn.bind("<Enter>", on_enter)
        btn.bind("<Leave>", on_leave)

    # ---- Placeholder in Entry ----
    def _add_placeholder(self, entry, text):
        def on_focus_in(e):
            if entry.get() == text:
                entry.delete(0, tk.END)
                entry.config(fg=self.t["FG"])
        def on_focus_out(e):
            if not entry.get():
                entry.insert(0, text)
                entry.config(fg=self.t["SUBTLE_FG"])
        entry.insert(0, text)
        entry.config(fg=self.t["SUBTLE_FG"])
        entry.bind("<FocusIn>", on_focus_in)
        entry.bind("<FocusOut>", on_focus_out)

    # --- Helpers for history preview formatting ---
    def _make_preview_lines(self, entry_text: str):
        lines = entry_text.splitlines()
        if len(lines) > self.PREVIEW_LINES:
            preview = lines[: self.PREVIEW_LINES - 1] + ["… (click to expand)"]
        else:
            preview = lines + [""] * (self.PREVIEW_LINES - len(lines))
        return preview

    # --- Load existing history into sidebar ---
    def load_history_into_ui(self):
        entries = [e.strip() for e in load_history() if e.strip()]
        self.history_entries = entries

        self.history_text.config(state=tk.NORMAL)
        self.history_text.delete("1.0", tk.END)

        for i, entry in enumerate(entries):
            preview_lines = self._make_preview_lines(entry)
            preview = "\n".join(preview_lines)

            start_idx = self.history_text.index(tk.END)
            spacing = "\n" * self.PREVIEW_SPACING_BLANK_LINES
            self.history_text.insert(tk.END, preview + spacing)

            end_idx = self.history_text.index(tk.END)
            tag = f"hist_item_{i}"

            self.history_text.tag_add(tag, start_idx, end_idx)
            self.history_text.tag_config(tag, foreground=self.t["ACCENT"])
            self.history_text.tag_add("hist_click", start_idx, end_idx)
            self.history_text.tag_bind(tag, "<Button-1>", lambda e, idx=i: self._on_history_click(e, idx))

        self.history_text.config(state=tk.DISABLED)

    def _on_history_click(self, event, idx: int):
        self.open_history_entry(idx)
        return "break"

    # --- Open full history entry in a popup window (single per entry) ---
    def open_history_entry(self, idx: int):
        if idx < 0 or idx >= len(self.history_entries):
            return

        full_text = self.history_entries[idx]

        win = tk.Toplevel(self.root)
        win.title(f"Chat Entry #{idx + 1}")
        win.geometry("700x500")
        win.configure(bg=self.t["BG"])

        header = tk.Frame(win, bg=self.t["BG"])
        header.pack(fill=tk.X, padx=8, pady=6)

        def copy_to_clipboard():
            try:
                win.clipboard_clear()
                win.clipboard_append(full_text)
            except Exception:
                pass

        copy_btn = tk.Button(header, text="Copy", command=copy_to_clipboard,
                             bg=self.t["BUTTON_BG"], fg=self.t["FG"],
                             activebackground=self.t["BUTTON_ACTIVE_BG"],
                             activeforeground=self.t["FG"], relief=tk.FLAT)
        copy_btn.pack(side=tk.RIGHT)

        txt = scrolledtext.ScrolledText(win, wrap=tk.WORD, bg=self.t["TEXT_BG"], fg=self.t["FG"],
                                        insertbackground=self.t["FG"], relief=tk.FLAT, font=("Segoe UI", 10))
        txt.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))
        txt.insert(tk.END, full_text)
        txt.config(state=tk.DISABLED)

    # --- Clear chat area ---
    def start_new_chat(self):
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.delete("1.0", tk.END)
        self.chat_display.config(state=tk.DISABLED)
        self.input_field.delete(0, tk.END)

    # --- Send message and get answer (non-blocking) ---
    def send_message(self, event=None):
        query = self.input_field.get().strip()
        # Avoid sending the placeholder
        if not query or query.lower().startswith("type your question"):
            return

        self.insert_bubble("You", query)
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
            self.root.after(0, lambda: self._finish_query(query, response, sources))

    def _finish_query(self, query, response, sources):
        try:
            self.insert_bubble("AI", response)
            if sources:
                self._insert_sources(sources)

            save_entry(query, response, sources)
            self.update_history()
        except Exception as ex:
            messagebox.showerror("Render error", f"Failed to render response/sources:\n{ex}")
        finally:
            self._stop_progress()

    # --- Insert chat bubble with styling ---
    def insert_bubble(self, sender: str, text: str):
        self.chat_display.config(state=tk.NORMAL)

        tag = "user_bubble" if sender.lower().startswith("you") else "ai_bubble"
        # Split long messages to avoid overly wide rectangles
        wrapped = self._wrap_text(text, self.BUBBLE_MAX_WIDTH_CHARS)

        self.chat_display.insert(tk.END, wrapped + "\n", (tag,))
        self.chat_display.insert(tk.END, "\n", ("space_after",))
        self.chat_display.config(state=tk.DISABLED)
        self.chat_display.see(tk.END)

    def _wrap_text(self, text, width):
        import textwrap
        # Keeps paragraphs intact
        parts = []
        for para in text.splitlines():
            if not para.strip():
                parts.append("")
            else:
                parts.extend(textwrap.wrap(para, width=width, replace_whitespace=False,
                                           drop_whitespace=False))
        return "\n".join(parts)

    # --- Insert source chips under the last AI message ---
    def _insert_sources(self, sources):
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.insert(tk.END, "Sources:\n", ("meta",))

        for src in sources:
            # Show shorter text
            label = self._shorten_source(src)
            start_idx = self.chat_display.index(tk.END)
            chip_text = f"  ↗ {label}\n"
            self.chat_display.insert(tk.END, chip_text, ("chip",))

            end_idx = self.chat_display.index(tk.END)
            tag = f"src_link_{self._link_counter}"
            self._link_counter += 1
            self.chat_display.tag_add(tag, start_idx, end_idx)
            self.chat_display.tag_bind(tag, "<Button-1>", lambda e, url=src: self._open_link(url))
            self.chat_display.tag_bind(tag, "<Enter>", lambda e: self.chat_display.config(cursor="hand2"))
            self.chat_display.tag_bind(tag, "<Leave>", lambda e: self.chat_display.config(cursor=""))

        self.chat_display.insert(tk.END, "\n", ("space_after",))
        self.chat_display.config(state=tk.DISABLED)
        self.chat_display.see(tk.END)

    def _shorten_source(self, src: str, max_len: int = 68):
        import os
        if src.startswith("http"):
            s = src.split("//", 1)[-1]
        else:
            s = os.path.basename(src)
        return (s[:max_len - 1] + "…") if len(s) > max_len else s

    # --- Open links or show info for non-HTTP paths ---
    def _open_link(self, url):
        if url.startswith("http://") or url.startswith("https://"):
            webbrowser.open(url)
        else:
            messagebox.showinfo("Local Source", f"Local file or non-HTTP path:\n{url}")

    # --- Update sidebar history ---
    def update_history(self):
        self.load_history_into_ui()

    # --- Progress helpers ---
    def _start_progress(self):
        try:
            self.progress_frame.pack(fill=tk.X, padx=12, pady=(0, 8))
            self.progress.start(12)
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

    # --- Theme switching ---
    def _toggle_theme(self):
        self.theme_name = "light" if self.theme_name == "dark" else "dark"
        self.t = self.THEMES[self.theme_name]
        self._apply_theme()

    def _apply_theme(self):
        # Repaint all relevant widgets
        self.root.configure(bg=self.t["BG"])

        # Header is the first child
        header = self.root.winfo_children()[0]
        header.configure(bg=self.t["PANEL_BG"], highlightbackground=self.t["DIVIDER"])
        for w in header.winfo_children():
            if isinstance(w, tk.Frame):
                w.configure(bg=self.t["PANEL_BG"])
            elif isinstance(w, tk.Button):
                w.configure(bg=self.t["BUTTON_BG"], fg=self.t["FG"],
                            activebackground=self.t["BUTTON_ACTIVE_BG"],
                            activeforeground=self.t["FG"])
            elif isinstance(w, tk.Label):
                # Model badge and title
                if "Model:" in getattr(w, "cget", lambda x: "")("text"):
                    w.configure(bg=self.t["BUTTON_BG"], fg=self.t["SUBTLE_FG"])
                else:
                    w.configure(bg=self.t["PANEL_BG"], fg=self.t["FG"])

        # Body
        body = self.root.winfo_children()[1]
        body.configure(bg=self.t["BG"])

        # History
        self.history_frame.configure(bg=self.t["PANEL_BG"], highlightbackground=self.t["DIVIDER"])
        for child in self.history_frame.winfo_children():
            if isinstance(child, tk.Frame):
                child.configure(bg=self.t["PANEL_BG"])
                for cc in child.winfo_children():
                    if isinstance(cc, tk.Label):
                        cc.configure(bg=self.t["PANEL_BG"], fg=self.t["FG"])

        self.history_text.configure(bg=self.t["TEXT_BG"], fg=self.t["FG"], insertbackground=self.t["FG"])

        # Chat side
        # It's the other child of body
        chat_side = body.winfo_children()[1]
        chat_side.configure(bg=self.t["BG"])

        self.chat_display.configure(bg=self.t["TEXT_BG"], fg=self.t["FG"], insertbackground=self.t["FG"])

        self.progress_frame.configure(bg=self.t["BG"])
        self.progress_label.configure(bg=self.t["BG"], fg=self.t["SUBTLE_FG"])

        # Input row is last child
        input_row = chat_side.winfo_children()[-1]
        input_row.configure(bg=self.t["BG"])

        self.input_field.configure(bg=self.t["ENTRY_BG"], fg=self.t["FG"], insertbackground=self.t["FG"])

        for btn in [self.send_button, self.new_chat_button]:
            btn.configure(bg=self.t["BUTTON_BG"], fg=self.t["FG"],
                          activebackground=self.t["BUTTON_ACTIVE_BG"],
                          activeforeground=self.t["FG"])

        # Update styles for bubbles and chips
        self._configure_text_tags()
        # Redraw history to apply accent color to tags
        self.update_history()

    # --- Clear history ---
    def _clear_history_confirm(self):
        if messagebox.askyesno("Clear History", "This will delete the saved chat_history.txt. Continue?"):
            try:
                if os.path.exists(HISTORY_FILE):
                    os.remove(HISTORY_FILE)
                self.update_history()
                messagebox.showinfo("Cleared", "Chat history has been cleared.")
            except Exception as e:
                messagebox.showerror("Error", f"Could not clear history:\n{e}")

    # --- Simple help dialog ---
    def _show_help(self):
        message = (
            "Tips:\n"
            "• Use Enter or Ctrl+Enter to send.\n"
            "• Click links under Sources to open them.\n"
            "• Toggle theme from the header.\n"
            "• Click a history item to view its full content."
        )
        messagebox.showinfo("Help", message)


# ========= MAIN ==========
if __name__ == "__main__":
    root = tk.Tk()
    # If you want the model badge to show the active LLM name:
    try:
        active_model = getattr(llm, "model_name", "Chat Model")
    except NameError:
        active_model = "Chat Model"
    app = RAGApp(root, model_name=active_model)
    root.mainloop()