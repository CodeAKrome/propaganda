#!/usr/bin/env python3
"""
MongoDB Pager TUI - Browse RSS News articles with keyboard navigation
Requires: pip install textual pymongo
"""

import os
import json
from textual.app import App, ComposeResult
from textual.widgets import DataTable, Static, Footer, Header, Button, Input
from textual.containers import Horizontal, Vertical, Container, ScrollableContainer
from textual.reactive import reactive
from textual.binding import Binding
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from bson import json_util
from datetime import datetime

# Configuration
MONGO_URI = os.getenv("MONGO_URI", "mongodb://root:example@localhost:27017")
MONGO_DB = "rssnews"
MONGO_COLL = "articles"
PAGE_SIZE = 10


class DocumentView(Static):
    """Full-screen document viewer overlay"""
    
    DEFAULT_CSS = """
    DocumentView {
        layer: overlay;
        width: 100%;
        height: 100%;
        background: $surface 95%;
        padding: 2 4;
        visibility: hidden;
    }
    
    DocumentView.visible {
        visibility: visible;
    }
    
    #doc-scroll {
        height: 90%;
        border: solid $primary;
        background: $surface-darken-1;
        padding: 1 2;
    }
    
    #doc-content {
        text-style: bold;
        width: auto;
        height: auto;
    }
    
    #doc-controls {
        height: auto;
        margin-top: 1;
        content-align: center middle;
    }
    
    .doc-label {
        text-style: bold;
        color: $primary;
        margin-bottom: 1;
    }
    """

    def compose(self) -> ComposeResult:
        yield Static("📄 Document Details (Press Enter or Esc to close)", classes="doc-label")
        with ScrollableContainer(id="doc-scroll"):
            yield Static(id="doc-content")
        with Horizontal(id="doc-controls"):
            yield Button("Close (Esc/Enter)", id="close-doc", variant="primary")
    
    def show(self, doc: dict) -> None:
        """Display the full document with all fields"""
        # Pretty print JSON with indentation
        json_str = json_util.dumps(doc, indent=2, default=str)
        content = self.query_one("#doc-content", Static)
        content.update(f"```json\n{json_str}\n```")
        self.add_class("visible")
        self.query_one("#close-doc", Button).focus()
    
    def hide(self) -> None:
        """Hide the document view"""
        self.remove_class("visible")
        # Return focus to table
        table = self.app.query_one("#article-table", DataTable)
        table.focus()
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "close-doc":
            self.hide()
    
    def on_key(self, event) -> None:
        """Close on Escape or Enter"""
        if event.key in ("escape", "enter"):
            self.hide()


class MongoPager(App):
    """Main TUI Application"""
    
    CSS = """
    Screen { align: center middle; }
    
    #main-container { 
        width: 100%; 
        height: 100%; 
        layout: vertical;
    }
    
    #table-container { 
        height: 1fr; 
        width: 100%;
        border: solid $primary;
    }
    
    #controls { 
        height: auto; 
        width: 100%;
        background: $surface-darken-1;
        padding: 1 2;
        dock: bottom;
    }
    
    DataTable { 
        height: 100%; 
        width: 100%;
    }
    
    #page-info {
        content-align: center middle;
        width: 30;
    }
    
    #filter-input {
        width: 40;
    }
    
    .instruction {
        color: $text-muted;
        text-style: italic;
    }
    """
    
    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("r", "refresh", "Refresh"),
        Binding("n", "next_page", "Next Page"),
        Binding("p", "prev_page", "Prev Page"),
        Binding("enter", "view_doc", "View Record", show=True),
        Binding("escape", "close_view", "Close View", show=False),
    ]
    
    page = reactive(0)
    total_docs = reactive(0)
    filter_query = reactive({})
    
    def __init__(self):
        super().__init__()
        self.client = None
        self.db = None
        self.collection = None
        self.current_docs = []
        self.doc_view = None
        
    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        
        with Vertical(id="main-container"):
            with Container(id="table-container"):
                yield DataTable(id="article-table")
            
            with Horizontal(id="controls"):
                yield Button("◀ Prev", id="prev", variant="primary")
                yield Button("Next ▶", id="next", variant="primary")
                yield Static(id="page-info")
                yield Input(placeholder='Filter: {"source": "BBC"}', id="filter-input")
                yield Button("Apply", id="filter-btn")
                yield Static("Enter: View | N/P: Page | R: Refresh | Q: Quit", classes="instruction")
        
        # Document viewer overlay
        self.doc_view = DocumentView(id="doc-viewer")
        yield self.doc_view
        
        yield Footer()
    
    def on_mount(self) -> None:
        """Initialize connection and load first page"""
        self.connect_db()
        self.setup_table()
        self.load_data()
        # Focus table on startup
        self.query_one("#article-table", DataTable).focus()
    
    def connect_db(self):
        """Establish MongoDB connection"""
        try:
            self.client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
            self.client.admin.command('ping')
            self.db = self.client[MONGO_DB]
            self.collection = self.db[MONGO_COLL]
            self.notify(f"Connected to {MONGO_DB}.{MONGO_COLL}", timeout=3)
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            self.notify(f"Connection failed: {e}", severity="error", timeout=10)
    
    def setup_table(self):
        """Configure DataTable columns"""
        table = self.query_one("#article-table", DataTable)
        table.add_columns("ID", "Title", "Source", "Published", "Added")
        table.cursor_type = "row"
        table.zebra_stripes = True
        table.show_cursor = True
    
    def load_data(self):
        """Load current page from MongoDB"""
        if self.collection is None:
            return
        
        try:
            self.total_docs = self.collection.count_documents(self.filter_query)
            skip = self.page * PAGE_SIZE
            cursor = self.collection.find(self.filter_query)\
                .sort("published", -1)\
                .skip(skip)\
                .limit(PAGE_SIZE)
            
            self.current_docs = list(cursor)
            self.update_table()
            self.update_pagination()
            
        except Exception as e:
            self.notify(f"Query error: {e}", severity="error")
    
    def update_table(self):
        """Refresh table with current documents"""
        table = self.query_one("#article-table", DataTable)
        table.clear()
        
        for doc in self.current_docs:
            doc_id = str(doc.get('_id', 'N/A'))[:8]
            title = str(doc.get('title', 'N/A'))[:45]
            source = str(doc.get('source', 'N/A'))[:15]
            pub_date = self.format_date(doc.get('published'))
            added_date = self.format_date(doc.get('added'))
            
            table.add_row(doc_id, title, source, pub_date, added_date)
        
        # Restore cursor if docs exist
        if self.current_docs:
            table.move_cursor(row=0)
    
    def format_date(self, date_val) -> str:
        """Format date fields for display"""
        if date_val is None:
            return "N/A"
        if isinstance(date_val, datetime):
            return date_val.strftime("%m/%d %H:%M")
        return str(date_val)[:16]
    
    def update_pagination(self):
        """Update page counter and button states"""
        total_pages = max(1, (self.total_docs + PAGE_SIZE - 1) // PAGE_SIZE)
        current = min(self.page + 1, total_pages)
        
        info = f"{current}/{total_pages} ({self.total_docs})"
        self.query_one("#page-info", Static).update(info)
        
        self.query_one("#prev", Button).disabled = self.page == 0
        self.query_one("#next", Button).disabled = (self.page + 1) * PAGE_SIZE >= self.total_docs
    
    # Actions
    def action_next_page(self):
        if (self.page + 1) * PAGE_SIZE < self.total_docs:
            self.page += 1
            self.load_data()
    
    def action_prev_page(self):
        if self.page > 0:
            self.page -= 1
            self.load_data()
    
    def action_refresh(self):
        self.load_data()
        self.notify("Refreshed", timeout=2)
    
    def action_view_doc(self):
        """Show full document details when Enter is pressed"""
        # If doc view is open, close it (toggle behavior)
        if self.doc_view and self.doc_view.has_class("visible"):
            self.doc_view.hide()
            return
            
        table = self.query_one("#article-table", DataTable)
        cursor_row = table.cursor_row
        
        if cursor_row is None or cursor_row < 0 or cursor_row >= len(self.current_docs):
            self.notify("No record selected", severity="warning")
            return
        
        doc = self.current_docs[cursor_row]
        self.doc_view.show(doc)
    
    def action_close_view(self):
        """Close document view"""
        if self.doc_view and self.doc_view.has_class("visible"):
            self.doc_view.hide()
    
    # Event handlers
    def on_button_pressed(self, event: Button.Pressed) -> None:
        btn_id = event.button.id
        if btn_id == "next":
            self.action_next_page()
        elif btn_id == "prev":
            self.action_prev_page()
        elif btn_id == "filter-btn":
            self.apply_filter()
    
    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "filter-input":
            self.apply_filter()
    
    def apply_filter(self):
        """Apply JSON filter from input"""
        input_widget = self.query_one("#filter-input", Input)
        filter_text = input_widget.value.strip()
        
        if not filter_text:
            self.filter_query = {}
        else:
            try:
                self.filter_query = json.loads(filter_text)
            except json.JSONDecodeError as e:
                self.notify(f"Invalid JSON: {e}", severity="error")
                return
        
        self.page = 0
        self.load_data()
        if filter_text:
            self.notify(f"Filter applied", timeout=2)


if __name__ == "__main__":
    app = MongoPager()
    app.run()
