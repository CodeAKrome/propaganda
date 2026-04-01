#!/usr/bin/env python3
"""
MongoDB Pager TUI - Browse RSS News articles with keyboard navigation
Requires: pip install textual pymongo
"""

import os
import json
import textwrap
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
        padding: 1 2;
        visibility: hidden;
    }
    
    DocumentView.visible {
        visibility: visible;
    }
    
    #doc-scroll {
        height: 88%;
        border: solid $primary;
        background: $surface-darken-1;
        padding: 0 1;
    }
    
    #doc-content {
        width: 100%;
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
        text-align: center;
    }
    """
    
    BINDINGS = [
        Binding("escape", "close", "Close"),
        Binding("up", "scroll_up", "Scroll Up", show=False),
        Binding("down", "scroll_down", "Scroll Down", show=False),
        Binding("pageup", "page_up", "Page Up", show=False),
        Binding("pagedown", "page_down", "Page Down", show=False),
        Binding("p", "prev_doc", "Prev Doc", show=False),
        Binding("n", "next_doc", "Next Doc", show=False),
    ]
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_doc_index = 0
        self.docs_list = []
        self.app_ref = None
    
    def compose(self) -> ComposeResult:
        yield Static("📄 ↑↓:Scroll | PgUp/PgDn:Page | P/N:Prev/Next Doc | Esc:Close", classes="doc-label")
        with ScrollableContainer(id="doc-scroll"):
            yield Static(id="doc-content", expand=True)
        with Horizontal(id="doc-controls"):
            yield Button("◀ Prev (p)", id="prev-doc", variant="primary")
            yield Button("Next (n) ▶", id="next-doc", variant="primary")
            yield Button("Close (Esc)", id="close-doc", variant="error")
    
    def set_docs(self, docs: list, app_ref):
        """Set the list of documents and app reference for navigation"""
        self.docs_list = docs
        self.app_ref = app_ref
    
    def show(self, index: int) -> None:
        """Display document at index with word wrapping"""
        if not self.docs_list or index < 0 or index >= len(self.docs_list):
            return
        
        self.current_doc_index = index
        doc = self.docs_list[index]
        
        # Format document with wrapping
        json_str = json_util.dumps(doc, indent=2, default=str)
        # Wrap lines to container width (subtract padding/borders)
        wrapped_lines = []
        for line in json_str.split('\n'):
            if len(line) > 100:
                # Use textwrap for long lines
                wrapped = textwrap.wrap(line, width=100, subsequent_indent="  ")
                wrapped_lines.extend(wrapped)
            else:
                wrapped_lines.append(line)
        
        content = self.query_one("#doc-content", Static)
        content.update("\n".join(wrapped_lines))
        
        self.add_class("visible")
        # Update button states
        self.query_one("#prev-doc", Button).disabled = (self.current_doc_index == 0)
        self.query_one("#next-doc", Button).disabled = (self.current_doc_index >= len(self.docs_list) - 1)
        
        # Focus scroll container for navigation
        scroll = self.query_one("#doc-scroll", ScrollableContainer)
        scroll.scroll_home()  # Scroll to top
        scroll.focus()
    
    def hide(self) -> None:
        """Hide the document view"""
        self.remove_class("visible")
        if self.app_ref:
            table = self.app_ref.query_one("#article-table", DataTable)
            table.focus()
    
    def action_close(self) -> None:
        """Close the document view"""
        self.hide()
    
    def action_scroll_up(self) -> None:
        """Scroll up one line"""
        scroll = self.query_one("#doc-scroll", ScrollableContainer)
        scroll.scroll_up()
    
    def action_scroll_down(self) -> None:
        """Scroll down one line"""
        scroll = self.query_one("#doc-scroll", ScrollableContainer)
        scroll.scroll_down()
    
    def action_page_up(self) -> None:
        """Scroll up one page"""
        scroll = self.query_one("#doc-scroll", ScrollableContainer)
        scroll.scroll_page_up()
    
    def action_page_down(self) -> None:
        """Scroll down one page"""
        scroll = self.query_one("#doc-scroll", ScrollableContainer)
        scroll.scroll_page_down()
    
    def action_prev_doc(self) -> None:
        """Go to previous document"""
        if self.current_doc_index > 0:
            self.show(self.current_doc_index - 1)
    
    def action_next_doc(self) -> None:
        """Go to next document"""
        if self.current_doc_index < len(self.docs_list) - 1:
            self.show(self.current_doc_index + 1)
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "close-doc":
            self.hide()
        elif event.button.id == "prev-doc":
            self.action_prev_doc()
        elif event.button.id == "next-doc":
            self.action_next_doc()
    
    def on_key(self, event) -> None:
        """Handle keys when document view is focused"""
        if event.key == "escape":
            self.hide()
            event.stop()


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
        Binding("enter", "view_doc", "View", show=True),
        Binding("escape", "close_view", "Back", show=True),
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
        
    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        
        with Vertical(id="main-container"):
            with Container(id="table-container"):
                yield DataTable(id="article-table")
            
            with Horizontal(id="controls"):
                yield Button("◀ Prev (p)", id="prev", variant="primary")
                yield Button("Next (n) ▶", id="next", variant="primary")
                yield Static(id="page-info")
                yield Input(placeholder='Filter: {"source": "BBC"}', id="filter-input")
                yield Button("Apply", id="filter-btn", variant="success")
                yield Static("Enter: View | Esc: Back | N/P: Page | R: Refresh | Q: Quit", classes="instruction")
        
        yield DocumentView(id="doc-viewer")
        yield Footer()
    
    def on_mount(self) -> None:
        """Initialize connection and load first page"""
        self.connect_db()
        self.setup_table()
        self.load_data()
        # Pass docs reference to document view
        doc_view = self.query_one("#doc-viewer", DocumentView)
        doc_view.set_docs(self.current_docs, self)
        table = self.query_one("#article-table", DataTable)
        table.focus()
    
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
            
            # Update document view's reference
            doc_view = self.query_one("#doc-viewer", DocumentView)
            doc_view.set_docs(self.current_docs, self)
            
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
    
    def check_view_visible(self) -> bool:
        """Check if document view is currently visible"""
        doc_view = self.query_one("#doc-viewer", DocumentView)
        return doc_view.has_class("visible")
    
    def action_next_page(self):
        if self.check_view_visible():
            return
        if (self.page + 1) * PAGE_SIZE < self.total_docs:
            self.page += 1
            self.load_data()
    
    def action_prev_page(self):
        if self.check_view_visible():
            return
        if self.page > 0:
            self.page -= 1
            self.load_data()
    
    def action_refresh(self):
        if self.check_view_visible():
            return
        self.load_data()
        self.notify("Refreshed", timeout=2)
    
    def action_view_doc(self):
        """Show full document details when Enter is pressed"""
        if self.check_view_visible():
            return
            
        table = self.query_one("#article-table", DataTable)
        cursor_row = table.cursor_row
        
        if cursor_row is None or cursor_row < 0 or cursor_row >= len(self.current_docs):
            self.notify("No record selected", severity="warning")
            return
        
        doc_view = self.query_one("#doc-viewer", DocumentView)
        doc_view.show(cursor_row)
    
    def action_close_view(self):
        """Close document view if open"""
        doc_view = self.query_one("#doc-viewer", DocumentView)
        if doc_view.has_class("visible"):
            doc_view.hide()
        else:
            self.action_prev_page()
    
    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Handle row selection with Enter key"""
        if not self.check_view_visible():
            self.action_view_doc()
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        btn_id = event.button.id
        if btn_id == "next":
            self.action_next_page()
        elif btn_id == "prev":
            self.action_prev_page()
        elif btn_id == "filter-btn":
            if not self.check_view_visible():
                self.apply_filter()
    
    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "filter-input" and not self.check_view_visible():
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
