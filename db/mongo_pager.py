#!/usr/bin/env python3
"""
MongoDB Pager TUI - Browse RSS News articles with keyboard navigation
Requires: pip install textual pymongo
"""

import os
import json
import textwrap
import shutil
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
PAGE_SIZE = 25  # Increased default page size


class DocumentView(Static):
    """Full-screen document viewer overlay"""
    
    DEFAULT_CSS = """
    DocumentView {
        layer: overlay;
        width: 100%;
        height: 100%;
        background: $surface 98%;
        padding: 0 1;
        visibility: hidden;
    }
    
    DocumentView.visible {
        visibility: visible;
    }
    
    #doc-scroll {
        height: 1fr;
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
        padding: 1;
        content-align: center middle;
    }
    
    .doc-label {
        text-style: bold;
        color: $primary;
        height: auto;
        padding: 1;
        text-align: center;
    }
    """
    
    BINDINGS = [
        Binding("escape", "close", "Close"),
        Binding("up", "scroll_up", "Scroll Up", show=False),
        Binding("down", "scroll_down", "Scroll Down", show=False),
        Binding("pageup", "page_up", "Page Up", show=False),
        Binding("pagedown", "page_down", "Page Down", show=False),
        Binding("p", "prev_doc_or_page", "Prev", show=False),
        Binding("n", "next_doc_or_page", "Next", show=False),
    ]
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_doc_index = 0
        self.docs_list = []
        self.app_ref = None
        self.term_width = 80
    
    def compose(self) -> ComposeResult:
        yield Static("📄 ↑↓:Scroll | PgUp/PgDn:Page | P/N:Prev/Next | Esc:Close", classes="doc-label")
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
    
    def update_terminal_size(self):
        """Update terminal size for wrapping calculations"""
        try:
            self.term_width = shutil.get_terminal_size().columns - 4  # Account for padding/borders
        except:
            self.term_width = 100
    
    def show(self, index: int) -> None:
        """Display document at index with dynamic word wrapping"""
        if not self.docs_list or index < 0 or index >= len(self.docs_list):
            return
        
        self.update_terminal_size()
        self.current_doc_index = index
        doc = self.docs_list[index]
        
        # Format document with dynamic wrapping based on terminal width
        json_str = json_util.dumps(doc, indent=2, default=str)
        wrapped_lines = []
        
        for line in json_str.split('\n'):
            if len(line) > self.term_width - 2:
                # Wrap long lines dynamically
                wrapped = textwrap.wrap(
                    line, 
                    width=self.term_width - 2, 
                    subsequent_indent="  ",
                    break_long_words=False,
                    replace_whitespace=False
                )
                wrapped_lines.extend(wrapped)
            else:
                wrapped_lines.append(line)
        
        content = self.query_one("#doc-content", Static)
        content.update("\n".join(wrapped_lines))
        
        self.add_class("visible")
        self._update_button_states()
        
        # Focus scroll container for navigation
        scroll = self.query_one("#doc-scroll", ScrollableContainer)
        scroll.scroll_home()
        scroll.focus()
    
    def _update_button_states(self):
        """Update navigation button states based on current position"""
        can_go_prev = self.current_doc_index > 0 or (self.app_ref and self.app_ref.page > 0)
        can_go_next = (self.current_doc_index < len(self.docs_list) - 1) or \
                      (self.app_ref and (self.app_ref.page + 1) * PAGE_SIZE < self.app_ref.total_docs)
        
        self.query_one("#prev-doc", Button).disabled = not can_go_prev
        self.query_one("#next-doc", Button).disabled = not can_go_next
    
    def hide(self) -> None:
        """Hide the document view"""
        self.remove_class("visible")
        if self.app_ref:
            table = self.app_ref.query_one("#article-table", DataTable)
            if self.current_doc_index < len(self.docs_list):
                table.move_cursor(row=self.current_doc_index)
            table.focus()
    
    def action_close(self) -> None:
        """Close the document view"""
        self.hide()
    
    def action_scroll_up(self) -> None:
        scroll = self.query_one("#doc-scroll", ScrollableContainer)
        scroll.scroll_up()
    
    def action_scroll_down(self) -> None:
        scroll = self.query_one("#doc-scroll", ScrollableContainer)
        scroll.scroll_down()
    
    def action_page_up(self) -> None:
        scroll = self.query_one("#doc-scroll", ScrollableContainer)
        scroll.scroll_page_up()
    
    def action_page_down(self) -> None:
        scroll = self.query_one("#doc-scroll", ScrollableContainer)
        scroll.scroll_page_down()
    
    def action_prev_doc_or_page(self) -> None:
        """Go to previous document, or previous page if at first document"""
        if self.current_doc_index > 0:
            self.show(self.current_doc_index - 1)
        elif self.app_ref and self.app_ref.page > 0:
            self.app_ref.page -= 1
            self.app_ref.load_data()
            if self.docs_list:
                self.show(len(self.docs_list) - 1)
    
    def action_next_doc_or_page(self) -> None:
        """Go to next document, or next page if at last document"""
        if self.current_doc_index < len(self.docs_list) - 1:
            self.show(self.current_doc_index + 1)
        elif self.app_ref and (self.app_ref.page + 1) * PAGE_SIZE < self.app_ref.total_docs:
            self.app_ref.page += 1
            self.app_ref.load_data()
            if self.docs_list:
                self.show(0)
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "close-doc":
            self.hide()
        elif event.button.id == "prev-doc":
            self.action_prev_doc_or_page()
        elif event.button.id == "next-doc":
            self.action_next_doc_or_page()
    
    def on_key(self, event) -> None:
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
        min-height: 10;
    }
    
    #controls { 
        height: auto; 
        width: 100%;
        background: $surface-darken-1;
        padding: 1 2;
    }
    
    DataTable { 
        height: 100%; 
        width: 100%;
    }
    
    #page-info {
        content-align: center middle;
        width: auto;
        min-width: 15;
    }
    
    #filter-input {
        width: 1fr;
        min-width: 20;
        max-width: 50;
    }
    
    .instruction {
        color: $text-muted;
        text-style: italic;
        width: auto;
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
        self.term_height = 24
        
    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        
        with Vertical(id="main-container"):
            with Container(id="table-container"):
                yield DataTable(id="article-table")
            
            with Horizontal(id="controls"):
                yield Button("◀ Prev", id="prev", variant="primary")
                yield Button("Next ▶", id="next", variant="primary")
                yield Static(id="page-info")
                yield Input(placeholder='Filter: {"source": "BBC"}...', id="filter-input")
                yield Button("Apply", id="filter-btn", variant="success")
                yield Static("Enter:View | P/N:Page | R:Refresh | Q:Quit", classes="instruction")
        
        yield DocumentView(id="doc-viewer")
        yield Footer()
    
    def on_mount(self) -> None:
        """Initialize connection and load first page"""
        self.update_terminal_size()
        self.connect_db()
        self.setup_table()
        self.load_data()
        doc_view = self.query_one("#doc-viewer", DocumentView)
        doc_view.set_docs(self.current_docs, self)
        table = self.query_one("#article-table", DataTable)
        table.focus()
    
    def update_terminal_size(self):
        """Update terminal dimensions for dynamic sizing"""
        try:
            size = shutil.get_terminal_size()
            self.term_height = size.lines
            global PAGE_SIZE
            # Calculate optimal page size based on terminal height
            # Reserve 8 lines for header, footer, controls, and margins
            available_lines = max(10, size.lines - 8)
            PAGE_SIZE = available_lines
        except:
            pass
    
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
        """Configure DataTable with dynamic column widths"""
        table = self.query_one("#article-table", DataTable)
        table.add_columns("ID", "Title", "Source", "Published", "Added")
        table.cursor_type = "row"
        table.zebra_stripes = True
        table.show_cursor = True
    
    def _update_column_widths(self):
        """Update table column widths based on terminal size"""
        try:
            width = shutil.get_terminal_size().columns
        except:
            width = 80
        
        table = self.query_one("#article-table", DataTable)
        
        # Calculate proportional widths
        if width < 80:
            # Compact mode for small terminals
            ratios = [8, width-35, 12, 10, 10]
        else:
            # Wide mode - give more space to title
            ratios = [10, int(width*0.5), 15, 12, 12]
        
        # Apply widths if columns exist
        if len(table.columns) == 5:
            for i, ratio in enumerate(ratios):
                table.columns[i].width = ratio
    
    def load_data(self):
        """Load current page from MongoDB with dynamic page size"""
        if self.collection is None:
            return
        
        try:
            self.update_terminal_size()  # Recalculate page size on each load
            self.total_docs = self.collection.count_documents(self.filter_query)
            skip = self.page * PAGE_SIZE
            cursor = self.collection.find(self.filter_query)\
                .sort("published", -1)\
                .skip(skip)\
                .limit(PAGE_SIZE)
            
            self.current_docs = list(cursor)
            
            doc_view = self.query_one("#doc-viewer", DocumentView)
            doc_view.set_docs(self.current_docs, self)
            
            self.update_table()
            self.update_pagination()
            self._update_column_widths()
            
        except Exception as e:
            self.notify(f"Query error: {e}", severity="error")
    
    def update_table(self):
        """Refresh table with current documents - dynamically truncate fields"""
        table = self.query_one("#article-table", DataTable)
        table.clear()
        
        # Get available width for dynamic truncation
        try:
            term_width = shutil.get_terminal_size().columns
        except:
            term_width = 80
        
        # Calculate title width based on terminal
        title_width = max(20, term_width - 40)
        
        for doc in self.current_docs:
            doc_id = str(doc.get('_id', 'N/A'))[:8]
            title = str(doc.get('title', 'N/A'))
            # Dynamic truncation based on available space
            if len(title) > title_width:
                title = title[:title_width-3] + "..."
            source = str(doc.get('source', 'N/A'))[:12]
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
            # Compact format for small screens
            try:
                if shutil.get_terminal_size().columns < 100:
                    return date_val.strftime("%m/%d %H:%M")
                else:
                    return date_val.strftime("%Y-%m-%d %H:%M")
            except:
                return date_val.strftime("%m/%d %H:%M")
        return str(date_val)[:16]
    
    def update_pagination(self):
        """Update page counter"""
        total_pages = max(1, (self.total_docs + PAGE_SIZE - 1) // PAGE_SIZE)
        current = min(self.page + 1, total_pages)
        
        info = f"{current}/{total_pages}"
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
    
    def on_key(self, event) -> None:
        """Handle key events for auto-pagination"""
        if self.check_view_visible():
            return
        
        table = self.query_one("#article-table", DataTable)
        cursor_row = table.cursor_row
        
        if event.key == "down" or event.key == "j":
            if cursor_row == len(self.current_docs) - 1 and len(self.current_docs) > 0:
                if (self.page + 1) * PAGE_SIZE < self.total_docs:
                    self.page += 1
                    self.load_data()
                    event.stop()
        
        elif event.key == "up" or event.key == "k":
            if cursor_row == 0 and self.page > 0:
                self.page -= 1
                self.load_data()
                table.move_cursor(row=len(self.current_docs) - 1)
                event.stop()
    
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
