# BPSA - Beyond Python SmolAgents
# https://github.com/joaopauloschuler/beyond-python-smolagents
#
# Copyright (c) 2024-2026 Joao Paulo Schwarz Schuler and others.
# Refer to the git commit history for individual authorship.
# Licensed under the Apache License, Version 2.0

"""
Browser integration for BPSA CLI using Playwright.

Provides Tool classes for browser automation that can be used by any agent.

Playwright runs in a dedicated background thread to avoid conflicting with
prompt_toolkit's asyncio.run() on the main thread.
"""

import os
import queue
import threading

from .tools import Tool


def _check_chromium_installed():
    """Check if Chromium browser is installed for Playwright.
    
    Returns the path to the Chromium executable if found, None otherwise.
    """
    # Get the Playwright cache directory
    cache_dir = os.path.expanduser("~/.cache/ms-playwright")
    
    if not os.path.exists(cache_dir):
        return None
    
    # Look for chromium directories (they start with 'chromium-')
    try:
        chromium_dirs = [d for d in os.listdir(cache_dir) if d.startswith('chromium-')]
        if not chromium_dirs:
            return None
        
        # Check if the chrome executable exists in any of the chromium directories
        for chromium_dir in chromium_dirs:
            chrome_path = os.path.join(cache_dir, chromium_dir, 'chrome-linux', 'chrome')
            if os.path.exists(chrome_path):
                return chrome_path
            
            # Also check for chrome-linux64 (newer versions)
            chrome_path = os.path.join(cache_dir, chromium_dir, 'chrome-linux64', 'chrome')
            if os.path.exists(chrome_path):
                return chrome_path
        
        return None
    except Exception:
        return None


class BrowserManager:
    """Manages a headed Chromium browser in a dedicated thread.

    All Playwright calls are dispatched to the background thread via a
    request/response queue pair so the main thread's event loop stays free
    for prompt_toolkit.
    """

    def __init__(self):
        self._req_q = queue.Queue()
        self._res_q = queue.Queue()
        self._thread = None
        self._ready = threading.Event()

    def _ensure_thread(self):
        """Start the Playwright thread on first use."""
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        self._ready.wait()

    def _run_loop(self):
        """Background thread: owns Playwright, browser, and page."""
        from playwright.sync_api import sync_playwright
        
        # Check if Chromium is installed before trying to launch
        chromium_path = _check_chromium_installed()
        if chromium_path is None:
            error_msg = (
                "Chromium browser is not installed for Playwright.\n"
                "Please run the following command to install it:\n"
                "  playwright install chromium\n"
                "Or if you're using a virtual environment:\n"
                "  python -m playwright install chromium"
            )
            self._res_q.put((False, RuntimeError(error_msg)))
            self._ready.set()
            return
        
        pw = sync_playwright().start()
        browser = pw.chromium.launch(headless=False, slow_mo=300)
        page = browser.new_page()
        self._ready.set()

        while True:
            fn = self._req_q.get()
            if fn is None:  # shutdown sentinel
                break
            try:
                result = fn(page)
                self._res_q.put((True, result))
            except Exception as e:
                self._res_q.put((False, e))

        try:
            browser.close()
        except Exception:
            pass
        try:
            pw.stop()
        except Exception:
            pass

    def run(self, fn):
        """Execute *fn(page)* on the Playwright thread and return the result."""
        self._ensure_thread()
        self._req_q.put(fn)
        ok, value = self._res_q.get()
        if ok:
            return value
        raise value

    def shutdown(self):
        """Clean up browser resources."""
        if self._thread is not None:
            self._req_q.put(None)
            self._thread.join(timeout=10)
            self._thread = None


class NavigateTool(Tool):
    """Tool to navigate the browser to a URL."""

    name = "browser_navigate"
    description = "Navigate the browser to a URL. Returns the page title. The browser launches visibly on first use and persists across turns."
    inputs = {
        "url": {"type": "string", "description": "The URL to navigate to"},
    }
    output_type = "string"

    def __init__(self, browser_manager: BrowserManager):
        self.browser_manager = browser_manager
        super().__init__()

    def forward(self, url: str) -> str:
        return self.browser_manager.run(lambda page: (page.goto(url), page.title())[1])


class GetPageHtmlTool(Tool):
    """Tool to get HTML content from the current page."""

    name = "browser_get_page_html"
    description = "Return innerHTML of a CSS selector, or the full page HTML if no selector is given."
    inputs = {
        "selector": {"type": "string", "description": "CSS selector to scope the HTML retrieval (e.g. '#id', '.class', 'div.content'). Leave empty for full page.", "nullable": True},
    }
    output_type = "string"

    def __init__(self, browser_manager: BrowserManager):
        self.browser_manager = browser_manager
        super().__init__()

    def forward(self, selector: str | None = None) -> str:
        if selector:
            return self.browser_manager.run(lambda page: page.locator(selector).inner_html())
        return self.browser_manager.run(lambda page: page.content())


class GetPageMarkdownTool(Tool):
    """Tool to get page content as markdown."""

    name = "browser_get_page_markdown"
    description = "Return page content as markdown, optionally scoped to a CSS selector. **Preferred method for retrieving page content** as markdown is much more compact than raw HTML."
    inputs = {
        "selector": {"type": "string", "description": "CSS selector to scope the content retrieval (e.g. '#id', '.class', 'div.content'). Leave empty for full page.", "nullable": True},
    }
    output_type = "string"

    def __init__(self, browser_manager: BrowserManager):
        self.browser_manager = browser_manager
        super().__init__()

    def forward(self, selector: str | None = None) -> str:
        from markdownify import markdownify
        if selector:
            html = self.browser_manager.run(lambda page: page.locator(selector).inner_html())
        else:
            html = self.browser_manager.run(lambda page: page.content())
        return markdownify(html)


class ClickTool(Tool):
    """Tool to click an element on the page."""

    name = "browser_click"
    description = "Click an element matching the CSS selector."
    inputs = {
        "selector": {"type": "string", "description": "CSS selector of the element to click (e.g. '#submit', 'button[type=submit]', 'a[href*=login]')"},
    }
    output_type = "null"

    def __init__(self, browser_manager: BrowserManager):
        self.browser_manager = browser_manager
        super().__init__()

    def forward(self, selector: str) -> None:
        self.browser_manager.run(lambda page: page.locator(selector).click())


class TypeTextTool(Tool):
    """Tool to type text into an input element."""

    name = "browser_type_text"
    description = "Fill an input element matching the CSS selector with text."
    inputs = {
        "selector": {"type": "string", "description": "CSS selector of the input element (e.g. '#username', 'input[name=email]')"},
        "text": {"type": "string", "description": "The text to type into the element"},
    }
    output_type = "null"

    def __init__(self, browser_manager: BrowserManager):
        self.browser_manager = browser_manager
        super().__init__()

    def forward(self, selector: str, text: str) -> None:
        self.browser_manager.run(lambda page: page.locator(selector).fill(text))


def create_browser_tools() -> tuple[BrowserManager, list[Tool]]:
    """Create a BrowserManager and return (manager, list_of_tools)."""
    manager = BrowserManager()
    tools = [
        NavigateTool(manager),
        GetPageHtmlTool(manager),
        GetPageMarkdownTool(manager),
        ClickTool(manager),
        TypeTextTool(manager),
    ]
    return manager, tools
