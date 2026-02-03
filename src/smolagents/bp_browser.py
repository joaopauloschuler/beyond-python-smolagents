"""
Browser integration for BPSA CLI using Playwright.

Provides functions that can be injected into the agent's executor namespace
for browser automation in <runcode> blocks.

Playwright runs in a dedicated background thread to avoid conflicting with
prompt_toolkit's asyncio.run() on the main thread.
"""

import threading
import queue

_browser_manager = None


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


def navigate(url: str) -> str:
    """Navigate to a URL. Returns the page title."""
    return _browser_manager.run(lambda page: (page.goto(url), page.title())[1])


def get_page_html(selector: str | None = None) -> str:
    """Return innerHTML of a selector, or the full page body."""
    if selector:
        return _browser_manager.run(lambda page: page.locator(selector).inner_html())
    return _browser_manager.run(lambda page: page.content())


def get_page_markdown(selector: str | None = None) -> str:
    """Return page content as markdown, optionally scoped to a selector."""
    from markdownify import markdownify
    html = get_page_html(selector)
    return markdownify(html)


def click(selector: str) -> None:
    """Click an element matching the selector."""
    _browser_manager.run(lambda page: page.locator(selector).click())


def type_text(selector: str, text: str) -> None:
    """Fill an element matching the selector with text."""
    _browser_manager.run(lambda page: page.locator(selector).fill(text))


def get_browser_manager() -> BrowserManager:
    """Access the underlying BrowserManager instance."""
    return _browser_manager


BROWSER_AGENT_INSTRUCTIONS = """
## Browser Functions

You have access to a headed Chromium browser. The following functions are available in your <runcode> blocks:

- `navigate(url: str) -> str` — Navigate to a URL. Returns the page title.
- `get_page_html(selector: str | None = None) -> str` — Return innerHTML of a CSS selector, or the full page HTML if no selector is given.
- `get_page_markdown(selector: str | None = None) -> str` — Same as get_page_html but converts the result to markdown. **Preferred method for retrieving page content** as markdown is much more compact than raw HTML.
- `click(selector: str) -> None` — Click an element matching the CSS selector.
- `type_text(selector: str, text: str) -> None` — Fill an input element matching the CSS selector with text.

Selectors are standard CSS selectors (e.g. `"#id"`, `".class"`, `"button[type=submit]"`, `"a[href*=login]"`).
The browser launches visibly on first use and persists across turns.
"""


def create_browser_functions() -> tuple["BrowserManager", dict]:
    """Create a BrowserManager and return (manager, functions_dict)."""
    global _browser_manager
    manager = BrowserManager()
    _browser_manager = manager
    funcs = {
        "navigate": navigate,
        "get_page_html": get_page_html,
        "get_page_markdown": get_page_markdown,
        "click": click,
        "type_text": type_text,
        "get_browser_manager": get_browser_manager,
    }
    return manager, funcs
