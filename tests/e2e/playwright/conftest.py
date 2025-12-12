"""
Pytest configuration and fixtures for session export Playwright tests.

These fixtures support data-driven E2E tests that validate the UI
produces transformations matching recorded session exports.
"""

import pytest
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from playwright.sync_api import Page, BrowserContext

# Add package to Python path
PACKAGE_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PACKAGE_ROOT))

# Directory paths
TEST_DATA_DIR = PACKAGE_ROOT / "test_data"
ARTIFACTS_DIR = Path(__file__).parent.parent.parent / "artifacts"
SCREENSHOTS_DIR = Path(__file__).parent.parent.parent / "screenshots" / "session_export_tests"

# App configuration
APP_URL = "http://localhost:8501"


# =============================================================================
# DATA CLASSES FOR SESSION EXPORTS
# =============================================================================

@dataclass
class SessionExportData:
    """Parsed session export data for test validation."""
    session_id: str
    config_name: str
    config_description: str

    # Navigation tree
    nodes: List[Dict[str, Any]]
    root_id: str
    current_id: str
    current_path: List[str]

    # Expected outputs at each level
    sources: Dict[str, Any]
    joined_table: Dict[str, Any]
    categorized_table: Dict[str, Any]
    vector: Dict[str, Any]
    datum: Dict[str, Any]

    # Design choices
    design_choices: Dict[str, str]

    @classmethod
    def from_file(cls, filepath: Path) -> 'SessionExportData':
        """Load session export from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        cumulative = data.get("cumulative_outputs", {})

        return cls(
            session_id=data.get("session_id", ""),
            config_name=data.get("config_name", ""),
            config_description=data.get("config_description", ""),
            nodes=data.get("navigation_tree", {}).get("nodes", []),
            root_id=data.get("navigation_tree", {}).get("root_id", ""),
            current_id=data.get("navigation_tree", {}).get("current_id", ""),
            current_path=data.get("current_path", []),
            sources=cumulative.get("sources", {}),
            joined_table=cumulative.get("joined_table", {}),
            categorized_table=cumulative.get("categorized_table", {}),
            vector=cumulative.get("vector", {}),
            datum=cumulative.get("datum", {}),
            design_choices=data.get("design_choices_summary", {})
        )

    def get_descent_nodes(self) -> List[Dict[str, Any]]:
        """Get nodes from the descent phase (L4→L0)."""
        return [n for n in self.nodes if n.get("action") in ("entry", "descend")]

    def get_ascent_nodes(self) -> List[Dict[str, Any]]:
        """Get nodes from the ascent phase (L0→L3)."""
        return [n for n in self.nodes if n.get("action") == "ascend"]

    def get_node_by_level(self, level: int, action: str = None) -> Optional[Dict[str, Any]]:
        """Get a specific node by level and optionally action type."""
        for node in self.nodes:
            if node.get("level") == level:
                if action is None or node.get("action") == action:
                    return node
        return None


# =============================================================================
# SESSION FIXTURES
# =============================================================================

@pytest.fixture(scope="session")
def browser_context_args(browser_context_args):
    """Configure browser context for Streamlit app testing."""
    return {
        **browser_context_args,
        "viewport": {"width": 1400, "height": 1000},
    }


@pytest.fixture(autouse=True)
def setup_screenshot_dir():
    """Create screenshot directory before tests."""
    SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)


@pytest.fixture(scope="session")
def test_data_dir():
    """Return the test data directory."""
    return TEST_DATA_DIR


@pytest.fixture(scope="session")
def artifacts_dir():
    """Return the artifacts directory."""
    return ARTIFACTS_DIR


# =============================================================================
# SESSION EXPORT FIXTURES
# =============================================================================

@pytest.fixture
def test0_schools_session(artifacts_dir) -> SessionExportData:
    """Load the test0 schools session export."""
    filepath = artifacts_dir / "20251208_domain_specific_v2" / "test0_schools" / "test0_schools_session_export.json"
    return SessionExportData.from_file(filepath)


@pytest.fixture
def test1_ademe_session(artifacts_dir) -> SessionExportData:
    """Load the test1 ADEME session export."""
    filepath = artifacts_dir / "20251208_domain_specific_v2" / "test1_ademe" / "test1_ademe_session_export.json"
    return SessionExportData.from_file(filepath)


# =============================================================================
# APP PAGE FIXTURE
# =============================================================================

@pytest.fixture
def app_page(page: Page) -> Page:
    """Navigate to app and wait for it to load."""
    page.goto(APP_URL)
    # Wait for Streamlit to fully load
    try:
        page.wait_for_selector('[data-testid="stApp"]', timeout=30000)
    except:
        page.wait_for_timeout(5000)
    # Wait for initial loading to complete
    page.wait_for_timeout(3000)
    return page


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def wait_for_streamlit_rerun(page: Page, timeout: int = 10000):
    """Wait for Streamlit to finish rerunning after an action."""
    page.wait_for_timeout(1000)
    try:
        page.wait_for_selector('[data-testid="stSpinner"]', state="hidden", timeout=timeout)
    except:
        pass
    page.wait_for_timeout(500)


def take_screenshot(page: Page, test_name: str, step_name: str) -> Path:
    """Take a screenshot with test name and step."""
    timestamp = datetime.now().strftime("%H%M%S")
    filepath = SCREENSHOTS_DIR / f"{test_name}_{timestamp}_{step_name}.png"
    page.screenshot(path=str(filepath), full_page=True)
    return filepath


def scroll_to_bottom(page: Page):
    """Scroll to the bottom of the page."""
    page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
    page.wait_for_timeout(500)


def scroll_to_top(page: Page):
    """Scroll to the top of the page."""
    page.evaluate("window.scrollTo(0, 0)")
    page.wait_for_timeout(500)


def click_button(page: Page, text: str, timeout: int = 5000) -> bool:
    """Find and click a button with given text."""
    try:
        btn = page.locator(f'button:has-text("{text}")')
        if btn.count() > 0:
            btn.first.scroll_into_view_if_needed()
            btn.first.click()
            wait_for_streamlit_rerun(page)
            return True
    except:
        pass
    return False


def upload_files(page: Page, filepaths: list) -> bool:
    """Upload files to the file input."""
    try:
        file_input = page.locator('input[type="file"]')
        if file_input.count() > 0:
            file_input.first.set_input_files(filepaths)
            wait_for_streamlit_rerun(page, timeout=20000)
            return True
    except Exception as e:
        print(f"Upload failed: {e}")
    return False


def find_text_on_page(page: Page, text: str) -> bool:
    """Check if text appears anywhere on the page."""
    try:
        return page.locator(f'text="{text}"').count() > 0 or \
               page.locator(f'text={text}').count() > 0
    except:
        return False


def get_displayed_value(page: Page, label: str) -> Optional[str]:
    """Extract a displayed value near a label."""
    try:
        # Look for metric displays
        metric = page.locator(f'[data-testid="stMetric"]:has-text("{label}")')
        if metric.count() > 0:
            value = metric.locator('[data-testid="stMetricValue"]')
            if value.count() > 0:
                return value.first.inner_text()
    except:
        pass
    return None


def select_dropdown_option(page: Page, dropdown_index: int, option_text: str) -> bool:
    """Select an option from a dropdown by index."""
    try:
        selectboxes = page.locator('[data-testid="stSelectbox"]')
        if selectboxes.count() > dropdown_index:
            selectbox = selectboxes.nth(dropdown_index)
            selectbox.scroll_into_view_if_needed()
            selectbox.click()
            page.wait_for_timeout(500)

            option = page.locator(f'li[role="option"]:has-text("{option_text}")')
            if option.count() > 0:
                option.first.click()
                page.wait_for_timeout(500)
                return True
            else:
                page.keyboard.press("Escape")
    except:
        pass
    return False


def fill_textarea(page: Page, value: str) -> bool:
    """Fill the first textarea on the page."""
    try:
        textareas = page.locator('textarea')
        if textareas.count() > 0:
            textareas.first.fill(value)
            return True
    except:
        pass
    return False
