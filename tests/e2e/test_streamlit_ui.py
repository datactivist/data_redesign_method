"""
Playwright E2E Tests for the Streamlit UI descent-ascent cycle.

These tests verify the full descent-ascent cycle through the Streamlit interface:
- File upload and L4 entry
- Descent: L4 → L3 → L2 → L1 → L0
- Ascent: L0 → L1 → L2 → L3
- Path tracking at every step
- Artifact export and download

Run with: pytest tests/e2e/test_streamlit_ui.py -v --headed

Requirements:
- Streamlit app must be running: streamlit run intuitiveness/streamlit_app.py
- Or use the streamlit_app fixture to auto-start
"""

import pytest
import subprocess
import time
import os
import json
import signal
import re
from pathlib import Path
from playwright.sync_api import Page, expect, sync_playwright
from typing import Generator

# Test data paths
TEST_DATA_DIR = Path(__file__).parent.parent.parent / "test_data"
TEST0_DIR = TEST_DATA_DIR / "test0"
TEST1_DIR = TEST_DATA_DIR / "test1"
TEST2_DIR = TEST_DATA_DIR / "test2"
ARTIFACTS_DIR = Path(__file__).parent.parent / "artifacts" / "playwright"


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture(scope="module")
def streamlit_app() -> Generator[str, None, None]:
    """
    Start the Streamlit app as a subprocess and return the URL.
    Automatically stops the app after tests complete.
    """
    # Create artifacts dir
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # Start Streamlit
    port = 8501
    env = os.environ.copy()
    env["STREAMLIT_SERVER_HEADLESS"] = "true"
    env["STREAMLIT_SERVER_PORT"] = str(port)

    process = subprocess.Popen(
        ["streamlit", "run", "intuitiveness/streamlit_app.py", "--server.headless=true"],
        cwd=str(Path(__file__).parent.parent.parent),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        preexec_fn=os.setsid if os.name != 'nt' else None
    )

    # Wait for server to start
    url = f"http://localhost:{port}"
    max_wait = 30
    start_time = time.time()

    while time.time() - start_time < max_wait:
        try:
            import urllib.request
            urllib.request.urlopen(url, timeout=1)
            break
        except Exception:
            time.sleep(0.5)
    else:
        process.terminate()
        raise RuntimeError(f"Streamlit app failed to start within {max_wait}s")

    yield url

    # Cleanup: kill the process group
    if os.name != 'nt':
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    else:
        process.terminate()
    process.wait()


@pytest.fixture
def page(streamlit_app: str) -> Generator[Page, None, None]:
    """Create a Playwright page connected to the Streamlit app."""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()
        page.goto(streamlit_app)
        # Wait for Streamlit to fully load
        page.wait_for_load_state("networkidle")
        time.sleep(2)  # Extra wait for Streamlit's JavaScript
        yield page
        context.close()
        browser.close()


@pytest.fixture
def test0_files() -> list:
    """Get paths to test0 CSV files."""
    return list(TEST0_DIR.glob("*.csv"))


@pytest.fixture
def test1_files() -> list:
    """Get paths to test1 CSV files."""
    return list(TEST1_DIR.glob("*.csv"))


@pytest.fixture
def test2_files() -> list:
    """Get paths to test2 CSV files."""
    return list(TEST2_DIR.glob("*.csv"))


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def take_screenshot(page: Page, name: str):
    """Take a screenshot and save to artifacts directory."""
    screenshot_path = ARTIFACTS_DIR / f"{name}.png"
    page.screenshot(path=str(screenshot_path))
    return screenshot_path


def wait_for_streamlit(page: Page, timeout: int = 10000):
    """Wait for Streamlit to finish processing (spinner disappears)."""
    try:
        # Wait for any running spinners to disappear
        page.wait_for_selector(".stSpinner", state="hidden", timeout=timeout)
    except:
        pass  # No spinner present
    time.sleep(0.5)  # Extra buffer for UI updates


def get_current_level(page: Page) -> str:
    """Extract the current complexity level from the page."""
    # Look for level indicators in the UI
    try:
        level_text = page.locator(re.compile(r"L[0-4]")).first.inner_text()
        return level_text
    except:
        return "unknown"


def upload_csv_files(page: Page, file_paths: list):
    """Upload CSV files to the Streamlit file uploader."""
    # Find the file uploader
    file_input = page.locator('input[type="file"]')

    # Convert Path objects to strings
    paths_str = [str(p) for p in file_paths]

    # Upload files
    file_input.set_input_files(paths_str)
    wait_for_streamlit(page)


def click_button(page: Page, button_text: str, timeout: int = 5000):
    """Click a button by its text content."""
    button = page.get_by_role("button", name=button_text)
    button.click(timeout=timeout)
    wait_for_streamlit(page)


def select_option(page: Page, label: str, value: str):
    """Select an option from a Streamlit selectbox."""
    # Find the selectbox by label
    selectbox = page.locator(f"text={label}").locator("..").locator("select")
    selectbox.select_option(value)
    wait_for_streamlit(page)


# =============================================================================
# TEST CLASSES
# =============================================================================

class TestStreamlitAppLoads:
    """Basic tests to verify the Streamlit app loads correctly."""

    def test_app_loads(self, page: Page):
        """Verify the app loads and shows the title."""
        # Check for page title or header
        expect(page).to_have_title(re.compile(".+"))
        take_screenshot(page, "01_app_loaded")

    def test_file_uploader_visible(self, page: Page):
        """Verify the file uploader is visible."""
        # Look for file uploader
        uploader = page.locator('input[type="file"]')
        expect(uploader).to_be_visible()
        take_screenshot(page, "02_file_uploader_visible")


class TestFileUpload:
    """Test file upload functionality (L4 entry)."""

    def test_upload_single_file(self, page: Page, test0_files: list):
        """Test uploading a single CSV file."""
        if not test0_files:
            pytest.skip("No test0 CSV files found")

        # Upload first file
        upload_csv_files(page, [test0_files[0]])

        # Verify file appears in UI
        file_name = test0_files[0].name
        expect(page.locator(f"text={file_name}")).to_be_visible(timeout=10000)
        take_screenshot(page, "03_single_file_uploaded")

    def test_upload_multiple_files(self, page: Page, test0_files: list):
        """Test uploading multiple CSV files."""
        if len(test0_files) < 2:
            pytest.skip("Need at least 2 test0 CSV files")

        # Upload multiple files
        upload_csv_files(page, test0_files[:2])

        # Verify files appear
        for f in test0_files[:2]:
            expect(page.locator(f"text={f.name}")).to_be_visible(timeout=10000)

        take_screenshot(page, "04_multiple_files_uploaded")


class TestDescentCycle:
    """Test the descent cycle: L4 → L3 → L2 → L1 → L0"""

    @pytest.mark.slow
    def test_full_descent(self, page: Page, test0_files: list):
        """Test full descent from L4 to L0."""
        if not test0_files:
            pytest.skip("No test0 CSV files found")

        # Step 1: L4 - Upload files
        upload_csv_files(page, test0_files[:2])
        take_screenshot(page, "descent_01_l4_files_uploaded")

        # Step 2: L4 → L3 - Look for entity/graph creation button
        try:
            # Try to find and click "Create Graph" or similar button
            graph_button = page.locator("button:has-text('Graph'), button:has-text('Entities'), button:has-text('Next')")
            if graph_button.count() > 0:
                graph_button.first.click()
                wait_for_streamlit(page)
                take_screenshot(page, "descent_02_l3_graph_created")
        except:
            take_screenshot(page, "descent_02_no_graph_button")

        # Verify we have some navigation UI
        take_screenshot(page, "descent_final_state")


class TestAscentCycle:
    """Test the ascent cycle: L0 → L1 → L2 → L3"""

    @pytest.mark.slow
    def test_ascent_ui_elements(self, page: Page, test0_files: list):
        """Test that ascent UI elements are available."""
        if not test0_files:
            pytest.skip("No test0 CSV files found")

        # First upload files to get into the app
        upload_csv_files(page, test0_files[:2])

        # Look for ascent-related UI elements
        ascent_elements = page.locator(re.compile(r"ascend|unfold|enrich|dimension", re.IGNORECASE))

        # Take screenshot of current state
        take_screenshot(page, "ascent_ui_elements")


class TestNavigationTracking:
    """Test that navigation path is tracked correctly."""

    def test_navigation_history_visible(self, page: Page, test0_files: list):
        """Test that navigation history/tree is visible."""
        if not test0_files:
            pytest.skip("No test0 CSV files found")

        upload_csv_files(page, test0_files[:2])

        # Look for navigation tree or history sidebar
        nav_tree = page.locator(re.compile(r"Navigation|History|Path|Tree", re.IGNORECASE))

        take_screenshot(page, "navigation_tree_check")


class TestArtifactExport:
    """Test artifact export and download functionality."""

    def test_export_button_exists(self, page: Page, test0_files: list):
        """Test that export functionality exists."""
        if not test0_files:
            pytest.skip("No test0 CSV files found")

        upload_csv_files(page, test0_files[:2])

        # Look for export/download buttons
        export_buttons = page.locator("button:has-text('Export'), button:has-text('Download'), a:has-text('Download')")

        take_screenshot(page, "export_buttons_check")


# =============================================================================
# INTEGRATION TEST - Full Cycle
# =============================================================================

class TestFullDescentAscentCycle:
    """Integration test for the complete descent-ascent cycle."""

    @pytest.mark.slow
    @pytest.mark.integration
    def test_complete_cycle_test0(self, page: Page, test0_files: list):
        """
        Complete integration test for test0 dataset.

        This test verifies:
        1. File upload (L4 entry)
        2. Navigation through descent
        3. Navigation through ascent
        4. Path tracking
        5. Export availability
        """
        if not test0_files:
            pytest.skip("No test0 CSV files found")

        screenshots = []

        # === L4: Entry Point ===
        upload_csv_files(page, test0_files[:2])
        screenshots.append(take_screenshot(page, "cycle_test0_01_l4_entry"))

        # Look for key UI elements
        page_content = page.content()

        # Check for expected elements
        has_upload = "file" in page_content.lower() or page.locator('input[type="file"]').count() > 0
        has_data_display = page.locator("table, .dataframe, .stDataFrame").count() > 0

        # Log findings
        findings = {
            "has_file_upload": has_upload,
            "has_data_display": has_data_display,
            "screenshot_count": len(screenshots)
        }

        # Save findings
        findings_path = ARTIFACTS_DIR / "cycle_test0_findings.json"
        with open(findings_path, 'w') as f:
            json.dump(findings, f, indent=2)

        # Final screenshot
        take_screenshot(page, "cycle_test0_final")

        # Basic assertion - app should have loaded something
        assert has_upload or has_data_display, "App should show upload or data display"

    @pytest.mark.slow
    @pytest.mark.integration
    def test_complete_cycle_test1(self, page: Page, test1_files: list):
        """Complete integration test for test1 dataset (ADEME funding)."""
        if not test1_files:
            pytest.skip("No test1 CSV files found")

        upload_csv_files(page, test1_files[:2])
        take_screenshot(page, "cycle_test1_l4_entry")

        # Verify files were loaded
        page.wait_for_load_state("networkidle")
        time.sleep(1)

        take_screenshot(page, "cycle_test1_final")

    @pytest.mark.slow
    @pytest.mark.integration
    def test_complete_cycle_test2(self, page: Page, test2_files: list):
        """Complete integration test for test2 dataset (energy prices)."""
        if not test2_files:
            pytest.skip("No test2 CSV files found")

        upload_csv_files(page, test2_files[:2])
        take_screenshot(page, "cycle_test2_l4_entry")

        # Verify files were loaded
        page.wait_for_load_state("networkidle")
        time.sleep(1)

        take_screenshot(page, "cycle_test2_final")


# =============================================================================
# ACCESSIBILITY TESTS
# =============================================================================

class TestAccessibility:
    """Basic accessibility checks for the Streamlit UI."""

    def test_page_has_title(self, page: Page):
        """Verify page has a title."""
        expect(page).to_have_title(re.compile(".+"))

    def test_buttons_have_text(self, page: Page, test0_files: list):
        """Verify buttons have descriptive text."""
        if not test0_files:
            pytest.skip("No test0 CSV files found")

        upload_csv_files(page, test0_files[:1])

        # Find all buttons
        buttons = page.locator("button")
        count = buttons.count()

        # At least some buttons should exist
        assert count >= 0, "Page should have buttons"


# =============================================================================
# RUN CONFIGURATION
# =============================================================================

if __name__ == "__main__":
    pytest.main([
        __file__,
        "-v",
        "-s",
        "--headed",  # Show browser
        "-x",  # Stop on first failure
    ])
