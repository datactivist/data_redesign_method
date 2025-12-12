"""
Playwright E2E Test for Full Descent-Ascent Cycle with Artifact Downloads.

This test automates the complete workflow through the Streamlit interface:
1. File upload (L4 entry)
2. Descent: L4 → L3 → L2 → L1 → L0
3. Ascent: L0 → L1 → L2 → L3
4. Download all artifacts at each level

Based on session exports:
- test0_schools_session_export.json
- test1_ademe_session_export.json

Run with:
    pytest tests/e2e/test_full_descent_ascent.py -v --headed -s

Or run directly:
    python tests/e2e/test_full_descent_ascent.py

Requirements:
- Streamlit app must be running: streamlit run intuitiveness/streamlit_app.py
- Or the fixture will auto-start it

Author: Data Redesign Method
"""

import pytest
import subprocess
import time
import os
import json
import signal
import re
import shutil
from pathlib import Path
from playwright.sync_api import Page, expect, sync_playwright, Download
from typing import Generator, Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass


# =============================================================================
# CONFIGURATION
# =============================================================================

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
TEST_DATA_DIR = PROJECT_ROOT / "test_data"
ARTIFACTS_BASE_DIR = PROJECT_ROOT / "tests" / "artifacts"

# Test configurations matching the CLAUDE.md requirements
TEST_CONFIGS = {
    "test0_schools": {
        "name": "test0_schools",
        "description": "French middle schools - student counts and performance scores",
        "data_dir": TEST_DATA_DIR / "test0",
        "files": [
            "fr-en-college-effectifs-niveau-sexe-lv.csv",
            "fr-en-indicateurs-valeur-ajoutee-colleges.csv"
        ],
        # Descent configuration
        "descent": {
            "L4_to_L3": {
                "left_columns": ["Dénomination principale", "Patronyme", "Commune"],
                "right_columns": ["Nom de l'établissement", "Commune"],
                "similarity_threshold": 0.85
            },
            "L3_to_L2": {
                "category_name": "location_type",
                "category_column": "nombre_eleves_total",
                "categories": ["downtown", "countryside"]
            },
            "L2_to_L1": {
                "column": "Taux de réussite G"
            },
            "L1_to_L0": {
                "aggregation": "mean",
                "description": "Average middle school success rate"
            }
        },
        # Ascent configuration
        "ascent": {
            "L0_to_L1": {
                "expansion_method": "source_recovery"
            },
            "L1_to_L2": {
                "dimension_name": "performance_category",
                "categories": ["above_median", "below_median"]
            },
            "L2_to_L3": {
                "enrichment_method": "dimension_merge"
            }
        }
    },
    "test1_ademe": {
        "name": "test1_ademe",
        "description": "ADEME funding - environmental subsidies",
        "data_dir": TEST_DATA_DIR / "test1",
        "files": [
            "ECS.csv",
            "Les aides financieres ADEME.csv"
        ],
        "descent": {
            "L4_to_L3": {
                "left_column": "dispositifAide",
                "right_column": "type_aides_financieres",
                "similarity_threshold": 0.75
            },
            "L3_to_L2": {
                "category_name": "funding_frequency",
                "category_column": "nomBeneficiaire",
                "categories": ["single_funding", "multiple_funding"]
            },
            "L2_to_L1": {
                "column": "montant",
                "group_by": "nomBeneficiaire",
                "aggregation": "sum"
            },
            "L1_to_L0": {
                "aggregation": "sum",
                "description": "Total ADEME funding amount"
            }
        },
        "ascent": {
            "L0_to_L1": {
                "expansion_method": "source_recovery"
            },
            "L1_to_L2": {
                "dimension_name": "funding_size",
                "categories": ["above_10k", "below_10k"]
            },
            "L2_to_L3": {
                "enrichment_method": "dimension_merge"
            }
        }
    },
    "test2_energy": {
        "name": "test2_energy",
        "description": "Energy prices and imports/exports",
        "data_dir": TEST_DATA_DIR / "test2",
        "files": [
            "Niveaux_prix_TRVG.csv",
            "imports-exports-commerciaux.csv"
        ],
        "descent": {
            "L4_to_L3": {
                "left_column": "pays",
                "right_column": "pays",
                "similarity_threshold": 0.8
            },
            "L3_to_L2": {
                "category_name": "price_category",
                "categories": ["high_price", "low_price"]
            },
            "L2_to_L1": {
                "column": "prix"
            },
            "L1_to_L0": {
                "aggregation": "sum",
                "description": "Total energy consumption"
            }
        },
        "ascent": {
            "L0_to_L1": {
                "expansion_method": "source_recovery"
            },
            "L1_to_L2": {
                "dimension_name": "energy_type_category",
                "categories": ["high_consumption", "low_consumption"]
            },
            "L2_to_L3": {
                "enrichment_method": "dimension_merge"
            }
        }
    }
}


@dataclass
class ArtifactInfo:
    """Information about a downloaded artifact."""
    level: str
    filename: str
    path: Path
    size_bytes: int
    timestamp: str


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture(scope="module")
def streamlit_app() -> Generator[str, None, None]:
    """
    Start the Streamlit app as a subprocess and return the URL.
    Automatically stops the app after tests complete.
    """
    port = 8501
    env = os.environ.copy()
    env["STREAMLIT_SERVER_HEADLESS"] = "true"
    env["STREAMLIT_SERVER_PORT"] = str(port)

    process = subprocess.Popen(
        ["streamlit", "run", "intuitiveness/streamlit_app.py", "--server.headless=true"],
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        preexec_fn=os.setsid if os.name != 'nt' else None
    )

    # Wait for server to start
    url = f"http://localhost:{port}"
    max_wait = 60
    start_time = time.time()

    while time.time() - start_time < max_wait:
        try:
            import urllib.request
            urllib.request.urlopen(url, timeout=2)
            print(f"Streamlit app started at {url}")
            break
        except Exception:
            time.sleep(1)
    else:
        process.terminate()
        raise RuntimeError(f"Streamlit app failed to start within {max_wait}s")

    yield url

    # Cleanup
    if os.name != 'nt':
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    else:
        process.terminate()
    process.wait()


@pytest.fixture
def artifacts_dir(request) -> Path:
    """Create a unique artifacts directory for this test run."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_name = request.node.name.replace("test_", "")
    artifacts_path = ARTIFACTS_BASE_DIR / f"playwright_{timestamp}_{test_name}"
    artifacts_path.mkdir(parents=True, exist_ok=True)
    return artifacts_path


@pytest.fixture
def page_with_downloads(streamlit_app: str, artifacts_dir: Path) -> Generator[tuple, None, None]:
    """Create a Playwright page with download handling configured."""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False, slow_mo=500)
        context = browser.new_context(
            accept_downloads=True,
            # Set download path
        )
        page = context.new_page()

        # Configure download behavior
        downloads: List[ArtifactInfo] = []

        def handle_download(download: Download):
            """Handle file downloads."""
            filename = download.suggested_filename
            save_path = artifacts_dir / filename
            download.save_as(str(save_path))
            downloads.append(ArtifactInfo(
                level="unknown",
                filename=filename,
                path=save_path,
                size_bytes=save_path.stat().st_size if save_path.exists() else 0,
                timestamp=datetime.now().isoformat()
            ))
            print(f"Downloaded: {filename} -> {save_path}")

        page.on("download", handle_download)

        page.goto(streamlit_app)
        page.wait_for_load_state("networkidle")
        time.sleep(3)  # Extra wait for Streamlit

        yield page, downloads, artifacts_dir

        context.close()
        browser.close()


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def wait_for_streamlit(page: Page, timeout: int = 30000):
    """Wait for Streamlit to finish processing."""
    try:
        # Wait for spinners to disappear
        page.wait_for_selector(".stSpinner", state="hidden", timeout=timeout)
    except:
        pass
    time.sleep(1)


def take_screenshot(page: Page, artifacts_dir: Path, name: str) -> Path:
    """Take a screenshot and save to artifacts directory."""
    screenshot_path = artifacts_dir / f"{name}.png"
    page.screenshot(path=str(screenshot_path), full_page=True)
    return screenshot_path


def upload_files(page: Page, file_paths: List[Path]):
    """Upload CSV files to Streamlit."""
    file_input = page.locator('input[type="file"]')
    paths_str = [str(p) for p in file_paths]
    file_input.set_input_files(paths_str)
    wait_for_streamlit(page, timeout=60000)


def switch_to_free_mode(page: Page):
    """Switch to Free Exploration mode."""
    # Find the radio button for Free Exploration
    free_radio = page.locator('label:has-text("Free Exploration")')
    if free_radio.count() > 0:
        free_radio.click()
        wait_for_streamlit(page)
        print("Switched to Free Exploration mode")


def click_button_safe(page: Page, text: str, timeout: int = 10000) -> bool:
    """Safely click a button by text, returns True if successful."""
    try:
        button = page.get_by_role("button", name=re.compile(text, re.IGNORECASE))
        if button.count() > 0:
            button.first.scroll_into_view_if_needed()
            button.first.click(timeout=timeout)
            wait_for_streamlit(page)
            return True
    except Exception as e:
        print(f"Could not click button '{text}': {e}")
    return False


def click_expander(page: Page, text: str):
    """Click an expander to open it."""
    expander = page.locator(f'text="{text}"')
    if expander.count() > 0:
        expander.first.click()
        time.sleep(0.5)


def fill_text_input(page: Page, placeholder_or_label: str, value: str):
    """Fill a text input by placeholder or label."""
    # Try by placeholder
    input_elem = page.locator(f'input[placeholder*="{placeholder_or_label}"]')
    if input_elem.count() == 0:
        # Try by nearby label
        input_elem = page.locator(f'text="{placeholder_or_label}"').locator("..").locator("input")

    if input_elem.count() > 0:
        input_elem.first.fill(value)
        time.sleep(0.5)


def select_from_dropdown(page: Page, label: str, value: str):
    """Select an option from a Streamlit selectbox."""
    selectbox = page.locator(f'[data-testid="stSelectbox"]:has-text("{label}")')
    if selectbox.count() > 0:
        selectbox.first.click()
        time.sleep(0.5)
        option = page.locator(f'li:has-text("{value}")')
        if option.count() > 0:
            option.first.click()
            time.sleep(0.5)


def scroll_to_bottom(page: Page):
    """Scroll to bottom of page."""
    page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
    time.sleep(0.5)


def scroll_to_top(page: Page):
    """Scroll to top of page."""
    page.evaluate("window.scrollTo(0, 0)")
    time.sleep(0.5)


def find_and_click_download_button(page: Page, text_pattern: str) -> bool:
    """Find and click a download button."""
    download_btn = page.locator(f'button:has-text("{text_pattern}"), a:has-text("{text_pattern}")')
    if download_btn.count() > 0:
        download_btn.first.click()
        time.sleep(2)  # Wait for download
        return True
    return False


def export_current_level_data(page: Page, artifacts_dir: Path, level: str, test_name: str):
    """Export data for the current level."""
    # Look for export/download buttons
    exports_done = []

    # Try common export patterns
    patterns = [
        ("Download CSV", f"{test_name}_{level}_data.csv"),
        ("Download JSON", f"{test_name}_{level}_data.json"),
        ("Export", f"{test_name}_{level}_export.json"),
        ("Download", f"{test_name}_{level}_download")
    ]

    for pattern, _ in patterns:
        if find_and_click_download_button(page, pattern):
            exports_done.append(pattern)

    return exports_done


# =============================================================================
# TEST CLASSES
# =============================================================================

class TestFullDescentAscentCycle:
    """Complete descent-ascent cycle tests for all datasets."""

    @pytest.mark.slow
    def test_complete_workflow_test0_schools(
        self,
        page_with_downloads: tuple,
    ):
        """
        Complete test for test0_schools dataset.

        Workflow:
        1. Upload files
        2. Navigate wizard to create L3 joined table
        3. Free mode: Descend L3 → L2 → L1 → L0
        4. Ascend: L0 → L1 → L2 → L3
        5. Export all artifacts
        """
        page, downloads, artifacts_dir = page_with_downloads
        config = TEST_CONFIGS["test0_schools"]

        print(f"\n{'='*60}")
        print(f"Testing: {config['name']}")
        print(f"Description: {config['description']}")
        print(f"Artifacts dir: {artifacts_dir}")
        print(f"{'='*60}\n")

        # Step 1: Upload files
        print("[Step 1] Uploading files...")
        file_paths = [config["data_dir"] / f for f in config["files"]]
        upload_files(page, file_paths)
        take_screenshot(page, artifacts_dir, "01_files_uploaded")

        # Step 2: Wait for AI analysis and wizard
        print("[Step 2] Waiting for AI analysis...")
        time.sleep(10)  # Wait for discovery to complete
        take_screenshot(page, artifacts_dir, "02_ai_analysis")

        # Step 3: Go through wizard steps
        print("[Step 3] Navigating wizard...")

        # Wizard Step 1: Select columns
        scroll_to_bottom(page)
        time.sleep(1)
        if click_button_safe(page, "Continue"):
            print("  Wizard Step 1 complete")
            take_screenshot(page, artifacts_dir, "03_wizard_step1")

        # Wizard Step 2: Define connections
        time.sleep(2)
        scroll_to_bottom(page)
        if click_button_safe(page, "Continue"):
            print("  Wizard Step 2 complete")
            take_screenshot(page, artifacts_dir, "04_wizard_step2")

        # Wizard Step 3: Confirm
        time.sleep(2)
        scroll_to_bottom(page)
        if click_button_safe(page, "Confirm"):
            print("  Wizard Step 3 complete - L3 created")
            take_screenshot(page, artifacts_dir, "05_wizard_complete")

        # Step 4: Switch to Free Exploration mode
        print("[Step 4] Switching to Free Exploration mode...")
        switch_to_free_mode(page)
        take_screenshot(page, artifacts_dir, "06_free_mode")

        # Step 5: Descend through levels
        print("[Step 5] Beginning descent...")

        # L3 → L2: Categorize
        print("  Descending L3 → L2...")
        if click_button_safe(page, "Explore deeper"):
            time.sleep(2)
            # Fill in categorization options
            scroll_to_bottom(page)
            if click_button_safe(page, "Categorize"):
                print("  L2 created")
                take_screenshot(page, artifacts_dir, "07_descent_L2")

        # L2 → L1: Extract vector
        print("  Descending L2 → L1...")
        if click_button_safe(page, "Explore deeper"):
            time.sleep(2)
            scroll_to_bottom(page)
            if click_button_safe(page, "Descend"):
                print("  L1 created")
                take_screenshot(page, artifacts_dir, "08_descent_L1")

        # L1 → L0: Aggregate
        print("  Descending L1 → L0...")
        if click_button_safe(page, "Explore deeper"):
            time.sleep(2)
            scroll_to_bottom(page)
            if click_button_safe(page, "Descend"):
                print("  L0 created (datum)")
                take_screenshot(page, artifacts_dir, "09_descent_L0")

        # Step 6: Ascend through levels
        print("[Step 6] Beginning ascent...")

        # L0 → L1: Unfold
        print("  Ascending L0 → L1...")
        if click_button_safe(page, "Build up"):
            time.sleep(2)
            scroll_to_bottom(page)
            if click_button_safe(page, "Unfold") or click_button_safe(page, "Ascend"):
                print("  Ascent L1 created")
                take_screenshot(page, artifacts_dir, "10_ascent_L1")

        # L1 → L2: Add dimension
        print("  Ascending L1 → L2...")
        if click_button_safe(page, "Build up"):
            time.sleep(2)
            scroll_to_bottom(page)
            if click_button_safe(page, "Add Domain") or click_button_safe(page, "Ascend"):
                print("  Ascent L2 created")
                take_screenshot(page, artifacts_dir, "11_ascent_L2")

        # L2 → L3: Enrich
        print("  Ascending L2 → L3...")
        if click_button_safe(page, "Build up"):
            time.sleep(2)
            scroll_to_bottom(page)
            if click_button_safe(page, "Enrich") or click_button_safe(page, "Ascend"):
                print("  Ascent L3 created")
                take_screenshot(page, artifacts_dir, "12_ascent_L3")

        # Step 7: Export session
        print("[Step 7] Exporting session...")
        if click_button_safe(page, "Exit") or click_button_safe(page, "Export"):
            time.sleep(2)
            take_screenshot(page, artifacts_dir, "13_export")

            # Try to download the session export
            find_and_click_download_button(page, "Download")

        # Final screenshot
        take_screenshot(page, artifacts_dir, "14_final")

        # Save test summary
        summary = {
            "test_name": config["name"],
            "description": config["description"],
            "artifacts_dir": str(artifacts_dir),
            "downloads": [{"filename": d.filename, "path": str(d.path)} for d in downloads],
            "timestamp": datetime.now().isoformat()
        }

        summary_path = artifacts_dir / "test_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\nTest complete! Artifacts saved to: {artifacts_dir}")
        print(f"Downloads: {len(downloads)} files")

        # Keep browser open briefly for inspection
        time.sleep(5)


# =============================================================================
# STANDALONE RUNNER
# =============================================================================

def run_interactive_test(test_name: str = "test0_schools", headless: bool = False):
    """
    Run a single test interactively.

    This can be called directly for debugging:
        python tests/e2e/test_full_descent_ascent.py
    """
    config = TEST_CONFIGS.get(test_name)
    if not config:
        print(f"Unknown test: {test_name}")
        print(f"Available tests: {list(TEST_CONFIGS.keys())}")
        return

    # Create artifacts dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    artifacts_dir = ARTIFACTS_BASE_DIR / f"interactive_{timestamp}_{test_name}"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Interactive Test: {test_name}")
    print(f"Artifacts: {artifacts_dir}")
    print(f"{'='*60}\n")

    # Check if streamlit is already running
    import urllib.request
    url = "http://localhost:8501"

    try:
        urllib.request.urlopen(url, timeout=2)
        print("Streamlit app is already running")
    except:
        print("Starting Streamlit app...")
        subprocess.Popen(
            ["streamlit", "run", "intuitiveness/streamlit_app.py"],
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        time.sleep(10)

    # Run with Playwright
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless, slow_mo=300)
        context = browser.new_context(accept_downloads=True)
        page = context.new_page()

        downloads = []

        def handle_download(download):
            filename = download.suggested_filename
            save_path = artifacts_dir / filename
            download.save_as(str(save_path))
            downloads.append(filename)
            print(f"  Downloaded: {filename}")

        page.on("download", handle_download)

        try:
            page.goto(url)
            page.wait_for_load_state("networkidle")
            time.sleep(3)

            print("\n[1] Uploading files...")
            file_paths = [config["data_dir"] / f for f in config["files"]]
            upload_files(page, file_paths)
            take_screenshot(page, artifacts_dir, "01_uploaded")

            print("[2] Waiting for AI analysis (up to 30s)...")
            time.sleep(15)
            take_screenshot(page, artifacts_dir, "02_analyzed")

            print("[3] Navigating wizard...")
            # Try to click through wizard
            for i in range(3):
                scroll_to_bottom(page)
                time.sleep(1)
                if click_button_safe(page, "Continue"):
                    print(f"  Step {i+1} done")
                elif click_button_safe(page, "Confirm"):
                    print(f"  Wizard complete")
                    break
                take_screenshot(page, artifacts_dir, f"03_wizard_{i}")
                time.sleep(2)

            print("[4] Switching to Free mode...")
            switch_to_free_mode(page)
            take_screenshot(page, artifacts_dir, "04_free_mode")

            print("[5] Manual exploration - browser will stay open...")
            print("    Navigate through descent/ascent manually")
            print("    Close browser when done")

            # Keep browser open for manual interaction
            input("\nPress Enter to close browser and save artifacts...")

        except Exception as e:
            print(f"Error: {e}")
            take_screenshot(page, artifacts_dir, "error")
        finally:
            take_screenshot(page, artifacts_dir, "final")

            # Save summary
            summary = {
                "test": test_name,
                "config": config["description"],
                "downloads": downloads,
                "timestamp": datetime.now().isoformat()
            }
            with open(artifacts_dir / "summary.json", 'w') as f:
                json.dump(summary, f, indent=2)

            context.close()
            browser.close()

    print(f"\nArtifacts saved to: {artifacts_dir}")
    return artifacts_dir


if __name__ == "__main__":
    import sys

    test_name = sys.argv[1] if len(sys.argv) > 1 else "test0_schools"
    headless = "--headless" in sys.argv

    run_interactive_test(test_name, headless)
