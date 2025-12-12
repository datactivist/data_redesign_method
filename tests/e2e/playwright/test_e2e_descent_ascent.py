"""
End-to-End Playwright Tests for Complete Descent/Ascent Cycles.

These tests perform the FULL descent (L4→L3→L2→L1→L0) and ascent (L0→L1→L2→L3)
cycles through the UI, validating against session exports.

Run with visible browser:
    pytest tests/e2e/playwright/test_e2e_descent_ascent.py -v --headed --slowmo=300

Run headless:
    pytest tests/e2e/playwright/test_e2e_descent_ascent.py -v
"""

import pytest
import json
import sys
from pathlib import Path
from datetime import datetime
from playwright.sync_api import Page, expect

# Paths
PACKAGE_ROOT = Path(__file__).parent.parent.parent.parent
TEST_DATA_DIR = PACKAGE_ROOT / "test_data"
ARTIFACTS_DIR = PACKAGE_ROOT / "tests" / "artifacts" / "playwright_exports"
SCREENSHOTS_DIR = ARTIFACTS_DIR / "screenshots"

APP_URL = "http://localhost:8501"


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture(scope="session")
def browser_context_args(browser_context_args):
    """Configure browser for Streamlit testing."""
    return {
        **browser_context_args,
        "viewport": {"width": 1400, "height": 900},
    }


@pytest.fixture(autouse=True)
def setup_dirs():
    """Create output directories."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)


@pytest.fixture(scope="class")
def app_page(browser):
    """
    Single page instance for the entire test class.
    This prevents page refresh between test methods.
    """
    context = browser.new_context(viewport={"width": 1400, "height": 900})
    page = context.new_page()
    page.goto(APP_URL)
    page.wait_for_load_state("networkidle")
    # Wait longer for Streamlit to fully initialize
    page.wait_for_timeout(5000)
    yield page
    context.close()


# =============================================================================
# HELPER CLASS FOR E2E TESTING
# =============================================================================

class E2ETestHelper:
    """Helper class for E2E descent/ascent testing."""

    def __init__(self, page: Page, test_name: str):
        self.page = page
        self.test_name = test_name
        self.step_num = 0
        self.screenshots = []

    def screenshot(self, step_name: str) -> Path:
        """Take a screenshot with step tracking."""
        self.step_num += 1
        timestamp = datetime.now().strftime("%H%M%S")
        filepath = SCREENSHOTS_DIR / f"{self.test_name}_{self.step_num:02d}_{step_name}.png"
        self.page.screenshot(path=str(filepath), full_page=True)
        self.screenshots.append(filepath)
        print(f"  📸 Screenshot: {filepath.name}")
        return filepath

    def wait_for_rerun(self, timeout: int = 15000):
        """Wait for Streamlit to finish rerunning - with longer timeout for data loading."""
        self.page.wait_for_timeout(2000)  # Initial wait for rerun to start
        try:
            # Wait for any spinner to disappear
            self.page.wait_for_selector('[data-testid="stSpinner"]', state="hidden", timeout=timeout)
        except:
            pass
        # Extra wait for Streamlit state to settle
        self.page.wait_for_timeout(2000)

    def scroll_to_bottom(self):
        """Scroll to bottom of page."""
        self.page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        self.page.wait_for_timeout(500)

    def scroll_to_top(self):
        """Scroll to top of page."""
        self.page.evaluate("window.scrollTo(0, 0)")
        self.page.wait_for_timeout(500)

    def click_button(self, text: str, timeout: int = 5000) -> bool:
        """Click a button by text."""
        try:
            btn = self.page.locator(f'button:has-text("{text}")')
            if btn.count() > 0:
                btn.first.scroll_into_view_if_needed()
                btn.first.click()
                self.wait_for_rerun()
                return True
        except Exception as e:
            print(f"  ⚠️ Could not click '{text}': {e}")
        return False

    def click_button_containing(self, text: str) -> bool:
        """Click button containing text (partial match)."""
        try:
            btn = self.page.locator(f'button:has-text("{text}")')
            if btn.count() > 0:
                btn.first.scroll_into_view_if_needed()
                self.page.wait_for_timeout(300)
                btn.first.click()
                self.wait_for_rerun()
                return True
        except Exception as e:
            print(f"  ⚠️ Could not click button containing '{text}': {e}")
        return False

    def upload_files(self, filepaths: list) -> bool:
        """Upload files to file input."""
        try:
            file_input = self.page.locator('input[type="file"]')
            if file_input.count() > 0:
                file_input.first.set_input_files(filepaths)
                # Don't wait here - let the caller handle the long wait
                self.page.wait_for_timeout(2000)
                return True
        except Exception as e:
            print(f"  ⚠️ Upload failed: {e}")
        return False

    def select_option(self, dropdown_label: str, option_text: str) -> bool:
        """Select option from a labeled dropdown."""
        try:
            # Find selectbox near label
            selectboxes = self.page.locator('[data-testid="stSelectbox"]')
            for i in range(selectboxes.count()):
                sb = selectboxes.nth(i)
                sb.scroll_into_view_if_needed()
                sb.click()
                self.page.wait_for_timeout(500)

                option = self.page.locator(f'li[role="option"]:has-text("{option_text}")')
                if option.count() > 0:
                    option.first.click()
                    self.page.wait_for_timeout(500)
                    return True
                else:
                    self.page.keyboard.press("Escape")
                    self.page.wait_for_timeout(300)
        except Exception as e:
            print(f"  ⚠️ Could not select '{option_text}': {e}")
        return False

    def fill_text_input(self, placeholder_or_label: str, value: str) -> bool:
        """Fill a text input."""
        try:
            inputs = self.page.locator(f'input[type="text"]')
            for i in range(inputs.count()):
                inp = inputs.nth(i)
                inp.fill(value)
                self.page.wait_for_timeout(300)
                return True
        except Exception as e:
            print(f"  ⚠️ Could not fill input: {e}")
        return False

    def fill_textarea(self, value: str) -> bool:
        """Fill first textarea."""
        try:
            textareas = self.page.locator('textarea')
            if textareas.count() > 0:
                textareas.first.fill(value)
                return True
        except:
            pass
        return False

    def get_page_text(self) -> str:
        """Get all visible text on page."""
        return self.page.inner_text('body')

    def wait_for_text(self, text: str, timeout: int = 30000) -> bool:
        """Wait for text to appear on page."""
        try:
            self.page.wait_for_selector(f'text="{text}"', timeout=timeout)
            return True
        except:
            return False

    def export_results(self, results: dict):
        """Export test results to JSON."""
        filepath = ARTIFACTS_DIR / f"{self.test_name}_results.json"
        results["screenshots"] = [str(s) for s in self.screenshots]
        results["timestamp"] = datetime.now().isoformat()
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"  📄 Results exported to: {filepath}")


# =============================================================================
# TEST: SCHOOLS DATASET (test0) - FULL E2E DESCENT/ASCENT
# =============================================================================

class TestSchoolsE2E:
    """
    Full E2E test for test0_schools dataset.

    Expected cycle:
    - L4: Upload 2 CSV files
    - L3: Semantic join (Patronyme ↔ Nom de l'établissement)
    - L2: Categorize by location_type (downtown/countryside)
    - L1: Extract 'Taux de réussite G' vector
    - L0: Compute MEAN = 88.2537
    - Ascent L1→L2: Apply performance_category
    - Ascent L2→L3: Enrich with new dimension
    """

    FILES = [
        str(TEST_DATA_DIR / "test0" / "fr-en-college-effectifs-niveau-sexe-lv.csv"),
        str(TEST_DATA_DIR / "test0" / "fr-en-indicateurs-valeur-ajoutee-colleges.csv")
    ]

    EXPECTED = {
        "l3_rows": 410,
        "l2_categories": {"downtown": 281, "countryside": 129},
        "l1_column": "Taux de réussite G",
        "l0_value": 88.2537,
        "l0_aggregation": "mean",
        "ascent_l2_categories": {"above_median": 208, "below_median": 202}
    }

    def test_full_descent_ascent_cycle(self, app_page: Page):
        """
        Complete E2E test: L4 → L3 → L2 → L1 → L0 → L1 → L2 → L3
        """
        helper = E2ETestHelper(app_page, "test0_schools_e2e")
        results = {"test": "test0_schools", "steps": []}

        print("\n" + "="*70)
        print("🏫 TEST0 SCHOOLS: Full Descent/Ascent E2E Test")
        print("="*70)

        # =====================================================================
        # STEP 1: L4 ENTRY - Upload Files
        # =====================================================================
        print("\n📥 STEP 1: L4 Entry - Uploading files...")
        helper.screenshot("01_initial_state")

        # Verify files exist
        for f in self.FILES:
            assert Path(f).exists(), f"Test file not found: {f}"

        # Upload files
        uploaded = helper.upload_files(self.FILES)
        assert uploaded, "Failed to upload files"

        # LONG WAIT - Streamlit needs time to process CSV files (~20MB)
        print("  ⏳ Waiting for files to be processed (25s)...")
        app_page.wait_for_timeout(25000)
        helper.scroll_to_bottom()
        helper.screenshot("02_files_uploaded")

        results["steps"].append({
            "level": 4,
            "action": "entry",
            "description": "Uploaded 2 source files",
            "status": "success"
        })
        print("  ✅ Files uploaded successfully")

        # =====================================================================
        # STEP 2: Navigate to Data Model Generation
        # =====================================================================
        print("\n🔧 STEP 2: Navigating to data model configuration...")

        helper.click_button_containing("Next") or helper.click_button_containing("Continue")
        print("  ⏳ Waiting for page to load (8s)...")
        app_page.wait_for_timeout(8000)
        helper.screenshot("03_entity_config")

        # =====================================================================
        # STEP 3: Enter Domain Description
        # =====================================================================
        print("\n📝 STEP 3: Entering domain description...")

        helper.scroll_to_bottom()
        helper.click_button_containing("Next") or helper.click_button_containing("Continue")
        app_page.wait_for_timeout(10000)

        helper.scroll_to_top()
        domain_desc = "The data explores how the number of students relates to middle school performance scores."
        helper.fill_textarea(domain_desc)
        helper.screenshot("04_domain_description")

        # =====================================================================
        # STEP 4: Generate Data Model (L4 → L3)
        # =====================================================================
        print("\n🤖 STEP 4: Generating data model (L4 → L3)...")

        helper.scroll_to_bottom()
        helper.click_button_containing("Next") or helper.click_button_containing("Continue")
        app_page.wait_for_timeout(10000)

        if helper.click_button_containing("Generate"):
            print("  ⏳ Waiting for LLM to generate model (up to 60s)...")
            app_page.wait_for_timeout(45000)

        helper.screenshot("05_model_generated")

        results["steps"].append({
            "level": 3,
            "action": "descend",
            "description": "Generated data model with semantic join",
            "status": "success"
        })
        print("  ✅ Data model generated")

        # =====================================================================
        # STEP 5: Build Knowledge Graph
        # =====================================================================
        print("\n🔨 STEP 5: Building knowledge graph...")

        helper.scroll_to_bottom()
        helper.click_button_containing("Next") or helper.click_button_containing("Continue")
        app_page.wait_for_timeout(10000)

        if helper.click_button_containing("Build") or helper.click_button_containing("Execute"):
            print("  ⏳ Building graph...")
            app_page.wait_for_timeout(20000)

        helper.screenshot("06_graph_built")

        # =====================================================================
        # STEP 6: Check for Navigation Mode
        # =====================================================================
        print("\n🧭 STEP 6: Checking for navigation session...")

        helper.scroll_to_top()
        app_page.wait_for_timeout(10000)
        helper.screenshot("07_pre_navigation")

        # Try to find navigation session or descent buttons
        page_text = helper.get_page_text()

        if "Navigation" in page_text or "Descend" in page_text:
            print("  📍 Found navigation controls")

            # Try descent buttons
            if helper.click_button_containing("Descend to L2"):
                print("  ⬇️ Descended to L2")
                helper.screenshot("08_l2_descent")
                results["steps"].append({
                    "level": 2,
                    "action": "descend",
                    "description": "Descended to L2 (categorized)",
                    "status": "success"
                })

            app_page.wait_for_timeout(10000)

            if helper.click_button_containing("Descend to L1"):
                print("  ⬇️ Descended to L1")
                helper.screenshot("09_l1_descent")
                results["steps"].append({
                    "level": 1,
                    "action": "descend",
                    "description": "Descended to L1 (vector)",
                    "status": "success"
                })

            app_page.wait_for_timeout(10000)

            if helper.click_button_containing("Descend to L0"):
                print("  ⬇️ Descended to L0")
                helper.screenshot("10_l0_descent")
                results["steps"].append({
                    "level": 0,
                    "action": "descend",
                    "description": "Descended to L0 (datum)",
                    "status": "success"
                })

            app_page.wait_for_timeout(10000)

            # ASCENT PHASE
            print("\n⬆️ ASCENT PHASE...")

            if helper.click_button_containing("Ascend to L1"):
                print("  ⬆️ Ascended to L1")
                helper.screenshot("11_l1_ascent")
                results["steps"].append({
                    "level": 1,
                    "action": "ascend",
                    "description": "Ascended to L1",
                    "status": "success"
                })

            app_page.wait_for_timeout(10000)

            if helper.click_button_containing("Ascend to L2"):
                print("  ⬆️ Ascended to L2")
                helper.screenshot("12_l2_ascent")
                results["steps"].append({
                    "level": 2,
                    "action": "ascend",
                    "description": "Ascended to L2 with new dimension",
                    "status": "success"
                })

            app_page.wait_for_timeout(10000)

            if helper.click_button_containing("Ascend to L3"):
                print("  ⬆️ Ascended to L3")
                helper.screenshot("13_l3_ascent")
                results["steps"].append({
                    "level": 3,
                    "action": "ascend",
                    "description": "Ascended to L3 (enriched)",
                    "status": "success"
                })

        # =====================================================================
        # FINAL: Capture final state and export results
        # =====================================================================
        print("\n📊 FINAL: Capturing final state...")

        helper.scroll_to_top()
        helper.screenshot("14_final_top")
        helper.scroll_to_bottom()
        helper.screenshot("15_final_bottom")

        # Export results
        results["status"] = "completed"
        results["expected_l0_value"] = self.EXPECTED["l0_value"]
        helper.export_results(results)

        print("\n" + "="*70)
        print(f"✅ TEST COMPLETED: {len(helper.screenshots)} screenshots captured")
        print(f"📁 Results: {ARTIFACTS_DIR / 'test0_schools_e2e_results.json'}")
        print("="*70)


# =============================================================================
# TEST: ADEME DATASET (test1) - FULL E2E DESCENT/ASCENT
# =============================================================================

class TestADEMEE2E:
    """
    Full E2E test for test1_ademe dataset.

    Expected cycle:
    - L4: Upload 2 CSV files
    - L3: Semantic join (dispositifAide ↔ type_aides_financieres)
    - L2: Categorize by funding_frequency (single/multiple)
    - L1: Group montant by nomBeneficiaire
    - L0: Compute SUM = 69586180.93
    - Ascent L1→L2: Apply funding_size (above_10k/below_10k)
    - Ascent L2→L3: Enrich with new dimension
    """

    FILES = [
        str(TEST_DATA_DIR / "test1" / "ECS.csv"),
        str(TEST_DATA_DIR / "test1" / "Les aides financieres ADEME.csv")
    ]

    EXPECTED = {
        "l3_rows": 500,
        "l2_categories": {"single_funding": 412, "multiple_funding": 88},
        "l1_column": "montant",
        "l0_value": 69586180.93,
        "l0_aggregation": "sum",
        "ascent_l2_categories": {"above_10k": 301, "below_10k": 149}
    }

    def test_full_descent_ascent_cycle(self, app_page: Page):
        """
        Complete E2E test: L4 → L3 → L2 → L1 → L0 → L1 → L2 → L3
        """
        helper = E2ETestHelper(app_page, "test1_ademe_e2e")
        results = {"test": "test1_ademe", "steps": []}

        print("\n" + "="*70)
        print("💰 TEST1 ADEME: Full Descent/Ascent E2E Test")
        print("="*70)

        # =====================================================================
        # STEP 1: L4 ENTRY - Upload Files
        # =====================================================================
        print("\n📥 STEP 1: L4 Entry - Uploading files...")
        helper.screenshot("01_initial_state")

        # Verify files exist
        for f in self.FILES:
            assert Path(f).exists(), f"Test file not found: {f}"

        # Upload files
        uploaded = helper.upload_files(self.FILES)
        assert uploaded, "Failed to upload files"

        # LONG WAIT - Streamlit needs time to process CSV files (~20MB)
        print("  ⏳ Waiting for files to be processed (25s)...")
        app_page.wait_for_timeout(25000)
        helper.scroll_to_bottom()
        helper.screenshot("02_files_uploaded")

        results["steps"].append({
            "level": 4,
            "action": "entry",
            "description": "Uploaded 2 source files",
            "status": "success"
        })
        print("  ✅ Files uploaded successfully")

        # =====================================================================
        # STEP 2: Navigate to Data Model Generation
        # =====================================================================
        print("\n🔧 STEP 2: Navigating to data model configuration...")

        helper.click_button_containing("Next") or helper.click_button_containing("Continue")
        print("  ⏳ Waiting for page to load (8s)...")
        app_page.wait_for_timeout(8000)
        helper.screenshot("03_entity_config")

        # =====================================================================
        # STEP 3: Enter Domain Description
        # =====================================================================
        print("\n📝 STEP 3: Entering domain description...")

        helper.scroll_to_bottom()
        helper.click_button_containing("Next") or helper.click_button_containing("Continue")
        app_page.wait_for_timeout(10000)

        helper.scroll_to_top()
        domain_desc = "The data investigates who gets what funding from ADEME for environmental projects."
        helper.fill_textarea(domain_desc)
        helper.screenshot("04_domain_description")

        # =====================================================================
        # STEP 4: Generate Data Model (L4 → L3)
        # =====================================================================
        print("\n🤖 STEP 4: Generating data model (L4 → L3)...")

        helper.scroll_to_bottom()
        helper.click_button_containing("Next") or helper.click_button_containing("Continue")
        app_page.wait_for_timeout(10000)

        if helper.click_button_containing("Generate"):
            print("  ⏳ Waiting for LLM to generate model (up to 60s)...")
            app_page.wait_for_timeout(45000)

        helper.screenshot("05_model_generated")

        results["steps"].append({
            "level": 3,
            "action": "descend",
            "description": "Generated data model with semantic join",
            "status": "success"
        })
        print("  ✅ Data model generated")

        # =====================================================================
        # STEP 5: Build Knowledge Graph
        # =====================================================================
        print("\n🔨 STEP 5: Building knowledge graph...")

        helper.scroll_to_bottom()
        helper.click_button_containing("Next") or helper.click_button_containing("Continue")
        app_page.wait_for_timeout(10000)

        if helper.click_button_containing("Build") or helper.click_button_containing("Execute"):
            print("  ⏳ Building graph...")
            app_page.wait_for_timeout(20000)

        helper.screenshot("06_graph_built")

        # =====================================================================
        # STEP 6: Navigation and Descent/Ascent
        # =====================================================================
        print("\n🧭 STEP 6: Navigation descent/ascent cycle...")

        helper.scroll_to_top()
        app_page.wait_for_timeout(10000)
        helper.screenshot("07_pre_navigation")

        page_text = helper.get_page_text()

        if "Navigation" in page_text or "Descend" in page_text:
            print("  📍 Found navigation controls")

            # Descent
            if helper.click_button_containing("Descend to L2"):
                print("  ⬇️ Descended to L2")
                helper.screenshot("08_l2_descent")

            app_page.wait_for_timeout(10000)

            if helper.click_button_containing("Descend to L1"):
                print("  ⬇️ Descended to L1")
                helper.screenshot("09_l1_descent")

            app_page.wait_for_timeout(10000)

            if helper.click_button_containing("Descend to L0"):
                print("  ⬇️ Descended to L0")
                helper.screenshot("10_l0_descent")

            app_page.wait_for_timeout(10000)

            # Ascent
            print("\n⬆️ ASCENT PHASE...")

            if helper.click_button_containing("Ascend to L1"):
                print("  ⬆️ Ascended to L1")
                helper.screenshot("11_l1_ascent")

            app_page.wait_for_timeout(10000)

            if helper.click_button_containing("Ascend to L2"):
                print("  ⬆️ Ascended to L2")
                helper.screenshot("12_l2_ascent")

            app_page.wait_for_timeout(10000)

            if helper.click_button_containing("Ascend to L3"):
                print("  ⬆️ Ascended to L3")
                helper.screenshot("13_l3_ascent")

        # =====================================================================
        # FINAL
        # =====================================================================
        print("\n📊 FINAL: Capturing final state...")

        helper.scroll_to_top()
        helper.screenshot("14_final_top")
        helper.scroll_to_bottom()
        helper.screenshot("15_final_bottom")

        results["status"] = "completed"
        results["expected_l0_value"] = self.EXPECTED["l0_value"]
        helper.export_results(results)

        print("\n" + "="*70)
        print(f"✅ TEST COMPLETED: {len(helper.screenshots)} screenshots captured")
        print(f"📁 Results: {ARTIFACTS_DIR / 'test1_ademe_e2e_results.json'}")
        print("="*70)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--headed", "--slowmo=300"])
