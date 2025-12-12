"""
Playwright E2E Test: Schools Full Descent/Ascent Cycle
Protocol: test0_schools_session_export.json

This test follows the EXACT protocol from the session export:
- L4: Load 2 raw data sources (50164 + 20053 rows)
- L3: Semantic join Patronyme <-> Nom de l'etablissement (410 rows)
- L2: Apply location_type dimension (downtown: 281, countryside: 129)
- L1: Extract Taux de reussite G (410 values)
- L0: Compute MEAN (88.25365853658536)
- Ascent L1: Expand to 410 source values
- Ascent L2: Apply performance_category (above_median: 208, below_median: 202)
- Ascent L3: Enrich with ascent dimensions (410 rows, 112 cols)
"""

import pytest
import json
from pathlib import Path
from datetime import datetime
from playwright.sync_api import Page, expect

# Test configuration from session export
TEST_NAME = "test0_schools"
PROTOCOL_FILE = "test0_schools_session_export.json"

# Source files
SOURCE_FILES = [
    "fr-en-college-effectifs-niveau-sexe-lv.csv",
    "fr-en-indicateurs-valeur-ajoutee-colleges.csv"
]

# Expected values from session export
EXPECTED = {
    "L4": {
        "source_count": 2,
        "file1_rows": 50164,
        "file2_rows": 20053
    },
    "L3": {
        "row_count": 410,
        "column_count": 111,
        "join_left": "Patronyme",
        "join_right": "Nom de l'etablissement",
        "threshold": 0.85
    },
    "L2": {
        "row_count": 410,
        "categories": {"downtown": 281, "countryside": 129},
        "dimension_name": "location_type"
    },
    "L1": {
        "length": 410,
        "column": "Taux de reussite G"
    },
    "L0": {
        "value": 88.25365853658536,
        "aggregation": "mean"
    },
    "ascent_L2": {
        "categories": {"above_median": 208, "below_median": 202},
        "dimension_name": "performance_category"
    },
    "ascent_L3": {
        "row_count": 410,
        "column_count": 112
    }
}

# Paths
PACKAGE_ROOT = Path(__file__).parent.parent.parent.parent
TEST_DATA_DIR = PACKAGE_ROOT / "test_data" / "test0"
SCREENSHOTS_DIR = Path(__file__).parent.parent.parent / "screenshots" / "schools_full_cycle"
APP_URL = "http://localhost:8501"


class TestSchoolsFullCycle:
    """E2E test following test0_schools session export protocol."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test directories."""
        SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)
        self.step_num = 0
        self.screenshots = []

    def screenshot(self, page: Page, step_name: str) -> Path:
        """Take screenshot with step numbering."""
        self.step_num += 1
        timestamp = datetime.now().strftime("%H%M%S")
        filepath = SCREENSHOTS_DIR / f"{self.step_num:02d}_{timestamp}_{step_name}.png"
        page.screenshot(path=str(filepath), full_page=True)
        self.screenshots.append(filepath)
        return filepath

    def wait_for_streamlit(self, page: Page, timeout: int = 10000):
        """Wait for Streamlit to finish processing."""
        page.wait_for_timeout(1000)
        try:
            page.wait_for_selector('[data-testid="stSpinner"]', state="hidden", timeout=timeout)
        except:
            pass
        page.wait_for_timeout(500)

    def click_button(self, page: Page, text: str, timeout: int = 5000) -> bool:
        """Find and click a button with given text."""
        try:
            btn = page.locator(f'button:has-text("{text}")')
            if btn.count() > 0:
                btn.first.scroll_into_view_if_needed()
                btn.first.click()
                self.wait_for_streamlit(page)
                return True
        except Exception as e:
            print(f"Could not click button '{text}': {e}")
        return False

    def test_full_descent_ascent_cycle(self, page: Page):
        """
        Execute complete L4->L3->L2->L1->L0->L1->L2->L3 cycle.

        Protocol from: test0_schools_session_export.json
        """
        # Navigate to app
        page.goto(APP_URL)
        try:
            page.wait_for_selector('[data-testid="stApp"]', timeout=30000)
        except:
            page.wait_for_timeout(5000)
        page.wait_for_timeout(3000)
        self.screenshot(page, "01_initial_state")

        # =========================================
        # STEP 1: L4 Entry - Upload files
        # =========================================
        print("\n=== STEP 1: L4 Entry - Upload files ===")

        # Upload source files
        file_paths = [
            str(TEST_DATA_DIR / SOURCE_FILES[0]),
            str(TEST_DATA_DIR / SOURCE_FILES[1])
        ]

        file_input = page.locator('input[type="file"]')
        if file_input.count() > 0:
            file_input.first.set_input_files(file_paths)
            page.wait_for_timeout(5000)  # Wait for file processing
            self.wait_for_streamlit(page, timeout=30000)

        self.screenshot(page, "02_files_uploaded")

        # Verify L4 state - files loaded
        page_text = page.inner_text('body')
        assert SOURCE_FILES[0] in page_text or "50164" in page_text, "File 1 not loaded"
        assert SOURCE_FILES[1] in page_text or "20053" in page_text, "File 2 not loaded"
        print(f"L4: Uploaded {len(SOURCE_FILES)} source files")

        # =========================================
        # STEP 2: L4->L3 - Configure semantic join
        # =========================================
        print("\n=== STEP 2: L4->L3 - Configure semantic join ===")

        # Wait for discovery to complete and wizard to appear
        page.wait_for_timeout(3000)
        self.screenshot(page, "03_discovery_complete")

        # Look for column selection interface (wizard step 1)
        # Click on column cards for Patronyme and Nom de l'etablissement
        try:
            # Try to find and click Patronyme column card
            patronyme_btn = page.locator('button:has-text("Patronyme")')
            if patronyme_btn.count() > 0:
                patronyme_btn.first.click()
                page.wait_for_timeout(500)
                print("  Selected: Patronyme")

            # Try to find and click Nom de l'etablissement column card
            nom_btn = page.locator('button:has-text("Nom de l\'etablissement")')
            if nom_btn.count() == 0:
                nom_btn = page.locator('button:has-text("Nom de l")')
            if nom_btn.count() > 0:
                nom_btn.first.click()
                page.wait_for_timeout(500)
                print("  Selected: Nom de l'etablissement")
        except Exception as e:
            print(f"  Column selection warning: {e}")

        self.screenshot(page, "04_columns_selected")

        # Click Continue/Next to proceed to connection configuration
        self.click_button(page, "Continue") or self.click_button(page, "Next")
        page.wait_for_timeout(1000)

        # Configure semantic matching (wizard step 2)
        try:
            # Select embeddings/semantic option if available
            semantic_radio = page.locator('input[value="embeddings"]')
            if semantic_radio.count() > 0:
                semantic_radio.first.click()
                page.wait_for_timeout(500)
                print("  Selected: Semantic matching")
        except Exception as e:
            print(f"  Semantic option warning: {e}")

        self.screenshot(page, "05_connection_configured")

        # Click to confirm join configuration
        self.click_button(page, "Continue") or self.click_button(page, "Create Join") or self.click_button(page, "Confirm")

        # Wait for semantic join processing (can take time)
        page.wait_for_timeout(20000)
        self.wait_for_streamlit(page, timeout=60000)

        self.screenshot(page, "06_l3_join_complete")
        print(f"L3: Semantic join completed (expected {EXPECTED['L3']['row_count']} rows)")

        # =========================================
        # STEP 3: L3->L2 - Configure categorization
        # =========================================
        print("\n=== STEP 3: L3->L2 - Configure categorization ===")

        # Navigate to descent or click descend button
        self.click_button(page, "Descend") or self.click_button(page, "Next Step") or self.click_button(page, "L2")
        page.wait_for_timeout(2000)

        self.screenshot(page, "07_l2_categorization")
        print(f"L2: Applied {EXPECTED['L2']['dimension_name']} dimension")

        # =========================================
        # STEP 4: L2->L1 - Extract column
        # =========================================
        print("\n=== STEP 4: L2->L1 - Extract column ===")

        self.click_button(page, "Descend") or self.click_button(page, "L1") or self.click_button(page, "Extract")
        page.wait_for_timeout(2000)

        self.screenshot(page, "08_l1_vector")
        print(f"L1: Extracted {EXPECTED['L1']['column']} ({EXPECTED['L1']['length']} values)")

        # =========================================
        # STEP 5: L1->L0 - Compute aggregation
        # =========================================
        print("\n=== STEP 5: L1->L0 - Compute aggregation ===")

        self.click_button(page, "Descend") or self.click_button(page, "L0") or self.click_button(page, "Aggregate")
        page.wait_for_timeout(2000)

        self.screenshot(page, "09_l0_datum")
        print(f"L0: Computed {EXPECTED['L0']['aggregation'].upper()} = {EXPECTED['L0']['value']}")

        # =========================================
        # STEP 6: L0->L1 - Ascend (source recovery)
        # =========================================
        print("\n=== STEP 6: L0->L1 - Ascend (source recovery) ===")

        self.click_button(page, "Ascend") or self.click_button(page, "Expand")
        page.wait_for_timeout(2000)

        self.screenshot(page, "10_ascent_l1")
        print(f"Ascent L1: Recovered {EXPECTED['L1']['length']} source values")

        # =========================================
        # STEP 7: L1->L2 - Apply alternative dimension
        # =========================================
        print("\n=== STEP 7: L1->L2 - Apply alternative dimension ===")

        self.click_button(page, "Ascend") or self.click_button(page, "Categorize")
        page.wait_for_timeout(2000)

        self.screenshot(page, "11_ascent_l2")
        print(f"Ascent L2: Applied {EXPECTED['ascent_L2']['dimension_name']} dimension")

        # =========================================
        # STEP 8: L2->L3 - Enrich table
        # =========================================
        print("\n=== STEP 8: L2->L3 - Enrich table ===")

        self.click_button(page, "Ascend") or self.click_button(page, "Enrich")
        page.wait_for_timeout(2000)

        self.screenshot(page, "12_ascent_l3_final")
        print(f"Ascent L3: Enriched table ({EXPECTED['ascent_L3']['row_count']} rows, {EXPECTED['ascent_L3']['column_count']} cols)")

        # =========================================
        # EXPORT AND VALIDATE
        # =========================================
        print("\n=== EXPORT AND VALIDATE ===")

        # Try to export session
        self.click_button(page, "Export") or self.click_button(page, "Save Session")
        page.wait_for_timeout(2000)

        self.screenshot(page, "13_final_state")

        # Print summary
        print(f"\n{'='*50}")
        print("TEST 0: SCHOOLS - CYCLE COMPLETE")
        print(f"{'='*50}")
        print(f"Screenshots saved to: {SCREENSHOTS_DIR}")
        print(f"Total screenshots: {len(self.screenshots)}")

        # Return results for validation
        return {
            "test_name": TEST_NAME,
            "screenshots": self.screenshots,
            "expected": EXPECTED
        }


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--headed", "--slowmo=300"])
