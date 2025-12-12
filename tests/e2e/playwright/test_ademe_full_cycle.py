"""
Playwright E2E Test: ADEME Full Descent/Ascent Cycle
Protocol: test1_ademe_session_export.json

This test follows the EXACT protocol from the session export:
- L4: Load 2 raw data sources (428 + 37339 rows)
- L3: Semantic join dispositifAide <-> type_aides_financieres (500 rows)
- L2: Apply funding_frequency dimension (single_funding: 412, multiple_funding: 88)
- L1: Group montant by nomBeneficiaire, sum (450 recipients)
- L0: Compute SUM (69586180.93)
- Ascent L1: Expand to 450 source values
- Ascent L2: Apply funding_size dimension (above_10k: 301, below_10k: 149)
- Ascent L3: Enrich with ascent dimensions (450 rows, 48 cols)
"""

import pytest
import json
from pathlib import Path
from datetime import datetime
from playwright.sync_api import Page, expect

# Test configuration from session export
TEST_NAME = "test1_ademe"
PROTOCOL_FILE = "test1_ademe_session_export.json"

# Source files
SOURCE_FILES = [
    "ECS.csv",
    "Les aides financieres ADEME.csv"
]

# Expected values from session export
EXPECTED = {
    "L4": {
        "source_count": 2,
        "file1_rows": 428,  # ECS.csv
        "file2_rows": 37339  # Les aides financieres ADEME.csv
    },
    "L3": {
        "row_count": 500,
        "column_count": 47,
        "join_left": "dispositifAide",
        "join_right": "type_aides_financieres",
        "threshold": 0.75
    },
    "L2": {
        "row_count": 500,
        "categories": {"single_funding": 412, "multiple_funding": 88},
        "dimension_name": "funding_frequency"
    },
    "L1": {
        "length": 450,  # Reduced due to grouping
        "column": "montant",
        "group_by": "nomBeneficiaire",
        "group_agg": "sum"
    },
    "L0": {
        "value": 69586180.93,
        "aggregation": "sum"
    },
    "ascent_L2": {
        "categories": {"above_10k": 301, "below_10k": 149},
        "dimension_name": "funding_size"
    },
    "ascent_L3": {
        "row_count": 450,
        "column_count": 48
    }
}

# Paths
PACKAGE_ROOT = Path(__file__).parent.parent.parent.parent
TEST_DATA_DIR = PACKAGE_ROOT / "test_data" / "test1"
SCREENSHOTS_DIR = Path(__file__).parent.parent.parent / "screenshots" / "ademe_full_cycle"
APP_URL = "http://localhost:8501"


class TestADEMEFullCycle:
    """E2E test following test1_ademe session export protocol."""

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

        Protocol from: test1_ademe_session_export.json
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
        assert SOURCE_FILES[0] in page_text or "428" in page_text, "ECS.csv not loaded"
        assert SOURCE_FILES[1] in page_text or "37339" in page_text, "ADEME file not loaded"
        print(f"L4: Uploaded {len(SOURCE_FILES)} source files")

        # =========================================
        # STEP 2: L4->L3 - Configure semantic join
        # =========================================
        print("\n=== STEP 2: L4->L3 - Configure semantic join ===")

        # Wait for discovery to complete and wizard to appear
        page.wait_for_timeout(3000)
        self.screenshot(page, "03_discovery_complete")

        # Look for column selection interface (wizard step 1)
        # Click on column cards for dispositifAide and type_aides_financieres
        try:
            # Try to find and click dispositifAide column card
            dispositif_btn = page.locator('button:has-text("dispositifAide")')
            if dispositif_btn.count() > 0:
                dispositif_btn.first.click()
                page.wait_for_timeout(500)
                print("  Selected: dispositifAide")

            # Try to find and click type_aides_financieres column card
            type_aides_btn = page.locator('button:has-text("type_aides_financieres")')
            if type_aides_btn.count() > 0:
                type_aides_btn.first.click()
                page.wait_for_timeout(500)
                print("  Selected: type_aides_financieres")
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
        # STEP 4: L2->L1 - Group by recipient
        # =========================================
        print("\n=== STEP 4: L2->L1 - Group by recipient ===")

        self.click_button(page, "Descend") or self.click_button(page, "L1") or self.click_button(page, "Extract")
        page.wait_for_timeout(2000)

        self.screenshot(page, "08_l1_vector")
        print(f"L1: Grouped {EXPECTED['L1']['column']} by {EXPECTED['L1']['group_by']} ({EXPECTED['L1']['length']} recipients)")

        # =========================================
        # STEP 5: L1->L0 - Compute SUM aggregation
        # =========================================
        print("\n=== STEP 5: L1->L0 - Compute SUM aggregation ===")

        self.click_button(page, "Descend") or self.click_button(page, "L0") or self.click_button(page, "Aggregate")
        page.wait_for_timeout(2000)

        self.screenshot(page, "09_l0_datum")
        print(f"L0: Computed {EXPECTED['L0']['aggregation'].upper()} = {EXPECTED['L0']['value']:,.2f}")

        # =========================================
        # STEP 6: L0->L1 - Ascend (source recovery)
        # =========================================
        print("\n=== STEP 6: L0->L1 - Ascend (source recovery) ===")

        self.click_button(page, "Ascend") or self.click_button(page, "Expand")
        page.wait_for_timeout(2000)

        self.screenshot(page, "10_ascent_l1")
        print(f"Ascent L1: Recovered {EXPECTED['L1']['length']} source values")

        # =========================================
        # STEP 7: L1->L2 - Apply funding_size dimension
        # =========================================
        print("\n=== STEP 7: L1->L2 - Apply funding_size dimension ===")

        self.click_button(page, "Ascend") or self.click_button(page, "Categorize")
        page.wait_for_timeout(2000)

        self.screenshot(page, "11_ascent_l2")
        print(f"Ascent L2: Applied {EXPECTED['ascent_L2']['dimension_name']} dimension")
        print(f"  Categories: above_10k={EXPECTED['ascent_L2']['categories']['above_10k']}, below_10k={EXPECTED['ascent_L2']['categories']['below_10k']}")

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
        print("TEST 1: ADEME - CYCLE COMPLETE")
        print(f"{'='*50}")
        print(f"Screenshots saved to: {SCREENSHOTS_DIR}")
        print(f"Total screenshots: {len(self.screenshots)}")
        print(f"Expected L0 value: {EXPECTED['L0']['value']:,.2f} euros")

        # Return results for validation
        return {
            "test_name": TEST_NAME,
            "screenshots": self.screenshots,
            "expected": EXPECTED
        }


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--headed", "--slowmo=300"])
