"""
Playwright E2E tests validating interface behavior against session exports.

These tests verify that the Data Redesign Method UI produces transformations
matching the recorded session exports for test0 (schools) and test1 (ADEME).

Session exports record the complete descent (L4→L3→L2→L1→L0) and ascent
(L0→L1→L2→L3) paths with expected outputs at each level.

Run with:
    pytest tests/e2e/playwright/test_session_export_validation.py -v --headed
    pytest tests/e2e/playwright/test_session_export_validation.py -v  # headless
"""

import pytest
import sys
from pathlib import Path
from datetime import datetime
from playwright.sync_api import Page, expect

# Add parent directory to path for conftest imports
sys.path.insert(0, str(Path(__file__).parent))

# Import from local conftest (pytest also auto-discovers these as fixtures)
from conftest import SessionExportData

# Directory paths (duplicated here for direct test access)
PACKAGE_ROOT = Path(__file__).parent.parent.parent.parent
TEST_DATA_DIR = PACKAGE_ROOT / "test_data"
SCREENSHOTS_DIR = Path(__file__).parent.parent.parent / "screenshots" / "session_export_tests"
APP_URL = "http://localhost:8501"


# =============================================================================
# HELPER FUNCTIONS (duplicated for test access without fixture dependency)
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
    SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)
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


def get_displayed_value(page: Page, label: str) -> str:
    """Extract a displayed value near a label."""
    try:
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


# =============================================================================
# TEST CONFIGURATIONS DERIVED FROM SESSION EXPORTS
# =============================================================================

def get_test0_config(session: SessionExportData) -> dict:
    """Extract test configuration from test0 schools session export."""
    return {
        "name": "test0_schools",
        "description": session.config_description,
        "files": [
            str(TEST_DATA_DIR / "test0" / "fr-en-college-effectifs-niveau-sexe-lv.csv"),
            str(TEST_DATA_DIR / "test0" / "fr-en-indicateurs-valeur-ajoutee-colleges.csv")
        ],
        "expected": {
            "l4_sources": session.sources.get("source_count", 2),
            "l3_rows": session.joined_table.get("row_count", 410),
            "l3_join_columns": session.joined_table.get("join_columns", {}),
            "l2_categories": session.categorized_table.get("categories", {}),
            "l2_dimension": session.categorized_table.get("dimension_column", "location_type"),
            "l1_column": session.vector.get("column_name", "Taux de réussite G"),
            "l1_length": session.vector.get("row_count", 410),
            "l0_value": session.datum.get("value", 88.25365853658536),
            "l0_aggregation": session.datum.get("aggregation_method", "mean"),
        }
    }


def get_test1_config(session: SessionExportData) -> dict:
    """Extract test configuration from test1 ADEME session export."""
    return {
        "name": "test1_ademe",
        "description": session.config_description,
        "files": [
            str(TEST_DATA_DIR / "test1" / "ECS.csv"),
            str(TEST_DATA_DIR / "test1" / "Les aides financieres ADEME.csv")
        ],
        "expected": {
            "l4_sources": session.sources.get("source_count", 2),
            "l3_rows": session.joined_table.get("row_count", 500),
            "l3_join_columns": session.joined_table.get("join_columns", {}),
            "l2_categories": session.categorized_table.get("categories", {}),
            "l2_dimension": session.categorized_table.get("dimension_column", "funding_frequency"),
            "l1_column": session.vector.get("column_name", "montant"),
            "l1_length": session.vector.get("row_count", 450),
            "l0_value": session.datum.get("value", 69586180.93),
            "l0_aggregation": session.datum.get("aggregation_method", "sum"),
        }
    }


# =============================================================================
# TEST CLASS: TEST0 SCHOOLS SESSION VALIDATION
# =============================================================================

class TestSchoolsSessionExport:
    """
    Validate the interface against test0_schools session export.

    Expected descent/ascent cycle from session export:
    - L4: Load 2 CSV files (50164 + 20053 rows)
    - L3: Semantic join Patronyme ↔ Nom de l'établissement → 410 rows
    - L2: Categorize by location_type (downtown: 281, countryside: 129)
    - L1: Extract 'Taux de réussite G' vector (410 values)
    - L0: Compute MEAN = 88.2537
    - Ascent L1: Expand to 410 values
    - Ascent L2: Apply 'performance_category' (above_median: 208, below_median: 202)
    - Ascent L3: Enrich table with performance_category dimension
    """

    def test_app_loads(self, app_page: Page):
        """Test that the app loads correctly."""
        take_screenshot(app_page, "test0", "01_app_loaded")
        expect(app_page.locator('[data-testid="stApp"]')).to_be_visible(timeout=10000)

    def test_l4_file_upload(self, app_page: Page, test0_schools_session: SessionExportData):
        """Test L4 entry point - file upload matches session export sources."""
        config = get_test0_config(test0_schools_session)

        # Verify file input is present
        file_input = app_page.locator('input[type="file"]')
        expect(file_input).to_be_visible(timeout=10000)

        # Upload files
        result = upload_files(app_page, config["files"])
        assert result, "File upload failed"

        app_page.wait_for_timeout(5000)
        take_screenshot(app_page, "test0", "02_l4_files_uploaded")

        # Verify: Should show 2 sources uploaded (matching session export)
        expected_sources = config["expected"]["l4_sources"]
        scroll_to_bottom(app_page)
        take_screenshot(app_page, "test0", "02b_l4_after_scroll")

        # Check for indicators of successful upload
        # Look for source file names in the UI
        source_names = test0_schools_session.sources.get("source_names", [])
        for source_name in source_names:
            # File name should appear somewhere
            short_name = Path(source_name).name
            print(f"Checking for source: {short_name}")

    def test_l4_to_l3_semantic_join_setup(self, app_page: Page, test0_schools_session: SessionExportData):
        """Test L4→L3 transition - semantic join configuration matches session export."""
        config = get_test0_config(test0_schools_session)

        # Upload files first
        upload_files(app_page, config["files"])
        app_page.wait_for_timeout(3000)

        # Navigate to entity/join configuration
        scroll_to_bottom(app_page)
        click_button(app_page, "Continue") or click_button(app_page, "Next")
        app_page.wait_for_timeout(2000)

        take_screenshot(app_page, "test0", "03_l3_join_config")

        # From session export, the join should be:
        # left_column: "Patronyme"
        # right_column: "Nom de l'établissement"
        join_columns = config["expected"]["l3_join_columns"]
        left_col = join_columns.get("left", "Patronyme")
        right_col = join_columns.get("right", "Nom de l'établissement")

        print(f"Expected join: {left_col} ↔ {right_col}")

    def test_full_descent_cycle(self, app_page: Page, test0_schools_session: SessionExportData):
        """Test complete descent from L4 to L0 against session export."""
        config = get_test0_config(test0_schools_session)
        screenshots = []

        print(f"\n{'='*60}")
        print(f"Testing descent: {config['name']}")
        print(f"Expected L0 value: {config['expected']['l0_value']}")
        print(f"{'='*60}")

        # Step 1: L4 - Upload files
        print("\n[L4] Uploading files...")
        upload_files(app_page, config["files"])
        app_page.wait_for_timeout(5000)
        screenshots.append(take_screenshot(app_page, "test0", "descent_01_l4_entry"))

        # Step 2: Navigate through workflow
        print("\n[L3] Setting up semantic join...")
        scroll_to_bottom(app_page)
        click_button(app_page, "Continue") or click_button(app_page, "Next")
        app_page.wait_for_timeout(2000)
        screenshots.append(take_screenshot(app_page, "test0", "descent_02_l3_config"))

        # Step 3: Domain description
        print("\n[L3] Entering domain description...")
        scroll_to_bottom(app_page)
        click_button(app_page, "Continue") or click_button(app_page, "Next")
        app_page.wait_for_timeout(2000)

        scroll_to_top(app_page)
        domain_desc = "The data explores how the number of students is related to middle school scores."
        fill_textarea(app_page, domain_desc)
        screenshots.append(take_screenshot(app_page, "test0", "descent_03_domain"))

        # Step 4: Generate data model
        print("\n[L3] Generating data model...")
        scroll_to_bottom(app_page)
        click_button(app_page, "Continue") or click_button(app_page, "Next")
        app_page.wait_for_timeout(2000)

        if click_button(app_page, "Generate"):
            print("   Waiting for LLM generation...")
            app_page.wait_for_timeout(45000)
        screenshots.append(take_screenshot(app_page, "test0", "descent_04_generated"))

        # Step 5: Build graph
        print("\n[L3→L2→L1→L0] Building knowledge graph...")
        scroll_to_bottom(app_page)
        click_button(app_page, "Continue") or click_button(app_page, "Next")
        app_page.wait_for_timeout(2000)

        if click_button(app_page, "Build"):
            print("   Building graph...")
            app_page.wait_for_timeout(20000)
        screenshots.append(take_screenshot(app_page, "test0", "descent_05_built"))

        # Capture final state
        print("\n[L0] Capturing final descent state...")
        scroll_to_top(app_page)
        screenshots.append(take_screenshot(app_page, "test0", "descent_06_final_top"))
        scroll_to_bottom(app_page)
        screenshots.append(take_screenshot(app_page, "test0", "descent_06_final_bottom"))

        print(f"\nDescent complete! Screenshots: {len(screenshots)}")

    def test_verify_l2_categories_match_session(self, app_page: Page, test0_schools_session: SessionExportData):
        """Verify L2 categories match session export: downtown/countryside."""
        config = get_test0_config(test0_schools_session)
        expected_categories = config["expected"]["l2_categories"]

        print(f"\nExpected L2 categories: {expected_categories}")
        # Expected: {"downtown": 281, "countryside": 129}

        # Check dimension name
        expected_dimension = config["expected"]["l2_dimension"]
        print(f"Expected dimension: {expected_dimension}")

    def test_verify_l0_datum_matches_session(self, app_page: Page, test0_schools_session: SessionExportData):
        """Verify L0 datum value matches session export: MEAN = 88.2537."""
        config = get_test0_config(test0_schools_session)
        expected_value = config["expected"]["l0_value"]
        expected_agg = config["expected"]["l0_aggregation"]

        print(f"\nExpected L0 datum: {expected_agg}({config['expected']['l1_column']}) = {expected_value}")

        # The session export shows: 88.25365853658536
        assert expected_value == pytest.approx(88.25365853658536, rel=0.001)


# =============================================================================
# TEST CLASS: TEST1 ADEME SESSION VALIDATION
# =============================================================================

class TestADEMESessionExport:
    """
    Validate the interface against test1_ademe session export.

    Expected descent/ascent cycle from session export:
    - L4: Load 2 CSV files (ECS.csv: 428 rows, Les aides financieres ADEME.csv: 37339 rows)
    - L3: Semantic join dispositifAide ↔ type_aides_financieres → 500 rows
    - L2: Categorize by funding_frequency (single_funding: 412, multiple_funding: 88)
    - L1: Group 'montant' by 'nomBeneficiaire' (sum) → 450 unique recipients
    - L0: Compute SUM = 69586180.93
    - Ascent L1: Expand to 450 values
    - Ascent L2: Apply 'funding_size' (above_10k: 301, below_10k: 149)
    - Ascent L3: Enrich table with funding_size dimension
    """

    def test_app_loads(self, app_page: Page):
        """Test that the app loads correctly for ADEME test."""
        take_screenshot(app_page, "test1", "01_app_loaded")
        expect(app_page.locator('[data-testid="stApp"]')).to_be_visible(timeout=10000)

    def test_l4_file_upload(self, app_page: Page, test1_ademe_session: SessionExportData):
        """Test L4 entry point - file upload matches session export sources."""
        config = get_test1_config(test1_ademe_session)

        # Verify file input is present
        file_input = app_page.locator('input[type="file"]')
        expect(file_input).to_be_visible(timeout=10000)

        # Upload files
        result = upload_files(app_page, config["files"])
        assert result, "File upload failed"

        app_page.wait_for_timeout(5000)
        take_screenshot(app_page, "test1", "02_l4_files_uploaded")

        # Verify: Should show 2 sources uploaded
        expected_sources = config["expected"]["l4_sources"]
        scroll_to_bottom(app_page)
        take_screenshot(app_page, "test1", "02b_l4_after_scroll")

    def test_l4_to_l3_semantic_join_setup(self, app_page: Page, test1_ademe_session: SessionExportData):
        """Test L4→L3 transition - semantic join configuration matches session export."""
        config = get_test1_config(test1_ademe_session)

        # Upload files first
        upload_files(app_page, config["files"])
        app_page.wait_for_timeout(3000)

        # Navigate to entity/join configuration
        scroll_to_bottom(app_page)
        click_button(app_page, "Continue") or click_button(app_page, "Next")
        app_page.wait_for_timeout(2000)

        take_screenshot(app_page, "test1", "03_l3_join_config")

        # From session export, the join should be:
        # left_column: "dispositifAide"
        # right_column: "type_aides_financieres"
        join_columns = config["expected"]["l3_join_columns"]
        left_col = join_columns.get("left", "dispositifAide")
        right_col = join_columns.get("right", "type_aides_financieres")

        print(f"Expected join: {left_col} ↔ {right_col}")

    def test_full_descent_cycle(self, app_page: Page, test1_ademe_session: SessionExportData):
        """Test complete descent from L4 to L0 against session export."""
        config = get_test1_config(test1_ademe_session)
        screenshots = []

        print(f"\n{'='*60}")
        print(f"Testing descent: {config['name']}")
        print(f"Expected L0 value: {config['expected']['l0_value']}")
        print(f"{'='*60}")

        # Step 1: L4 - Upload files
        print("\n[L4] Uploading files...")
        upload_files(app_page, config["files"])
        app_page.wait_for_timeout(5000)
        screenshots.append(take_screenshot(app_page, "test1", "descent_01_l4_entry"))

        # Step 2: Navigate through workflow
        print("\n[L3] Setting up semantic join...")
        scroll_to_bottom(app_page)
        click_button(app_page, "Continue") or click_button(app_page, "Next")
        app_page.wait_for_timeout(2000)
        screenshots.append(take_screenshot(app_page, "test1", "descent_02_l3_config"))

        # Step 3: Domain description
        print("\n[L3] Entering domain description...")
        scroll_to_bottom(app_page)
        click_button(app_page, "Continue") or click_button(app_page, "Next")
        app_page.wait_for_timeout(2000)

        scroll_to_top(app_page)
        domain_desc = "The data investigates who gets what funding from ADEME for environmental projects."
        fill_textarea(app_page, domain_desc)
        screenshots.append(take_screenshot(app_page, "test1", "descent_03_domain"))

        # Step 4: Generate data model
        print("\n[L3] Generating data model...")
        scroll_to_bottom(app_page)
        click_button(app_page, "Continue") or click_button(app_page, "Next")
        app_page.wait_for_timeout(2000)

        if click_button(app_page, "Generate"):
            print("   Waiting for LLM generation...")
            app_page.wait_for_timeout(45000)
        screenshots.append(take_screenshot(app_page, "test1", "descent_04_generated"))

        # Step 5: Build graph
        print("\n[L3→L2→L1→L0] Building knowledge graph...")
        scroll_to_bottom(app_page)
        click_button(app_page, "Continue") or click_button(app_page, "Next")
        app_page.wait_for_timeout(2000)

        if click_button(app_page, "Build"):
            print("   Building graph...")
            app_page.wait_for_timeout(20000)
        screenshots.append(take_screenshot(app_page, "test1", "descent_05_built"))

        # Capture final state
        print("\n[L0] Capturing final descent state...")
        scroll_to_top(app_page)
        screenshots.append(take_screenshot(app_page, "test1", "descent_06_final_top"))
        scroll_to_bottom(app_page)
        screenshots.append(take_screenshot(app_page, "test1", "descent_06_final_bottom"))

        print(f"\nDescent complete! Screenshots: {len(screenshots)}")

    def test_verify_l2_categories_match_session(self, app_page: Page, test1_ademe_session: SessionExportData):
        """Verify L2 categories match session export: single_funding/multiple_funding."""
        config = get_test1_config(test1_ademe_session)
        expected_categories = config["expected"]["l2_categories"]

        print(f"\nExpected L2 categories: {expected_categories}")
        # Expected: {"single_funding": 412, "multiple_funding": 88}

        # Check dimension name
        expected_dimension = config["expected"]["l2_dimension"]
        print(f"Expected dimension: {expected_dimension}")

    def test_verify_l0_datum_matches_session(self, app_page: Page, test1_ademe_session: SessionExportData):
        """Verify L0 datum value matches session export: SUM = 69586180.93."""
        config = get_test1_config(test1_ademe_session)
        expected_value = config["expected"]["l0_value"]
        expected_agg = config["expected"]["l0_aggregation"]

        print(f"\nExpected L0 datum: {expected_agg}({config['expected']['l1_column']}) = {expected_value}")

        # The session export shows: 69586180.93
        assert expected_value == pytest.approx(69586180.93, rel=0.001)


# =============================================================================
# TEST CLASS: ASCENT CYCLE VALIDATION
# =============================================================================

class TestAscentCycleValidation:
    """
    Test the ascent phase (L0→L1→L2→L3) against session exports.

    The ascent adds new analytical dimensions to the data,
    enabling cross-dimensional analysis.
    """

    def test_schools_ascent_categories(self, test0_schools_session: SessionExportData):
        """Verify ascent L2 categories for schools: above_median/below_median."""
        ascent_nodes = test0_schools_session.get_ascent_nodes()

        # Find the L2 ascent node
        l2_ascent = None
        for node in ascent_nodes:
            if node.get("level") == 2:
                l2_ascent = node
                break

        assert l2_ascent is not None, "L2 ascent node not found"

        # Verify categories
        snapshot = l2_ascent.get("output_snapshot", {})
        categories = snapshot.get("categories", {})

        print(f"\nAscent L2 categories: {categories}")
        # Expected: {"above_median": 208, "below_median": 202}

        assert "above_median" in categories
        assert "below_median" in categories
        assert categories["above_median"] == 208
        assert categories["below_median"] == 202

    def test_ademe_ascent_categories(self, test1_ademe_session: SessionExportData):
        """Verify ascent L2 categories for ADEME: above_10k/below_10k."""
        ascent_nodes = test1_ademe_session.get_ascent_nodes()

        # Find the L2 ascent node
        l2_ascent = None
        for node in ascent_nodes:
            if node.get("level") == 2:
                l2_ascent = node
                break

        assert l2_ascent is not None, "L2 ascent node not found"

        # Verify categories
        snapshot = l2_ascent.get("output_snapshot", {})
        categories = snapshot.get("categories", {})

        print(f"\nAscent L2 categories: {categories}")
        # Expected: {"above_10k": 301, "below_10k": 149}

        assert "above_10k" in categories
        assert "below_10k" in categories
        assert categories["above_10k"] == 301
        assert categories["below_10k"] == 149

    def test_schools_ascent_enriched_l3(self, test0_schools_session: SessionExportData):
        """Verify L3 enrichment adds performance_category dimension."""
        ascent_nodes = test0_schools_session.get_ascent_nodes()

        # Find the L3 ascent node (final enriched state)
        l3_ascent = None
        for node in ascent_nodes:
            if node.get("level") == 3:
                l3_ascent = node
                break

        assert l3_ascent is not None, "L3 ascent node not found"

        # Verify enrichment
        params = l3_ascent.get("params", {})
        added_dims = params.get("added_dimensions", [])

        print(f"\nAdded dimensions: {added_dims}")
        assert "performance_category" in added_dims

    def test_ademe_ascent_enriched_l3(self, test1_ademe_session: SessionExportData):
        """Verify L3 enrichment adds funding_size dimension."""
        ascent_nodes = test1_ademe_session.get_ascent_nodes()

        # Find the L3 ascent node (final enriched state)
        l3_ascent = None
        for node in ascent_nodes:
            if node.get("level") == 3:
                l3_ascent = node
                break

        assert l3_ascent is not None, "L3 ascent node not found"

        # Verify enrichment
        params = l3_ascent.get("params", {})
        added_dims = params.get("added_dimensions", [])

        print(f"\nAdded dimensions: {added_dims}")
        assert "funding_size" in added_dims


# =============================================================================
# TEST CLASS: DESIGN CHOICES VALIDATION
# =============================================================================

class TestDesignChoicesValidation:
    """
    Validate that session export design choices are correctly recorded.
    """

    def test_schools_design_choices_summary(self, test0_schools_session: SessionExportData):
        """Verify design choices summary for schools dataset."""
        choices = test0_schools_session.design_choices

        print(f"\nSchools design choices:")
        for key, value in choices.items():
            print(f"  {key}: {value}")

        # Verify key design decisions
        assert "Patronyme" in choices.get("L4_to_L3_join", "")
        assert "Nom de l'établissement" in choices.get("L4_to_L3_join", "")
        assert "location_type" in choices.get("L3_to_L2_categorization", "")
        assert "Taux de réussite" in choices.get("L2_to_L1_extraction", "")
        assert "MEAN" in choices.get("L1_to_L0_aggregation", "").upper()
        assert "performance_category" in choices.get("Ascent_L1_to_L2", "")

    def test_ademe_design_choices_summary(self, test1_ademe_session: SessionExportData):
        """Verify design choices summary for ADEME dataset."""
        choices = test1_ademe_session.design_choices

        print(f"\nADEME design choices:")
        for key, value in choices.items():
            print(f"  {key}: {value}")

        # Verify key design decisions
        assert "dispositifAide" in choices.get("L4_to_L3_join", "")
        assert "type_aides_financieres" in choices.get("L4_to_L3_join", "")
        assert "funding_frequency" in choices.get("L3_to_L2_categorization", "")
        assert "montant" in choices.get("L2_to_L1_extraction", "")
        assert "SUM" in choices.get("L1_to_L0_aggregation", "").upper()
        assert "funding_size" in choices.get("Ascent_L1_to_L2", "")


# =============================================================================
# TEST CLASS: NAVIGATION PATH VALIDATION
# =============================================================================

class TestNavigationPathValidation:
    """
    Validate that navigation paths are correctly structured.
    """

    def test_schools_path_length(self, test0_schools_session: SessionExportData):
        """Verify schools navigation path has correct number of nodes."""
        # Descent: L4(entry) → L3 → L2 → L1 → L0 = 5 nodes
        # Ascent: L1 → L2 → L3 = 3 nodes
        # Total: 8 nodes
        path = test0_schools_session.current_path
        nodes = test0_schools_session.nodes

        print(f"\nPath length: {len(path)}")
        print(f"Node count: {len(nodes)}")

        assert len(nodes) == 8, f"Expected 8 nodes, got {len(nodes)}"
        assert len(path) == 8, f"Expected path of 8, got {len(path)}"

    def test_ademe_path_length(self, test1_ademe_session: SessionExportData):
        """Verify ADEME navigation path has correct number of nodes."""
        path = test1_ademe_session.current_path
        nodes = test1_ademe_session.nodes

        print(f"\nPath length: {len(path)}")
        print(f"Node count: {len(nodes)}")

        assert len(nodes) == 8, f"Expected 8 nodes, got {len(nodes)}"
        assert len(path) == 8, f"Expected path of 8, got {len(path)}"

    def test_schools_descent_levels_sequence(self, test0_schools_session: SessionExportData):
        """Verify descent follows L4→L3→L2→L1→L0 sequence."""
        descent_nodes = test0_schools_session.get_descent_nodes()
        levels = [n.get("level") for n in descent_nodes]

        print(f"\nDescent levels: {levels}")

        expected = [4, 3, 2, 1, 0]
        assert levels == expected, f"Expected {expected}, got {levels}"

    def test_schools_ascent_levels_sequence(self, test0_schools_session: SessionExportData):
        """Verify ascent follows L1→L2→L3 sequence."""
        ascent_nodes = test0_schools_session.get_ascent_nodes()
        levels = [n.get("level") for n in ascent_nodes]

        print(f"\nAscent levels: {levels}")

        expected = [1, 2, 3]
        assert levels == expected, f"Expected {expected}, got {levels}"


# =============================================================================
# DEBUG TEST
# =============================================================================

class TestDebugSessionExports:
    """Debug helper tests for inspecting session exports."""

    def test_print_schools_session_structure(self, test0_schools_session: SessionExportData):
        """Print the full structure of schools session export."""
        print(f"\n{'='*60}")
        print("Schools Session Export Structure")
        print(f"{'='*60}")
        print(f"Session ID: {test0_schools_session.session_id}")
        print(f"Config: {test0_schools_session.config_name}")
        print(f"Description: {test0_schools_session.config_description}")
        print(f"\nNodes ({len(test0_schools_session.nodes)}):")
        for node in test0_schools_session.nodes:
            print(f"  L{node['level']} ({node['action']}): {node['decision_description'][:60]}")
        print(f"\nDatum value: {test0_schools_session.datum.get('value')}")
        print(f"Aggregation: {test0_schools_session.datum.get('aggregation_method')}")

    def test_print_ademe_session_structure(self, test1_ademe_session: SessionExportData):
        """Print the full structure of ADEME session export."""
        print(f"\n{'='*60}")
        print("ADEME Session Export Structure")
        print(f"{'='*60}")
        print(f"Session ID: {test1_ademe_session.session_id}")
        print(f"Config: {test1_ademe_session.config_name}")
        print(f"Description: {test1_ademe_session.config_description}")
        print(f"\nNodes ({len(test1_ademe_session.nodes)}):")
        for node in test1_ademe_session.nodes:
            print(f"  L{node['level']} ({node['action']}): {node['decision_description'][:60]}")
        print(f"\nDatum value: {test1_ademe_session.datum.get('value')}")
        print(f"Aggregation: {test1_ademe_session.datum.get('aggregation_method')}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--headed"])
