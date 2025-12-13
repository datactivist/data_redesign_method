"""
Search component styling for data.gouv.fr integration.

Feature: 008-datagouv-search
Uses Klein Blue (#002fa7) accent color from existing design system.
"""


def get_search_styles() -> str:
    """Return CSS styles for the search interface components."""
    return """
    <style>
    /* Search tagline styling */
    .search-tagline {
        font-size: 2rem;
        font-weight: 600;
        color: #1e293b;
        text-align: center;
        margin-bottom: 0.5rem;
        line-height: 1.3;
    }

    .search-tagline-accent {
        color: #002fa7;
    }

    .search-subtitle {
        font-size: 1rem;
        color: #64748b;
        text-align: center;
        margin-bottom: 2rem;
    }

    /* Dataset card styling */
    .dataset-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        transition: all 0.2s ease;
        cursor: pointer;
    }

    .dataset-card:hover {
        border-color: #002fa7;
        box-shadow: 0 4px 12px rgba(0, 47, 167, 0.1);
    }

    .dataset-card-selected {
        border-color: #002fa7;
        border-width: 2px;
        background: #f8faff;
    }

    .dataset-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 0.5rem;
        line-height: 1.4;
    }

    .dataset-description {
        font-size: 0.9rem;
        color: #64748b;
        margin-bottom: 0.75rem;
        line-height: 1.5;
    }

    .dataset-meta {
        display: flex;
        flex-wrap: wrap;
        gap: 1rem;
        font-size: 0.8rem;
        color: #94a3b8;
    }

    .dataset-meta-item {
        display: flex;
        align-items: center;
        gap: 0.25rem;
    }

    .dataset-org {
        color: #002fa7;
        font-weight: 500;
    }

    /* CSV availability badge */
    .csv-badge {
        display: inline-block;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 500;
    }

    .csv-badge-available {
        background: #dcfce7;
        color: #166534;
    }

    .csv-badge-unavailable {
        background: #fee2e2;
        color: #991b1b;
    }

    /* Resource selector */
    .resource-list {
        background: #f8fafc;
        border-radius: 8px;
        padding: 1rem;
        margin-top: 1rem;
    }

    .resource-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.75rem;
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        margin-bottom: 0.5rem;
    }

    .resource-item:hover {
        border-color: #002fa7;
    }

    .resource-title {
        font-weight: 500;
        color: #1e293b;
    }

    .resource-size {
        font-size: 0.85rem;
        color: #64748b;
    }

    /* Loading state */
    .loading-container {
        text-align: center;
        padding: 3rem;
    }

    .loading-text {
        font-size: 1rem;
        color: #64748b;
        margin-top: 1rem;
    }

    /* Error state */
    .error-container {
        background: #fef2f2;
        border: 1px solid #fecaca;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }

    .error-text {
        color: #991b1b;
    }

    /* No results state */
    .no-results {
        text-align: center;
        padding: 2rem;
        color: #64748b;
    }

    .no-results-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
    }

    /* Search container layout */
    .search-container {
        max-width: 800px;
        margin: 0 auto;
        padding: 2rem 1rem;
    }

    /* Results header */
    .results-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #e2e8f0;
    }

    .results-count {
        font-size: 0.9rem;
        color: #64748b;
    }

    /* Divider between search and upload */
    .entry-divider {
        display: flex;
        align-items: center;
        text-align: center;
        margin: 2rem 0;
        color: #94a3b8;
    }

    .entry-divider::before,
    .entry-divider::after {
        content: '';
        flex: 1;
        border-bottom: 1px solid #e2e8f0;
    }

    .entry-divider span {
        padding: 0 1rem;
        font-size: 0.9rem;
    }

    /* Large file warning */
    .large-file-warning {
        background: #fefce8;
        border: 1px solid #fef08a;
        border-radius: 8px;
        padding: 0.75rem;
        margin-top: 0.5rem;
        font-size: 0.85rem;
        color: #854d0e;
    }
    </style>
    """
