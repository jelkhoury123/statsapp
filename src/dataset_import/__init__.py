"""Dataset Manager: built-in, bundled R datasets, and user uploads."""

from src.dataset_import.analyse import (
    DatasetPreview,
    missing_value_summary,
    numeric_column_names,
    pair_complete_count,
    preview_frame,
)
from src.dataset_import.io_files import parse_uploaded_table, read_excel_sheets
from src.dataset_import.registry import (
    DatasetSourceKind,
    activate_dataset,
    builtin_options,
    dataset_display_label,
    download_csv_bytes,
    ensure_registry_state,
    get_active_fitting_dataset,
    get_catalog,
    get_frame,
    is_synthetic_dataset,
    library_options,
    list_r_options,
    register_upload,
    resolve_fitting_dataset,
    unified_dataset_options,
    upload_options,
)

__all__ = [
    "DatasetPreview",
    "DatasetSourceKind",
    "activate_dataset",
    "builtin_options",
    "dataset_display_label",
    "download_csv_bytes",
    "ensure_registry_state",
    "get_active_fitting_dataset",
    "get_catalog",
    "get_frame",
    "is_synthetic_dataset",
    "library_options",
    "list_r_options",
    "missing_value_summary",
    "numeric_column_names",
    "pair_complete_count",
    "parse_uploaded_table",
    "preview_frame",
    "read_excel_sheets",
    "register_upload",
    "resolve_fitting_dataset",
    "unified_dataset_options",
    "upload_options",
]
