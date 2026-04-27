
from __future__ import annotations

import re
import warnings
from typing import Optional

import numpy as np
import pandas as pd



_ENTITY_NAME_PATTERNS: list[tuple[str, float]] = [
    (r"\bcustomer[_\s]?id\b",  3.0),
    (r"\buser[_\s]?id\b",      3.0),
    (r"\bclient[_\s]?id\b",    3.0),
    (r"\baccount[_\s]?id\b",   3.0),
    (r"\bmember[_\s]?id\b",    3.0),
    (r"\bperson[_\s]?id\b",    2.5),
    (r"\bsubject[_\s]?id\b",   2.5),
    (r"\bentity[_\s]?id\b",    2.5),
    (r"\bpatient[_\s]?id\b",   2.5),
    (r"\bemployee[_\s]?id\b",  2.5),
    (r"\bcustomer\b",          2.0),
    (r"\buser\b",              2.0),
    (r"\bclient\b",            2.0),
    (r"\baccount\b",           2.0),
    (r"\bmember\b",            2.0),
    (r"_id$",                  1.5),
    (r"^id_",                  1.5),
    (r"\bid\b",                1.0),
    (r"[_\-\s]?no$",           0.8),   # e.g. "order_no", "invoice_no"
    (r"[_\-\s]?num$",          0.8),
    (r"[_\-\s]?number$",       0.8),
    (r"[_\-\s]?code$",         0.8),
    (r"[_\-\s]?key$",          0.8),
    (r"[_\-\s]?ref$",          0.6),
]

# Patterns that *decrease* the score — these columns are unlikely to be the
# entity identifier even if they contain "id"-like tokens.
_ENTITY_NAME_PENALTIES: list[tuple[str, float]] = [
    (r"\bproduct\b",    1.5),
    (r"\border\b",      1.5),
    (r"\btransaction\b", 1.5),
    (r"\binvoice\b",    1.5),
    (r"\bsession\b",    1.2),
    (r"\bevent\b",      1.2),
    (r"\bitem\b",       1.0),
    (r"\bsku\b",        1.0),
    (r"\bticket\b",     1.0),
]

# Column names that are very unlikely to be entity identifiers regardless of
# their content.
_SKIP_COLUMN_NAMES: frozenset[str] = frozenset({
    "index", "row", "rownum", "row_num", "row_id",
    "unnamed", "unnamed: 0",
})

# Fraction thresholds for cardinality gate (unique / total rows).
# The entity column should have "many but not all" unique values.
_CARDINALITY_MIN = 0.005   # at least 0.5 % unique → not literally one value
_CARDINALITY_MAX = 0.98    # at most 98 % unique → there must be *some* repetition

# A column is considered transactional (worth aggregating) when the entity
# column has an average of at least this many rows per entity.
_MIN_ROWS_PER_ENTITY = 1.5

# Maximum number of unique values in a column for it to be treated as
# categorical (not high-cardinality free-text).
_CATEGORICAL_MAX_UNIQUE = 50

# Minimum fraction of non-null values that must parse as datetime for a column
# to be treated as a datetime column.
_DATETIME_PARSE_THRESHOLD = 0.8

# Maximum sample size used when probing an object column for datetime content.
_DATETIME_PROBE_SAMPLE = 200


# ─────────────────────────────────────────────────────────────────────────────
# Helper utilities
# ─────────────────────────────────────────────────────────────────────────────

def _normalise_col_name(name: str) -> str:
    """Lower-case and strip the column name for regex matching."""
    return name.lower().strip()


def _score_entity_name(name: str) -> float:
    """
    Compute a name-heuristic score for a column being the entity identifier.

    A positive score means the name looks like an entity ID; negative means it
    looks like a transaction/event identifier instead.
    """
    n = _normalise_col_name(name)
    score = 0.0
    for pattern, weight in _ENTITY_NAME_PATTERNS:
        if re.search(pattern, n):
            score += weight
    for pattern, weight in _ENTITY_NAME_PENALTIES:
        if re.search(pattern, n):
            score -= weight
    return score


def _is_likely_datetime_column(series: pd.Series) -> bool:
    """
    Return True if the series is (or can be parsed as) a datetime column.

    Strategy:
    1. Already a datetime dtype → True immediately.
    2. Object dtype → sample up to ``_DATETIME_PROBE_SAMPLE`` non-null values
       and attempt pd.to_datetime; accept if the success rate exceeds
       ``_DATETIME_PARSE_THRESHOLD``.
    """
    if pd.api.types.is_datetime64_any_dtype(series):
        return True

    if series.dtype != object:
        return False

    non_null = series.dropna()
    if len(non_null) == 0:
        return False

    sample = non_null.sample(
        min(len(non_null), _DATETIME_PROBE_SAMPLE),
        random_state=0,
    )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed = pd.to_datetime(sample, errors="coerce", infer_datetime_format=True)
        success_rate = parsed.notna().mean()
        return success_rate >= _DATETIME_PARSE_THRESHOLD
    except Exception:
        return False


def _safe_mode(series: pd.Series):
    """Return the most frequent value of *series*, or NaN if empty."""
    mode = series.mode(dropna=True)
    return mode.iloc[0] if len(mode) > 0 else np.nan


# ─────────────────────────────────────────────────────────────────────────────
# Main class
# ─────────────────────────────────────────────────────────────────────────────

class EntityAggregator:
    """
    Detect and aggregate transactional DataFrames to entity level.

    The aggregator is fully dataset-agnostic: it uses heuristics to find the
    entity column, datetime columns, and target column; it then collapses all
    rows that share the same entity identifier into a single summary row.

    Parameters
    ----------
    target_col : str, optional
        Name of the column to predict.  If provided it is handled separately
        (binary → max aggregation; multiclass → mode aggregation) to avoid
        data leakage.  If None, the target is treated like any other column.
    prefer_col : str, optional
        Manually override entity column detection by providing the column name.
    verbose : bool
        Print a summary of detected columns and applied transformations.

    Attributes (set after fit / fit_transform)
    ------------------------------------------
    entity_col_ : str or None
        Name of the detected entity column.
    datetime_cols_ : list[str]
        Names of detected datetime columns.
    was_aggregated_ : bool
        True if transactional structure was found and aggregation was applied.
    aggregation_report_ : dict
        Human-readable summary of what was done to each column.
    """

    def __init__(
        self,
        target_col: Optional[str] = None,
        prefer_col: Optional[str] = None,
        verbose: bool = True,
    ) -> None:
        self.target_col = target_col
        self.prefer_col = prefer_col
        self.verbose = verbose

        # Populated during fit
        self.entity_col_: Optional[str] = None
        self.datetime_cols_: list[str] = []
        self.was_aggregated_: bool = False
        self.aggregation_report_: dict = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def fit_transform(
        self,
        df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, Optional[str], bool]:
        """
        Detect structure and, if transactional, aggregate the DataFrame.

        This is the main entry point.  It performs:
        1. Entity column detection (or use ``prefer_col``).
        2. Datetime column detection.
        3. Transactional check — does the entity column have repeated values?
        4. If transactional: aggregate and return the collapsed DataFrame.
        5. If not transactional: return the original DataFrame unchanged.

        Parameters
        ----------
        df : pd.DataFrame
            Raw input DataFrame (before train/test split).

        Returns
        -------
        df_out : pd.DataFrame
            Entity-level DataFrame (or the original if not transactional).
        entity_col : str or None
            Detected (or provided) entity column name.
        was_aggregated : bool
            True if aggregation was applied.
        """
        if df is None or df.empty:
            raise ValueError("Input DataFrame is None or empty.")

        df = df.copy()

        # ── Step 1: detect entity column ──────────────────────────────────
        if self.prefer_col is not None:
            if self.prefer_col not in df.columns:
                raise ValueError(
                    f"prefer_col '{self.prefer_col}' not found in DataFrame columns."
                )
            self.entity_col_ = self.prefer_col
        else:
            self.entity_col_ = self._detect_entity_column(df)

        # ── Step 2: detect datetime columns ───────────────────────────────
        self.datetime_cols_ = self._detect_datetime_columns(df)

        # ── Step 3: is the dataset transactional? ─────────────────────────
        if self.entity_col_ is None or not self._is_transactional(df):
            self.was_aggregated_ = False
            if self.verbose:
                print(
                    "[EntityAggregator] Dataset does not appear transactional. "
                    "Aggregation skipped."
                )
            return df, self.entity_col_, False

        # ── Step 4: aggregate ─────────────────────────────────────────────
        df_agg = self._aggregate(df)
        self.was_aggregated_ = True

        if self.verbose:
            self._print_summary(df, df_agg)

        return df_agg, self.entity_col_, True

    # ── Detection helpers ─────────────────────────────────────────────────────

    def _detect_entity_column(self, df: pd.DataFrame) -> Optional[str]:
        """
        Score every non-target column and return the best entity candidate.

        Scoring combines three signals:
        - **Name heuristic**: regex patterns that match ID-like names.
        - **Cardinality gate**: unique-ratio must be in [min, max] thresholds.
        - **Repetition boost**: columns with lower unique-ratio (more repetition)
          receive a small bonus because an entity identifier *should* repeat.

        Returns None if no convincing candidate is found (score ≤ 0).
        """
        candidates: list[tuple[str, float]] = []

        n_rows = len(df)

        for col in df.columns:
            # Skip the target
            if self.target_col and col == self.target_col:
                continue

            # Skip known non-entity column names
            if _normalise_col_name(col) in _SKIP_COLUMN_NAMES:
                continue

            # Gate: must be string/object or integer dtype (IDs are often int)
            if not (
                df[col].dtype == object
                or pd.api.types.is_integer_dtype(df[col])
            ):
                continue

            # Gate: skip datetime columns — they are not entity keys
            if _is_likely_datetime_column(df[col]):
                continue

            n_unique = df[col].nunique(dropna=True)
            if n_rows == 0 or n_unique == 0:
                continue

            unique_ratio = n_unique / n_rows

            # Cardinality gate: must have both repetition and diversity
            if not (_CARDINALITY_MIN <= unique_ratio <= _CARDINALITY_MAX):
                continue

            # Base score from column name
            name_score = _score_entity_name(col)

            # Repetition bonus: lower unique-ratio means more repetition
            # Entity columns typically have unique_ratio in [0.01, 0.4]
            rep_bonus = max(0.0, 1.0 - unique_ratio) * 0.5

            # Penalise if the column has *too many* unique values relative to
            # rows (looks more like a transaction ID than an entity ID)
            if unique_ratio > 0.7:
                rep_bonus -= 1.0

            total_score = name_score + rep_bonus

            candidates.append((col, total_score))

        if not candidates:
            return None

        # Sort by score descending; pick the top candidate only if score > 0
        candidates.sort(key=lambda x: x[1], reverse=True)
        best_col, best_score = candidates[0]

        if best_score <= 0:
            return None

        return best_col

    def _detect_datetime_columns(self, df: pd.DataFrame) -> list[str]:
        """
        Return names of columns that are (or can be parsed as) datetimes.

        Already-datetime columns are coerced in-place so subsequent logic can
        use them directly.
        """
        dt_cols: list[str] = []

        for col in df.columns:
            if col == self.target_col:
                continue
            if col == self.entity_col_:
                continue
            if _is_likely_datetime_column(df[col]):
                dt_cols.append(col)
                # Coerce to datetime in-place if it is still object dtype
                if df[col].dtype == object:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        df[col] = pd.to_datetime(
                            df[col], errors="coerce", infer_datetime_format=True
                        )

        return dt_cols

    def _is_transactional(self, df: pd.DataFrame) -> bool:
        """
        Return True if the entity column contains repeated values indicating
        multiple rows per entity (i.e., a transactional / event-log structure).

        The threshold is controlled by ``_MIN_ROWS_PER_ENTITY``.
        """
        if self.entity_col_ is None:
            return False

        n_rows = len(df)
        n_unique_entities = df[self.entity_col_].nunique(dropna=True)

        if n_unique_entities == 0:
            return False

        avg_rows_per_entity = n_rows / n_unique_entities
        return avg_rows_per_entity >= _MIN_ROWS_PER_ENTITY

    # ── Aggregation engine ────────────────────────────────────────────────────

    def _aggregate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Collapse the DataFrame from row-per-event to row-per-entity.

        Rules applied per column type:
        - **Numeric (non-target)**: mean, sum, min, max, std, count
          → creates six derived columns per original column.
        - **Categorical (low-cardinality object)**: mode (most frequent),
          nunique (count of distinct values seen).
        - **Datetime**: min (first event), max (last event),
          recency_days = global_max_date − entity_last_date,
          tenure_days  = entity_last_date − entity_first_date.
        - **Target column**:
          - Binary (2 distinct values) → max()  [any positive event = 1]
          - Multiclass                 → mode()
        - **Entity column**: used as the groupby key; retained as-is.
        - **High-cardinality object / other**: skipped (excluded from output).
        """
        entity = self.entity_col_
        target = self.target_col
        dt_cols = set(self.datetime_cols_)
        report: dict[str, str] = {}

        # ── Classify every non-entity column ──────────────────────────────
        numeric_cols: list[str] = []
        categorical_cols: list[str] = []   # low-cardinality object columns
        skipped_cols: list[str] = []

        for col in df.columns:
            if col == entity:
                continue
            if col == target:
                continue
            if col in dt_cols:
                continue

            if pd.api.types.is_numeric_dtype(df[col]):
                numeric_cols.append(col)
            elif df[col].dtype == object or pd.api.types.is_categorical_dtype(df[col]):
                if df[col].nunique(dropna=True) <= _CATEGORICAL_MAX_UNIQUE:
                    categorical_cols.append(col)
                else:
                    skipped_cols.append(col)
            else:
                skipped_cols.append(col)

        # ── Build partial aggregation dicts ───────────────────────────────
        agg_dict: dict[str, object] = {}

        # Numeric columns
        for col in numeric_cols:
            agg_dict[col] = ["mean", "sum", "min", "max", "std", "count"]
            report[col] = "numeric → mean, sum, min, max, std, count"

        # Categorical columns
        for col in categorical_cols:
            agg_dict[col] = [_safe_mode, "nunique"]
            report[col] = "categorical → mode, nunique"

        # Datetime columns
        for col in dt_cols:
            agg_dict[col] = ["min", "max"]
            report[col] = "datetime → min (first), max (last), + recency/tenure"

        # ── Run groupby aggregation ────────────────────────────────────────
        if not agg_dict:
            # Nothing to aggregate apart from the entity column itself;
            # return entity deduplication only.
            df_agg = df[[entity]].drop_duplicates(subset=[entity])
        else:
            df_agg = df.groupby(entity, sort=False).agg(agg_dict)

        # Flatten MultiIndex columns produced by multiple agg functions.
        # Strip any leading underscores from the function-name part so that
        # private helpers like ``_safe_mode`` don't create double-underscore
        # column names (e.g. ``col__safe_mode`` → ``col_safe_mode``).
        if isinstance(df_agg.columns, pd.MultiIndex):
            def _flatten(col, agg_fn) -> str:
                fn_name = agg_fn.__name__ if callable(agg_fn) else str(agg_fn)
                fn_name = fn_name.lstrip("_")          # strip leading underscores
                return f"{col}_{fn_name}"
            df_agg.columns = [_flatten(c, f) for c, f in df_agg.columns]

        df_agg = df_agg.reset_index()

        # ── Rename categorical mode columns (has ugly func name) ──────────
        rename_map: dict[str, str] = {}
        for col in categorical_cols:
            old_name = f"{col}_safe_mode"
            if old_name in df_agg.columns:
                rename_map[old_name] = f"{col}_mode"
        if rename_map:
            df_agg = df_agg.rename(columns=rename_map)

        # ── Enrich datetime columns with recency and tenure ────────────────
        for col in dt_cols:
            max_col = f"{col}_max"
            min_col = f"{col}_min"

            if max_col not in df_agg.columns or min_col not in df_agg.columns:
                continue

            # Convert if still object after groupby (use loc to avoid CoW warning)
            df_agg = df_agg.copy()   # ensure we own this frame
            df_agg.loc[:, max_col] = pd.to_datetime(df_agg[max_col], errors="coerce")
            df_agg.loc[:, min_col] = pd.to_datetime(df_agg[min_col], errors="coerce")

            # Global reference date: the latest observed date across all entities
            global_max_date = df_agg[max_col].max()

            recency_col = f"{col}_recency_days"
            tenure_col  = f"{col}_tenure_days"

            df_agg = df_agg.assign(**{
                recency_col: (
                    global_max_date - df_agg[max_col]
                ).dt.total_seconds() / 86_400,
                tenure_col: (
                    df_agg[max_col] - df_agg[min_col]
                ).dt.total_seconds() / 86_400,
            })

            report[f"{col}_recency_days"] = (
                f"datetime → recency_days (global_max={global_max_date.date()})"
            )
            report[f"{col}_tenure_days"] = "datetime → tenure_days (last − first)"

        # ── Aggregate target column separately ────────────────────────────
        if target and target in df.columns:
            n_target_unique = df[target].nunique(dropna=True)

            if n_target_unique <= 2:
                # Binary: max → if *any* row had label=1, entity label=1
                target_agg = df.groupby(entity, sort=False)[target].max()
                agg_rule = "binary target → max (any-positive)"
            else:
                # Multiclass: mode → most common label for the entity
                target_agg = (
                    df.groupby(entity, sort=False)[target]
                    .agg(_safe_mode)
                )
                agg_rule = "multiclass target → mode"

            target_agg = target_agg.reset_index()
            df_agg = df_agg.merge(target_agg, on=entity, how="left")
            report[target] = agg_rule

        # ── Record skipped columns ────────────────────────────────────────
        for col in skipped_cols:
            report[col] = "skipped (high-cardinality object or unsupported dtype)"

        self.aggregation_report_ = report

        return df_agg

    # ── Reporting ─────────────────────────────────────────────────────────────

    def _print_summary(self, df_raw: pd.DataFrame, df_agg: pd.DataFrame) -> None:
        """Print a human-readable aggregation summary to stdout."""
        print("\n" + "=" * 60)
        print("  EntityAggregator — Aggregation Summary")
        print("=" * 60)
        print(f"  Entity column      : {self.entity_col_}")
        print(f"  Datetime columns   : {self.datetime_cols_ or 'none detected'}")
        print(f"  Target column      : {self.target_col or 'none'}")
        print(f"  Input shape        : {df_raw.shape}")
        print(f"  Output shape       : {df_agg.shape}")
        print(f"  Entities detected  : {df_agg.shape[0]:,}")
        avg = df_raw.shape[0] / max(df_agg.shape[0], 1)
        print(f"  Avg rows/entity    : {avg:.2f}")
        print()
        print("  Column aggregation rules:")
        for col, rule in self.aggregation_report_.items():
            print(f"    {col:<35} {rule}")
        print("=" * 60 + "\n")

    def get_report(self) -> pd.DataFrame:
        """
        Return the aggregation report as a tidy DataFrame.

        Columns: ``column``, ``rule``.
        Only populated after :meth:`fit_transform` has been called.
        """
        if not self.aggregation_report_:
            return pd.DataFrame(columns=["column", "rule"])
        return pd.DataFrame(
            [{"column": k, "rule": v} for k, v in self.aggregation_report_.items()]
        )

    def get_entity_candidates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Return a scored ranking of all entity-column candidates for inspection.

        Useful for debugging detection or picking a manual override.

        Parameters
        ----------
        df : pd.DataFrame
            The raw input DataFrame.

        Returns
        -------
        pd.DataFrame with columns: ``column``, ``dtype``, ``unique_count``,
        ``unique_ratio``, ``name_score``, ``total_score``.
        Sorted by ``total_score`` descending.
        """
        rows = []
        n_rows = max(len(df), 1)

        for col in df.columns:
            if self.target_col and col == self.target_col:
                continue
            if _normalise_col_name(col) in _SKIP_COLUMN_NAMES:
                continue

            is_dt = _is_likely_datetime_column(df[col])
            n_unique = df[col].nunique(dropna=True)
            unique_ratio = n_unique / n_rows
            name_score = _score_entity_name(col)
            rep_bonus = max(0.0, 1.0 - unique_ratio) * 0.5
            if unique_ratio > 0.7:
                rep_bonus -= 1.0
            cardinality_ok = _CARDINALITY_MIN <= unique_ratio <= _CARDINALITY_MAX
            rows.append({
                "column":        col,
                "dtype":         str(df[col].dtype),
                "is_datetime":   is_dt,
                "unique_count":  n_unique,
                "unique_ratio":  round(unique_ratio, 4),
                "name_score":    round(name_score, 2),
                "rep_bonus":     round(rep_bonus, 2),
                "total_score":   round(name_score + rep_bonus, 2),
                "cardinality_ok": cardinality_ok,
            })

        return (
            pd.DataFrame(rows)
            .sort_values("total_score", ascending=False)
            .reset_index(drop=True)
        )


# ─────────────────────────────────────────────────────────────────────────────
# Example usage
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── Build a synthetic transactional dataset ───────────────────────────
    import random
    from datetime import datetime, timedelta

    random.seed(42)
    n_events = 500
    n_customers = 100

    records = []
    for _ in range(n_events):
        cid = f"CUST{random.randint(1, n_customers):04d}"
        days_ago = random.randint(0, 365)
        event_date = datetime(2024, 1, 1) + timedelta(days=days_ago)
        records.append({
            "customer_id":    cid,
            "order_id":       f"ORD{random.randint(1000, 9999)}",
            "order_date":     event_date,
            "product_category": random.choice(["Electronics", "Apparel", "Food"]),
            "quantity":       random.randint(1, 10),
            "unit_price":     round(random.uniform(5.0, 200.0), 2),
            "discount_pct":   round(random.uniform(0.0, 0.3), 2),
            "country":        random.choice(["US", "UK", "DE", "FR"]),
            "churned":        random.choice([0, 0, 0, 1]),  # ~25 % churn
        })

    df_raw = pd.DataFrame(records)
    print(f"Raw dataset shape: {df_raw.shape}")

    # ── Run the aggregator ────────────────────────────────────────────────
    agg = EntityAggregator(target_col="churned", verbose=True)
    df_entity, detected_entity_col, aggregated = agg.fit_transform(df_raw)

    print(f"Entity column  : {detected_entity_col}")
    print(f"Aggregated     : {aggregated}")
    print(f"Entity df shape: {df_entity.shape}")
    print()
    print("First 3 rows of entity-level DataFrame:")
    print(df_entity.head(3).to_string())
    print()
    print("Aggregation report:")
    print(agg.get_report().to_string(index=False))
