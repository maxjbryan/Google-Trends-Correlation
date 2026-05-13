"""
Google Trends Pearson Correlation Analyzer
-------------------------------------------
Interactively prompts for two search terms, a date range, and a region,
then fetches Google Trends data and computes the Pearson correlation.

Requirements:
    pip install pytrends scipy pandas
    (If you hit a urllib3 >=2.0 error, also run: pip install "urllib3<2.0")

Usage:
    python trends_correlation.py
"""

import sys
import time
import re
from datetime import datetime, date

# ── Dependency checks ─────────────────────────────────────────────────────────
try:
    from pytrends.request import TrendReq
except ImportError:
    sys.exit("❌  pytrends not found.  Run:  pip install pytrends")

try:
    from scipy import stats
except ImportError:
    sys.exit("❌  scipy not found.  Run:  pip install scipy")

try:
    import pandas as pd
except ImportError:
    sys.exit("❌  pandas not found.  Run:  pip install pandas")

# ── Constants ─────────────────────────────────────────────────────────────────
DATE_FMT   = "%Y-%m-%d"
MIN_DATE   = date(2004, 1, 1)   # Google Trends earliest available date
TODAY      = date.today()

GPROP_OPTIONS = {
    "1": ("",        "Web search (default)"),
    "2": ("news",    "Google News"),
    "3": ("youtube", "YouTube"),
    "4": ("images",  "Google Images"),
    "5": ("froogle", "Google Shopping"),
}


# ── Input helpers ─────────────────────────────────────────────────────────────

def prompt(message: str, default: str = "") -> str:
    """Print a prompt and return stripped input, falling back to default."""
    suffix = f" [{default}]" if default else ""
    try:
        value = input(f"  {message}{suffix}: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n\n  Goodbye!")
        sys.exit(0)
    return value if value else default


def prompt_term(label: str, example: str) -> str:
    """Ask for a non-empty search term."""
    while True:
        term = prompt(f"Search term {label} (e.g. '{example}')")
        if term:
            return term
        print("  ⚠️   Term cannot be empty. Please try again.")


def prompt_date(label: str, default: str) -> str:
    """Ask for a date in YYYY-MM-DD format within the valid Google Trends range."""
    while True:
        raw = prompt(f"{label} (YYYY-MM-DD)", default=default)
        try:
            d = datetime.strptime(raw, DATE_FMT).date()
        except ValueError:
            print(f"  ⚠️   '{raw}' is not a valid date. Use YYYY-MM-DD format.")
            continue
        if d < MIN_DATE:
            print(f"  ⚠️   Date must be on or after {MIN_DATE} (Google Trends limit).")
            continue
        if d > TODAY:
            print(f"  ⚠️   Date cannot be in the future (today is {TODAY}).")
            continue
        return raw


def prompt_dates() -> tuple:
    """Ask for start and end dates, ensuring start < end."""
    while True:
        start = prompt_date("Start date", default="2022-01-01")
        end   = prompt_date("End date",   default=str(TODAY))
        if start >= end:
            print("  ⚠️   Start date must be before end date. Please try again.")
            continue
        return start, end


def prompt_geo() -> str:
    """Ask for an optional country/region code."""
    print()
    print("  Region code examples: US, GB, DE, FR, JP, AU")
    print("  Leave blank for Worldwide data.")
    geo = prompt("Region code", default="").upper()
    if geo and not re.match(r"^[A-Z]{2}(-[A-Z0-9]{1,3})?$", geo):
        print(f"  ⚠️   '{geo}' doesn't look like a standard code — using it anyway.")
    return geo


def prompt_gprop() -> str:
    """Let the user pick a data source."""
    print()
    print("  Data source options:")
    for key, (_, label) in GPROP_OPTIONS.items():
        print(f"    {key}) {label}")
    while True:
        choice = prompt("Choose a data source", default="1")
        if choice in GPROP_OPTIONS:
            return GPROP_OPTIONS[choice][0]
        print(f"  ⚠️   Please enter a number between 1 and {len(GPROP_OPTIONS)}.")


def ask_again() -> bool:
    """Ask if the user wants to run another comparison."""
    print()
    while True:
        ans = prompt("Run another comparison? (y/n)", default="n").lower()
        if ans in ("y", "yes"):
            return True
        if ans in ("n", "no", ""):
            return False
        print("  ⚠️   Please enter y or n.")


# pytrends logic

def fetch_trends(term_a: str, term_b: str,
                 start: str, end: str,
                 geo: str, gprop: str) -> pd.DataFrame:
    """Return a DataFrame with interest scores for both terms."""
    pytrends  = TrendReq(hl="en-US", tz=0, retries=3, backoff_factor=1)
    timeframe = f"{start} {end}"

    print(f"\n  Fetching Google Trends data …")
    print(f"      Terms     : '{term_a}'  vs  '{term_b}'")
    print(f"      Timeframe : {timeframe}")
    print(f"      Geo       : {geo or 'Worldwide'}")
    print(f"      Source    : {gprop or 'Web search'}\n")

    pytrends.build_payload(
        kw_list=[term_a, term_b],
        timeframe=timeframe,
        geo=geo,
        gprop=gprop,
    )
    time.sleep(1)   # be polite to the API

    df = pytrends.interest_over_time()

    if df.empty:
        print("  ❌  No data returned for those terms / date range.")
        print("      Try broader dates, different terms, or a different region.\n")
        return pd.DataFrame()

    if "isPartial" in df.columns:
        df = df.drop(columns=["isPartial"])

    return df


def pearson_correlation(df: pd.DataFrame, col_a: str, col_b: str) -> None:
    """Compute and pretty-print Pearson r and descriptive stats."""
    series_a = df[col_a]
    series_b = df[col_b]

    r, p_value = stats.pearsonr(series_a, series_b)
    n          = len(series_a)

    abs_r = abs(r)
    if abs_r >= 0.9:
        strength = "very strong"
    elif abs_r >= 0.7:
        strength = "strong"
    elif abs_r >= 0.5:
        strength = "moderate"
    elif abs_r >= 0.3:
        strength = "weak"
    else:
        strength = "very weak / negligible"

    direction = "positive" if r > 0 else "negative"

    if p_value < 0.001:
        sig = "highly significant (p < 0.001)"
    elif p_value < 0.01:
        sig = "significant (p < 0.01)"
    elif p_value < 0.05:
        sig = "significant (p < 0.05)"
    else:
        sig = "NOT statistically significant (p >= 0.05)"

    sep = "─" * 57
    print(sep)
    print("  PEARSON CORRELATION RESULTS")
    print(sep)
    print(f"  Term A        : {col_a}")
    print(f"  Term B        : {col_b}")
    print(f"  Data points   : {n} observations")
    print(sep)
    print(f"  Pearson r     : {r:+.4f}")
    print(f"  p-value       : {p_value:.4e}")
    print(sep)
    print(f"  Interpretation: {strength.capitalize()} {direction} correlation")
    print(f"                  Result is {sig}.")
    print(sep)

    print("\n  DESCRIPTIVE STATISTICS")
    a_label = col_a[:14]
    b_label = col_b[:14]
    print(f"  {'Metric':<20} {a_label:>14} {b_label:>14}")
    print(f"  {'─'*20} {'─'*14} {'─'*14}")
    for label, fn in [("Mean", "mean"), ("Std Dev", "std"),
                      ("Min",  "min"),  ("Max",    "max")]:
        va = getattr(series_a, fn)()
        vb = getattr(series_b, fn)()
        print(f"  {label:<20} {va:>14.2f} {vb:>14.2f}")
    print()


# ── Entry point ───────────────────────────────────────────────────────────────

def run_once() -> None:
    """Collect inputs, fetch data, and print results for one comparison."""
    print()
    print("=" * 57)
    print("  GOOGLE TRENDS  ·  PEARSON CORRELATION ANALYZER")
    print("=" * 57)

    # Terms
    print()
    term_a = prompt_term("A", "")
    term_b = prompt_term("B", "")

    # Dates
    print()
    start, end = prompt_dates()

    # Region
    geo = prompt_geo()

    # Data source
    gprop = prompt_gprop()

    # Fetch
    df = fetch_trends(term_a, term_b, start, end, geo, gprop)
    if df.empty:
        return   # error already printed inside fetch_trends

    # Preview
    print("  Sample data (first 5 rows):")
    print(df.head().to_string())
    print()

    # Results
    pearson_correlation(df, col_a=term_a, col_b=term_b)


def main() -> None:
    while True:
        run_once()
        if not ask_again():
            print("\n  Goodbye!\n")
            break


if __name__ == "__main__":
    main()
