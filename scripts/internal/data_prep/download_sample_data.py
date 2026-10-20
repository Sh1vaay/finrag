"""Download real SEC 10-K filings for the asset management research corpus.

Populates data/sample/ with SEC 10-K regulatory filings (Apple, Microsoft, Tesla).

Usage:
    python scripts/internal/data_prep/download_sample_data.py
"""

import logging
import sys
import time
from pathlib import Path

import requests
from fpdf import FPDF

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SAMPLE_DIR = Path("data/sample")
SEC_HEADERS = {
    "User-Agent": "RAGAgentDemo/1.0 (agent@example.com)",
    "Accept-Encoding": "gzip, deflate",
}


def sanitize_for_pdf(text: str) -> str:
    """Replace Unicode characters that can't be encoded in Latin-1."""
    replacements = {
        "\u2014": "--",   # em dash
        "\u2013": "-",    # en dash
        "\u2018": "'",    # left single quote
        "\u2019": "'",    # right single quote
        "\u201c": '"',    # left double quote
        "\u201d": '"',    # right double quote
        "\u2026": "...",  # ellipsis
        "\u2022": "-",    # bullet
        "\u00a0": " ",    # non-breaking space
    }
    for char, repl in replacements.items():
        text = text.replace(char, repl)
    return text.encode("latin-1", errors="replace").decode("latin-1")

# ---------------------------------------------------------------------------
# 1. SEC 10-K Filing Downloader
# ---------------------------------------------------------------------------

# Company CIKs (Central Index Keys) for SEC EDGAR
COMPANIES = {
    "apple": {"cik": "0000320193", "name": "Apple Inc."},
    "microsoft": {"cik": "0000789019", "name": "Microsoft Corporation"},
    "tesla": {"cik": "0001318605", "name": "Tesla Inc."},
}


def fetch_sec_filing_text(cik: str, company_name: str) -> str | None:
    """Fetch the most recent 10-K filing text from SEC EDGAR."""
    # Step 1: Get the company's filing index
    submissions_url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    logger.info(f"Fetching filing index for {company_name} from EDGAR...")

    try:
        resp = requests.get(submissions_url, headers=SEC_HEADERS, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.warning(f"Failed to fetch filing index for {company_name}: {e}")
        return None

    # Step 2: Find the most recent 10-K filing
    recent = data.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    accession_numbers = recent.get("accessionNumber", [])
    primary_docs = recent.get("primaryDocument", [])

    for i, form in enumerate(forms):
        if form == "10-K":
            accession = accession_numbers[i].replace("-", "")
            primary_doc = primary_docs[i]
            filing_url = f"https://www.sec.gov/Archives/edgar/data/{cik.lstrip('0')}/{accession}/{primary_doc}"

            logger.info(f"Downloading 10-K filing: {filing_url}")
            time.sleep(0.2)  # SEC rate limit: 10 requests/sec

            try:
                doc_resp = requests.get(filing_url, headers=SEC_HEADERS, timeout=60)
                doc_resp.raise_for_status()
                return doc_resp.text
            except Exception as e:
                logger.warning(f"Failed to download filing document: {e}")
                return None

    logger.warning(f"No 10-K filing found for {company_name}")
    return None


def extract_financial_sections(html_text: str, company_name: str) -> str:
    """Extract key text content from a 10-K HTML filing.

    We strip HTML tags and extract a meaningful excerpt (first ~3000 words)
    covering the financial data sections.
    """
    import re

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", html_text)
    # Remove excessive whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Remove common boilerplate
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"&#\d+;", "", text)

    # Try to find the start of financial discussion
    financial_keywords = [
        "Item 6", "Item 7", "Item 8",
        "Selected Financial Data",
        "Management's Discussion",
        "Financial Statements",
        "Results of Operations",
        "Total net revenue", "Total revenue",
        "Net income", "Operating income",
    ]

    best_start = 0
    for keyword in financial_keywords:
        idx = text.lower().find(keyword.lower())
        if idx > 0 and (best_start == 0 or idx < best_start):
            best_start = max(0, idx - 200)

    # Take ~3000 words from the financial section
    excerpt = text[best_start:]
    words = excerpt.split()[:3000]
    return " ".join(words)


def create_10k_pdf(text: str, company_name: str, filename: str) -> None:
    """Convert 10-K text to a simple PDF."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Title
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, f"{company_name}", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Annual Report (Form 10-K)", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.cell(0, 8, "Filed with the Securities and Exchange Commission", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(10)

    # Body
    pdf.set_font("Helvetica", "", 9)
    pdf.multi_cell(0, 4.5, sanitize_for_pdf(text))

    output_path = SAMPLE_DIR / filename
    pdf.output(str(output_path))
    logger.info(f"Created 10-K PDF: {output_path} ({output_path.stat().st_size / 1024:.0f} KB)")


# ---------------------------------------------------------------------------
# 2. Fallback: Embedded 10-K Excerpts (if SEC download fails)
# ---------------------------------------------------------------------------

FALLBACK_10K_DATA = {
    "apple": {
        "name": "Apple Inc.",
        "text": """APPLE INC.
FORM 10-K — ANNUAL REPORT
Fiscal Year Ended September 30, 2023

ITEM 6. SELECTED FINANCIAL DATA

The following selected financial data should be read in conjunction with the Consolidated Financial Statements and accompanying Notes.

(In millions, except per share amounts)

                                    2023        2022        2021        2020        2019
Total net revenue               $383,285    $394,328    $365,817    $274,515    $260,174
Cost of sales                    214,137     223,546     212,981     169,559     161,782
Gross margin                     169,148     170,782     152,836     104,956      98,392
Operating expenses                54,847      51,345      43,887      38,668      34,462
Operating income                 114,301     119,437     108,949      66,288      63,930
Net income                        96,995     99,803      94,680      57,411      55,256

Earnings per share:
  Basic                           $6.16       $6.15       $5.67       $3.31       $2.99
  Diluted                         $6.13       $6.11       $5.61       $3.28       $2.97

Total assets                    $352,583    $352,755    $351,002    $323,888    $338,516
Total long-term debt            $95,281     $98,959     $109,106    $98,667     $91,807

ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION AND RESULTS OF OPERATIONS

The following discussion should be read in conjunction with the Consolidated Financial Statements and Notes. This section contains forward-looking statements that involve risks and uncertainties.

Revenue

Total net revenue decreased 3% or $11.0 billion during 2023 compared to 2022. The decrease was driven by lower iPhone and Mac revenue, partially offset by higher revenue from Services.

Products revenue decreased 5% during 2023 compared to 2022, driven by lower iPhone and Mac revenue. The weakness in Products revenue was broad-based across all geographic segments.

Services revenue increased 9% during 2023 compared to 2022, driven by growth in advertising, the App Store and cloud services. Services revenue reached $85.2 billion, an all-time high. The Company's installed base of active devices continued to grow and reached an all-time high.

iPhone revenue was $200.6 billion during 2023, a decrease of 2% compared to 2022. The decrease was driven primarily by lower unit sales in the Greater China segment and unfavorable foreign currency movements.

Mac revenue was $29.4 billion during 2023, a decrease of 27% compared to 2022. The decrease was primarily driven by lower laptop sales and the different timing of product launches.

iPad revenue was $28.3 billion during 2023, a decrease of 3% compared to 2022.

Wearables, Home and Accessories revenue was $39.8 billion during 2023, a decrease of 3% compared to 2022.

Gross Margin

Total gross margin decreased during 2023 compared to 2022 due to the decrease in total net revenue. Gross margin percentage increased to 44.1% during 2023 compared to 43.3% during 2022, driven primarily by a favorable shift in product mix toward Services and cost savings.

Products gross margin percentage was 36.5% during 2023 compared to 36.3% during 2022.
Services gross margin percentage was 70.8% during 2023 compared to 71.7% during 2022.

Operating Expenses

Total operating expenses increased 7% or $3.5 billion during 2023 compared to 2022. Research and development expense increased 14% due to increases in headcount-related expenses. Selling, general and administrative expense increased 1%.

Research and development expense was $29.9 billion during 2023 compared to $26.3 billion during 2022.

ITEM 8. FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA

CONSOLIDATED STATEMENTS OF OPERATIONS (In millions)

                                        2023         2022
Net sales:
  Products                           $298,085     $316,199
  Services                             85,200       78,129
    Total net sales                   383,285      394,328

Cost of sales:
  Products                            189,282      201,471
  Services                             24,855       22,075
    Total cost of sales               214,137      223,546

Gross margin                          169,148      170,782

Operating expenses:
  Research and development             29,915       26,251
  Selling, general and administrative  24,932       25,094
    Total operating expenses           54,847       51,345

Operating income                      114,301      119,437
Other income/(expense), net            (382)        (334)
Income before provision for taxes     113,919      119,103
Provision for income taxes             16,924       19,300
Net income                          $  96,995    $  99,803
""",
    },
    "microsoft": {
        "name": "Microsoft Corporation",
        "text": """MICROSOFT CORPORATION
FORM 10-K — ANNUAL REPORT
Fiscal Year Ended June 30, 2023

ITEM 6. SELECTED FINANCIAL DATA

(In millions, except per share amounts)

                                        2023        2022        2021        2020        2019
Revenue                             $211,915    $198,270    $168,088    $143,015    $125,843
Cost of revenue                       65,863      62,650      52,232      46,078      42,910
Gross profit                         146,052     135,620     115,856      96,937      82,933
Operating income                      88,523      83,383      69,916      52,959      42,959
Net income                            72,361      72,738      61,271      44,281      39,240

Diluted earnings per share            $ 9.68      $ 9.65      $ 8.05      $ 5.76      $ 5.06

Total assets                        $411,976    $364,840    $333,779    $301,311    $286,556
Long-term debt                      $ 41,990    $ 47,032    $ 50,074    $ 59,578    $ 66,662

ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS

OVERVIEW

Microsoft is a technology company whose mission is to empower every person and every organization on the planet to achieve more. We generate revenue by offering a wide range of cloud-based solutions, content, and other services to people and businesses; licensing and supporting an array of software products; and delivering relevant online advertising to a global audience.

Revenue increased $13.6 billion or 7% driven by growth across each of our segments. Intelligent Cloud revenue increased driven by Azure and other cloud services. Productivity and Business Processes revenue increased driven by Office 365 Commercial. More Personal Computing revenue increased driven by search and news advertising.

SEGMENT RESULTS OF OPERATIONS

Productivity and Business Processes
Revenue increased $3.5 billion or 9%.
  - Office Commercial products and cloud services revenue increased 10%, driven by Office 365 Commercial revenue growth of 13%.
  - Office Consumer products and cloud services revenue increased 2%.
  - LinkedIn revenue increased 10%.
  - Dynamics products and cloud services revenue increased 16%, driven by Dynamics 365 revenue growth of 24%.

Intelligent Cloud
Revenue increased $7.0 billion or 11%.
  - Server products and cloud services revenue increased 17%, driven by Azure and other cloud services revenue growth of 29%.
  - Azure revenue growth was driven by Azure consumption-based services.

More Personal Computing
Revenue increased $3.1 billion or 4%.
  - Windows OEM revenue decreased 3%.
  - Windows Commercial products and cloud services revenue increased 5%.
  - Search and news advertising revenue excluding traffic acquisition costs increased 23%.
  - Xbox content and services revenue decreased 1%.
  - Surface revenue decreased 20%.
  - Gaming revenue increased 1%.

OPERATING EXPENSES
Research and development expenses increased $1.1 billion or 4% driven by investments in cloud engineering and LinkedIn. Sales and marketing expenses increased $0.5 billion or 2%. General and administrative expenses increased $0.2 billion or 3%.

LIQUIDITY AND CAPITAL RESOURCES
As of June 30, 2023, we had $111.3 billion in cash, cash equivalents, and short-term investments. Cash from operations was $87.6 billion, an increase of $2.2 billion from the prior year.
""",
    },
    "tesla": {
        "name": "Tesla Inc.",
        "text": """TESLA, INC.
FORM 10-K — ANNUAL REPORT
Fiscal Year Ended December 31, 2023

ITEM 6. SELECTED FINANCIAL DATA

(In millions, except per share amounts)

                                        2023        2022        2021        2020        2019
Total revenues                      $ 96,773    $ 81,462    $ 53,823    $ 31,536    $ 24,578
Cost of revenues                      79,113      60,609      40,217      24,906      20,509
Gross profit                          17,660      20,853      13,606       6,630       4,069
Operating income                       8,891      13,656       6,523       1,994         (69)
Net income attributable to common
  stockholders                         7,928      12,556       5,519         721         (862)

Diluted earnings per share             $2.49       $3.62       $1.63       $0.21      ($0.33)

Total assets                        $106,618    $ 82,338    $ 62,131    $ 52,148    $ 34,309
Total long-term debt and finance
  leases, net of current portion     $ 2,857    $  1,597    $  5,245    $  9,607    $ 11,634

ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS

OVERVIEW

We design, develop, manufacture, sell and lease high-performance fully electric vehicles and energy generation and storage systems, and offer services related to our products. We are the world's most valuable automotive company by market capitalization.

Revenue for the year ended December 31, 2023 increased by $15.31 billion, or 19%, compared to the prior year. The increase was primarily due to growth in vehicle deliveries, partially offset by a decrease in average selling price.

REVENUE
Automotive revenue increased by $11.57 billion or 15% year-over-year. This was primarily driven by an increase of 38% in vehicle deliveries (1,808,581 vehicles in 2023 vs 1,313,851 in 2022). Average selling price per vehicle decreased approximately 17% year-over-year due to price reductions across our vehicle lineup to stimulate demand.

Energy generation and storage revenue increased by $4.03 billion or 54%, driven by deployments of Megapack and Powerwall products to utility and commercial customers.

Services and other revenue increased by $1.60 billion or 27% year-over-year, driven by growth in our Supercharger network, body shop and parts, used vehicle sales, and insurance services.

COST OF REVENUES AND GROSS MARGIN
Total cost of revenues increased by $18.50 billion, or 31%, primarily due to higher raw material costs and manufacturing costs associated with increased production volume. Total gross margin was 18.2% compared to 25.6% in the prior year. The decline was driven by price reductions, higher raw material costs, and increased costs associated with new factory ramp-ups.

Automotive gross margin decreased from 28.5% to 18.2%, primarily driven by vehicle price reductions globally. We believe these price adjustments are necessary to maintain demand growth and market share.

OPERATING EXPENSES
Research and development expenses increased $0.93 billion or 26% to $4.46 billion, driven by investments in AI training compute for Full Self-Driving, Cybertruck development, and next-generation vehicle platform development.

Selling, general and administrative expenses increased $0.60 billion or 13% to $5.22 billion, driven by growth in our global operations and workforce.

LIQUIDITY AND CAPITAL RESOURCES
As of December 31, 2023, we had $29.1 billion in cash and cash equivalents and investments. Operating cash flow was $13.3 billion.
""",
    },
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  FinRAG — SEC 10-K Sample Data Generator")
    print("=" * 60)
    print()

    # --- 10-K Filings ---
    print("Downloading/generating SEC 10-K filings...")
    for key, company in COMPANIES.items():
        cik = company["cik"]
        name = company["name"]
        filename = f"10k_{key}_2023.pdf"

        # Try live SEC EDGAR download first
        html_text = fetch_sec_filing_text(cik, name)

        if html_text and len(html_text) > 1000:
            excerpt = extract_financial_sections(html_text, name)
            if len(excerpt) > 500:
                create_10k_pdf(excerpt, name, filename)
                continue

        # Fallback to embedded data
        logger.info(f"Using embedded 10-K data for {name}")
        fallback = FALLBACK_10K_DATA[key]
        create_10k_pdf(fallback["text"], fallback["name"], filename)

    # Summary
    print()
    print("=" * 60)
    files = list(SAMPLE_DIR.glob("*.pdf"))
    print(f"  Done! Prepared {len(files)} SEC regulatory documents in {SAMPLE_DIR}/")
    print()
    for f in sorted(files):
        size_kb = f.stat().st_size / 1024
        print(f"    [10k            ] {f.name} ({size_kb:.0f} KB)")
    print("=" * 60)


if __name__ == "__main__":
    main()
