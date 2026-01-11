# output/formatter.py

def format_to_markdown(ticker, section_summaries, date):
    """
    Formats the output into a Markdown report.

    Args:
        ticker (str): Stock ticker (e.g., "AAPL")
        section_summaries (dict): Mapping of section names to summaries
        date (str): Current date string (YYYY-MM-DD)

    Returns:
        str: Markdown-formatted string
    """

    lines = [
        f"# Infera Report: {ticker}",
        f"**Date:** {date}",
        "\n---\n"
    ]

    for section, summary in section_summaries.items():
        lines.append(f"## {section}")
        lines.append(summary.strip())
        lines.append("\n---\n")

    return "\n".join(lines)
