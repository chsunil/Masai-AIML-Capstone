"""Shared console-plus-Markdown reporting used by 01_eda.py and 02_modeling.py.

Each script builds one Markdown file as it runs, so the printed output and the
committed report can never drift apart.
"""

import os

import matplotlib.pyplot as plt

CHART_DIR = "charts"


class Report:
    def __init__(self):
        self.lines = []

    def say(self, text=""):
        """Print to console and collect for the Markdown report."""
        print(text)
        self.lines.append(text)

    def table(self, df, index=False):
        """Append a DataFrame as a Markdown table (printed as plain text)."""
        print(df.to_string(index=index))
        self.lines.append(df.to_markdown(index=index) + "\n")

    def block(self, content, title=None):
        """Append a fenced code block, for df.info() / describe() style output."""
        if title:
            self.say(f"**{title}**\n")
        text = content if isinstance(content, str) else content.to_string()
        print(text)
        self.lines.append(f"```\n{text}\n```\n")

    def chart(self, name):
        """Save the current matplotlib figure and reference it in the report."""
        os.makedirs(CHART_DIR, exist_ok=True)
        path = f"{CHART_DIR}/{name}.png"
        plt.tight_layout()
        plt.savefig(path, dpi=110)
        plt.close()
        self.lines.append(f"![{name}]({path})\n")
        print(f"  saved {path}")

    def write(self, path):
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(self.lines))
        print(f"\nWrote report -> {path}")
