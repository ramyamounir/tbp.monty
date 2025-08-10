# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.


def create_stats_text(data: dict) -> str:
    """Generates formatted statistics text based on the two-level dict structure.

    Returns:
        Formatted statistics text as a string.
    """
    stats_lines = []

    # Iterate over headers and their corresponding entries
    for header, entries in data.items():
        # Add header line
        stats_lines.append(f"=== {header} ===")

        # Process each entry under the header
        for entry in entries:
            # If the entry is a dictionary (like a location or rotation), handle it
            if isinstance(entry, dict):
                for sub_key, sub_value in entry.items():
                    stats_lines.append(f"{sub_key}: {sub_value}")
            # If it's a list or simple value, format accordingly
            elif isinstance(entry, list):
                stats_lines.append(f"{header} values: {', '.join(map(str, entry))}")
            else:
                stats_lines.append(f"{entry}")

        stats_lines.append("")  # Add an empty line for separation between sections

    # Combine all the lines into a single string
    return "\n".join(stats_lines)
