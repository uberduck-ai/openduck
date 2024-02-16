import re


def equal_dicts(d1, d2, ignore_keys):
    d1_filtered = {k: v for k, v in d1.items() if k not in ignore_keys}
    d2_filtered = {k: v for k, v in d2.items() if k not in ignore_keys}
    return d1_filtered == d2_filtered


def make_url_name(display_name: str) -> str:
    """
    Converts a display name string like 'Test Name' to a URL friendly string like 'test-name',
    which is lowercase with hyphens instead of spaces.
    """
    # Remove special characters and trim whitespace
    name = display_name
    name = re.sub(r"[^\w\s-]", "", name).strip()
    # Replace multiple spaces with a single space
    name = re.sub(r"\s+", " ", name)
    # Convert to lowercase and replace spaces with hyphens
    return name.lower().replace(" ", "-")
