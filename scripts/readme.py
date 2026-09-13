"""Write measured numbers back into ``README.md``.


Three brief requirements say the *measured* values must appear in the README,
not only in a side report:


* Task 4 - "State your measured values and chosen threshold in ``README.md``."
* Task 5 - the strategy recommendation, citing your own two sets of numbers.
* Task 6 - "State your exact formula **and the threshold**..."


None of those numbers can be known without running the code, so the README ships
with a placeholder inside each marker pair and the relevant script replaces it.
The markers are HTML comments, so they are invisible in rendered Markdown.
"""


from __future__ import annotations


import re
from pathlib import Path
from typing import Final


from app.config import REPO_ROOT


README_PATH: Final[Path] = REPO_ROOT / "README.md"


MARKER_CALIBRATION: Final[str] = "AUTO:CALIBRATION"
MARKER_CHUNKING: Final[str] = "AUTO:CHUNKING"
MARKER_ESCALATION: Final[str] = "AUTO:ESCALATION"




class ReadmeSectionError(RuntimeError):
    """Raised when a marker pair is missing or malformed."""




def _pattern(marker: str) -> re.Pattern[str]:
    return re.compile(
        rf"(<!-- {re.escape(marker)} -->)(.*?)(<!-- /{re.escape(marker)} -->)",
        re.DOTALL,
    )




def replace_section(marker: str, body: str, *, path: Path = README_PATH) -> Path:
    """Replace the text between ``<!-- marker -->`` and ``<!-- /marker -->``.


    Raises:
        ReadmeSectionError: when the file or the marker pair is missing.
    """
    if not path.is_file():
        raise ReadmeSectionError(f"README not found at {path}")


    text = path.read_text(encoding="utf-8")
    pattern = _pattern(marker)
    if not pattern.search(text):
        raise ReadmeSectionError(
            f"marker pair for {marker!r} not found in {path.name}. Expected "
            f"'<!-- {marker} -->' ... '<!-- /{marker} -->'."
        )


    updated = pattern.sub(
        lambda match: f"{match.group(1)}\n{body.strip()}\n{match.group(3)}", text
    )
    path.write_text(updated, encoding="utf-8")
    return path




def read_section(marker: str, *, path: Path = README_PATH) -> str:
    """Return the current content of a marked section, for verification."""
    if not path.is_file():
        raise ReadmeSectionError(f"README not found at {path}")
    match = _pattern(marker).search(path.read_text(encoding="utf-8"))
    if match is None:
        raise ReadmeSectionError(f"marker pair for {marker!r} not found in {path.name}.")
    return match.group(2).strip()



