"""Render HTML (local file path or http(s) URL) to PDF via WeasyPrint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from urllib.parse import urlparse

PAGE_SIZES = {
    "letter": "Letter",
    "a4": "A4",
    "a3": "A3",
    "legal": "Legal",
}


def _is_url(spec: str) -> bool:
    """Return True only for an http(s) URL.

    ``file`` is deliberately excluded: WeasyPrint's ``HTML(url=...)`` fetcher
    speaks ``file://`` natively, so treating it as a URL bypasses the local
    path branch's own ``html_path.is_file()`` check below and turns the
    ``--html`` argument into an arbitrary local-file reader
    (``file:///etc/passwd``). A local file must always go through the path
    branch, matching the same scheme allowlist already used for untrusted
    URLs elsewhere in this codebase (see
    ``skills/bundled/cron-watchers/scripts/_url.py:require_http_url``).
    """
    parsed = urlparse(spec)
    return parsed.scheme in {"http", "https"}


def render(html_spec: str, out_path: Path, page_size: str | None) -> None:
    try:
        from weasyprint import CSS, HTML
    except ImportError as exc:  # pragma: no cover - covered by --help path
        print(
            "error: weasyprint is not installed — `pip install 'use-agent-os[document-extras]'`",
            file=sys.stderr,
        )
        raise SystemExit(2) from exc

    if _is_url(html_spec):
        html = HTML(url=html_spec)
    else:
        html_path = Path(html_spec)
        if not html_path.is_file():
            print(f"error: html source {html_path} not found", file=sys.stderr)
            raise SystemExit(2)
        html = HTML(filename=str(html_path))

    stylesheets: list[CSS] = []
    if page_size:
        normalized = PAGE_SIZES.get(page_size.lower(), page_size)
        stylesheets.append(CSS(string=f"@page {{ size: {normalized}; }}"))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    html.write_pdf(target=str(out_path), stylesheets=stylesheets or None)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render HTML to PDF via WeasyPrint.")
    parser.add_argument("--html", required=True, help="HTML file path or URL")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--page-size",
        default=None,
        help="Page size override (Letter, A4, A3, Legal, or any valid CSS size value)",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    render(args.html, args.out, args.page_size)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
