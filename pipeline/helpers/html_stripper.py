from html.parser import HTMLParser
from html import unescape
import re


class _HTMLStripper(HTMLParser):
    _BLOCK_BREAK_TAGS = {
        "p", "div", "section", "article", "header", "footer",
        "ul", "ol", "li", "br", "hr", "h1", "h2", "h3", "h4", "h5", "h6",
    }

    def __init__(self):
        super().__init__()
        self._chunks: list[str] = []

        self._in_table = False
        self._in_row = False
        self._in_cell = False
        self._current_cell_is_header = False
        self._current_colspan = 1

        self._cell_buffer: list[str] = []
        self._current_row: list[str] = []
        self._current_row_header_flags: list[bool] = []
        self._table_rows: list[list[str]] = []
        self._table_row_header_flags: list[list[bool]] = []

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        attr_dict = {k.lower(): v for k, v in attrs}

        if tag == "table":
            self._flush_text_break()
            self._in_table = True
            self._table_rows = []
            self._table_row_header_flags = []
            return

        if self._in_table and tag == "tr":
            self._in_row = True
            self._current_row = []
            self._current_row_header_flags = []
            return

        if self._in_table and tag in ("th", "td"):
            self._in_cell = True
            self._current_cell_is_header = tag == "th"
            self._cell_buffer = []
            try:
                self._current_colspan = max(1, int(attr_dict.get("colspan", "1")))
            except (TypeError, ValueError):
                self._current_colspan = 1
            return

        if tag == "li":
            self._flush_text_break()
            self._chunks.append("* ")
            return

        if tag in self._BLOCK_BREAK_TAGS:
            self._flush_text_break()

    def handle_endtag(self, tag):
        tag = tag.lower()

        if self._in_table and tag in ("th", "td") and self._in_cell:
            cell_text = self._normalize_inline(" ".join(self._cell_buffer))
            self._current_row.append(cell_text)
            self._current_row_header_flags.append(self._current_cell_is_header)

            for _ in range(self._current_colspan - 1):
                self._current_row.append("")
                self._current_row_header_flags.append(self._current_cell_is_header)

            self._cell_buffer = []
            self._in_cell = False
            self._current_cell_is_header = False
            self._current_colspan = 1
            return

        if self._in_table and tag == "tr" and self._in_row:
            if self._current_row:
                self._table_rows.append(self._current_row)
                self._table_row_header_flags.append(self._current_row_header_flags)
            self._current_row = []
            self._current_row_header_flags = []
            self._in_row = False
            return

        if self._in_table and tag == "table":
            self._emit_table_as_markdown()
            self._in_table = False
            self._flush_text_break()
            return

        if tag in self._BLOCK_BREAK_TAGS:
            self._flush_text_break()

    def handle_data(self, data):
        data = unescape(data or "")
        if not data.strip():
            return

        if self._in_table and self._in_cell:
            self._cell_buffer.append(data)
        else:
            self._chunks.append(data)

    def get_data(self):
        text = "".join(self._chunks)
        lines = [self._normalize_inline(line) for line in text.split("\n")]
        cleaned = [line for line in lines if line]
        return "\n".join(cleaned)

    def _emit_table_as_markdown(self):
        if not self._table_rows:
            return

        max_cols = max(len(r) for r in self._table_rows)
        if max_cols == 0:
            return

        rows = [r + [""] * (max_cols - len(r)) for r in self._table_rows]
        header_flags = [
            flags + [False] * (max_cols - len(flags))
            for flags in self._table_row_header_flags
        ]

        use_first_row_as_header = any(header_flags[0]) if header_flags else False
        if use_first_row_as_header:
            header = rows[0]
            body = rows[1:]
        else:
            header = [f"col_{i + 1}" for i in range(max_cols)]
            body = rows

        self._chunks.append(self._format_markdown_row(header) + "\n")
        self._chunks.append(self._format_markdown_row(["---"] * max_cols) + "\n")
        for row in body:
            self._chunks.append(self._format_markdown_row(row) + "\n")

    def _flush_text_break(self):
        if not self._chunks:
            return
        if not self._chunks[-1].endswith("\n"):
            self._chunks.append("\n")

    @staticmethod
    def _normalize_inline(value: str) -> str:
        return re.sub(r"\s+", " ", value).strip()

    @staticmethod
    def _format_markdown_row(cells: list[str]) -> str:
        escaped = [c.replace("|", "\\|") for c in cells]
        return "| " + " | ".join(escaped) + " |"