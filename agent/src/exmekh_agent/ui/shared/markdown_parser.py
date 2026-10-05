from __future__ import annotations

import html
import re

import markdown2
import nh3
from markupsafe import Markup

from exmekh_agent.ui.resources import presentation_config, render_html

try:
    from latex2mathml.converter import convert as _latex_to_mathml
except ImportError:
    _latex_to_mathml = None


# MathML is restored only after sanitization; user HTML follows this allowlist.
_SANITIZER_POLICY = presentation_config("markdown_sanitizer")
_ALLOWED_TAGS = set(_SANITIZER_POLICY["tags"])
_ALLOWED_ATTRIBUTES = {
    tag: set(attributes) for tag, attributes in _SANITIZER_POLICY["attributes"].items()
}

_HTML_CLEANER = nh3.Cleaner(
    tags=_ALLOWED_TAGS,
    attributes=_ALLOWED_ATTRIBUTES,
    clean_content_tags={
        "script",
        "style",
        "iframe",
        "object",
        "embed",
    },
    url_schemes={
        "http",
        "https",
        "mailto",
    },
    link_rel="noopener noreferrer",
)


def convert_latex_to_mathml(
    latex_str: str,
    display: str = "inline",
) -> str:
    """Convert LaTeX to MathML.

    Falls back to safely escaped HTML when conversion fails or while
    receiving an incomplete formula during streaming.
    """
    clean_latex = latex_str.strip()
    clean_latex = re.sub(r"[\r\n]+", " ", clean_latex)

    if _latex_to_mathml is not None:
        try:
            return _latex_to_mathml(clean_latex, display=display)
        except Exception:
            pass

    return render_html("shared/math_fallback.html", display=display, latex=clean_latex)


def _close_unfinished_fenced_code(text: str) -> str:
    """Temporarily close an unfinished ``` fence.

    parse_markdown_to_html is called after every streamed token.
    A code fence may therefore be temporarily incomplete.

    We only modify the temporary rendering input; the original response
    text remains untouched.
    """
    inside_fence = False

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            if not inside_fence and len(stripped) >= 6 and stripped.endswith("```"):
                continue
            inside_fence = not inside_fence

    if inside_fence:
        return text + "\n```"

    return text


def _extract_fenced_code_parts(fence_block: str) -> tuple[str, str]:
    """Extract (info_string, raw_code) from a triple-backtick fenced block."""
    if "\n" in fence_block:
        first_line, rest = fence_block.split("\n", 1)
        info_string = re.sub(r"^\s*```+", "", first_line).strip()
        raw_code = re.sub(r"\r?\n?[ \t]*```+\s*$", "", rest)
        return info_string, raw_code

    raw_code = re.sub(r"^\s*```+|```+\s*$", "", fence_block)
    return "", raw_code


def _render_fenced_code_block(info_string: str, raw_code: str) -> str:
    """Render a sanitized <pre><code> block wrapped with its dedicated copy button."""
    normalized_fence = f"```{info_string}\n{raw_code}\n```"
    inner_html = markdown2.markdown(
        normalized_fence,
        safe_mode="escape",
        extras=[
            "fenced-code-blocks",
            "code-friendly",
        ],
    )
    inner_html = _HTML_CLEANER.clean(inner_html).strip()

    escaped_code_attr = (
        html.escape(raw_code, quote=True)
        .replace("\r\n", "\n")
        .replace("\r", "\n")
        .replace("\n", "&#10;")
    )

    return render_html(
        "shared/code_block.html",
        code=Markup(escaped_code_attr),
        content=Markup(inner_html),
    )


def parse_markdown_to_html(text: str) -> str:
    """Convert streamed Markdown + LaTeX into safe HTML."""

    if not text:
        return ""

    # The source stored by ChatView remains untouched.
    # This copy is only used for the current rendering frame.
    text = _close_unfinished_fenced_code(text)

    # ------------------------------------------------------------------
    # 1. Protect code from the math regexes.
    # ------------------------------------------------------------------

    fenced_fragments: dict[str, str] = {}

    def preserve_fenced_code(match: re.Match[str]) -> str:
        key = f"FENCEDBLOCKPLACEHOLDER{len(fenced_fragments)}TOKEN"
        info_string, raw_code = _extract_fenced_code_parts(match.group(0))
        fenced_fragments[key] = _render_fenced_code_block(info_string, raw_code)
        return f"\n\n{key}\n\n"

    processed = re.sub(
        r"```[\s\S]*?```",
        preserve_fenced_code,
        text,
    )

    code_blocks: dict[str, str] = {}

    def preserve_code(match: re.Match[str]) -> str:
        key = f"CODEBLOCKPLACEHOLDER{len(code_blocks)}TOKEN"
        code_blocks[key] = match.group(0)
        return key

    processed = re.sub(
        r"`[^`\n]+?`",
        preserve_code,
        processed,
    )

    # ------------------------------------------------------------------
    # 2. Protect generated MathML as trusted fragments.
    #
    # Do NOT put the MathML directly through markdown2 because
    # safe_mode="escape" would correctly escape it as raw HTML.
    # ------------------------------------------------------------------

    math_fragments: dict[str, tuple[str, bool]] = {}

    def preserve_math(fragment: str, *, block: bool) -> str:
        key = f"MATHPLACEHOLDER{len(math_fragments)}TOKEN"
        math_fragments[key] = (fragment, block)

        if block:
            return f"\n\n{key}\n\n"

        return key

    # ------------------------------------------------------------------
    # 3. Display math: $$...$$
    # ------------------------------------------------------------------

    def replace_display_math(match: re.Match[str]) -> str:
        latex_code = match.group(1)
        mathml = convert_latex_to_mathml(
            latex_code,
            display="block",
        )

        fragment = render_html("shared/math_block.html", mathml=Markup(mathml))

        return preserve_math(fragment, block=True)

    processed = re.sub(
        r"\$\$(.*?)\$\$",
        replace_display_math,
        processed,
        flags=re.DOTALL,
    )

    # ------------------------------------------------------------------
    # 4. Display math: \[...\]
    # ------------------------------------------------------------------

    processed = re.sub(
        r"\\\[(.*?)\\\]",
        replace_display_math,
        processed,
        flags=re.DOTALL,
    )

    # ------------------------------------------------------------------
    # 5. Inline math: $...$
    # ------------------------------------------------------------------

    def replace_inline_math(match: re.Match[str]) -> str:
        content = match.group(1)
        stripped = content.strip()

        # Do not interpret common currency values as math.
        if stripped.replace(".", "", 1).replace(",", "", 1).isdigit():
            return match.group(0)

        mathml = convert_latex_to_mathml(
            content,
            display="inline",
        )

        fragment = render_html("shared/math_inline.html", mathml=Markup(mathml))

        return preserve_math(fragment, block=False)

    processed = re.sub(
        r"(?<!\$)\$(?!\$)(?!\s)(.*?)(?<!\s)(?<!\$)\$(?!\$)",
        replace_inline_math,
        processed,
    )

    # ------------------------------------------------------------------
    # 6. Inline math: \(...\)
    # ------------------------------------------------------------------

    def replace_parenthesis_math(match: re.Match[str]) -> str:
        content = match.group(1)

        mathml = convert_latex_to_mathml(
            content,
            display="inline",
        )

        fragment = render_html("shared/math_inline.html", mathml=Markup(mathml))

        return preserve_math(fragment, block=False)

    processed = re.sub(
        r"\\\((.*?)\\\)",
        replace_parenthesis_math,
        processed,
        flags=re.DOTALL,
    )

    # ------------------------------------------------------------------
    # 7. Restore Markdown code blocks.
    #
    # markdown2 itself must see ```...``` so that it generates
    # <pre><code> and escapes things like <script>.
    # ------------------------------------------------------------------

    for placeholder, original in code_blocks.items():
        processed = processed.replace(placeholder, original)

    # ------------------------------------------------------------------
    # 8. Markdown -> HTML
    #
    # safe_mode is defense-in-depth, not our final sanitizer.
    # ------------------------------------------------------------------

    html_output = markdown2.markdown(
        processed,
        safe_mode="escape",
        extras=[
            "fenced-code-blocks",
            "tables",
            "code-friendly",
            "strike",
            "break-on-newline",
            "cuddled-lists",
        ],
    )

    # ------------------------------------------------------------------
    # 9. Final security boundary.
    #
    # This removes scripts, event-handler attributes, dangerous
    # protocols, iframes, etc.
    # ------------------------------------------------------------------

    html_output = _HTML_CLEANER.clean(html_output)

    # ------------------------------------------------------------------
    # 10. Restore trusted MathML.
    #
    # latex2mathml output originates from our converter rather than from
    # arbitrary user/model supplied raw HTML.
    # ------------------------------------------------------------------

    for placeholder, (fragment, is_block) in math_fragments.items():
        if is_block:
            # markdown2 normally wraps a standalone placeholder in <p>.
            html_output = html_output.replace(
                f"<p>{placeholder}</p>",
                fragment,
            )

        html_output = html_output.replace(
            placeholder,
            fragment,
        )

    for placeholder, fragment in fenced_fragments.items():
        html_output = html_output.replace(
            f"<p>{placeholder}</p>",
            fragment,
        )
        html_output = html_output.replace(
            placeholder,
            fragment,
        )

    return html_output
