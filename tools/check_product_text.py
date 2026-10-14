"""Guard product prose in production code; comments and protocol syntax stay in code."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path

C_TOKEN = re.compile(
    r"//[^\n]*|/\*[\s\S]*?\*/|'(?:\\.|[^'\\])*'|\"(?:\\.|[^\"\\])*\"", re.DOTALL
)
C_LEXEME = re.compile(C_TOKEN.pattern + r"|[A-Za-z_]\w*|[()]", re.DOTALL)
C_FUNCTION = re.compile(r"(?m)^static\s+[\w *]+?\b(\w+)\s*\(")
PROSE = re.compile(r"[A-Za-z]{2,} [A-Za-z]{2,}")
SERVER_TEST_BOUNDARY = "\n#else\n\nstatic int test_failures = 0;"
PROMPT_FUNCTIONS = {
    "prepend_response_schema_instruction",
    "append_tools_prompt_text",
    "append_qwen_tools_prompt_text",
    "append_glm_tools_prompt_text",
    "append_qwen4_tools_prompt_text",
}


@dataclass(frozen=True, slots=True)
class ProductLiteral:
    start: int
    end: int
    value: str
    function: str = ""


def has_prose(value: str) -> bool:
    # Metadata, JSON field names, markup attributes and data URLs are wire syntax.
    if value.startswith("<") and ">" not in value:
        return False
    value = re.sub(r"<[^>]*>", "", value)
    if value.startswith(("data:", "role=", "noopener ")):
        return False
    return PROSE.search(value) is not None


def c_literal_calls(source: str) -> dict[int, tuple[str, ...]]:
    """Track enclosing calls without a distance limit or compiler dependency."""
    stack: list[tuple[str, int]] = []
    calls: dict[int, tuple[str, ...]] = {}
    previous = ""
    for token in C_LEXEME.finditer(source):
        value = token[0]
        if value.startswith(("//", "/*")):
            continue
        if value == "(":
            stack.append((previous, token.end()))
        elif value == ")":
            if stack:
                stack.pop()
        elif value.startswith('"'):
            calls[token.start()] = tuple(
                name
                for name, start in stack
                if name != "fprintf"
                or re.match(r"\s*stderr\s*,", source[start : token.start()])
            )
        previous = value
    return calls


def c_product_literals(source: str) -> tuple[ProductLiteral, ...]:
    boundary = source.find(SERVER_TEST_BOUNDARY)
    production = source if boundary < 0 else source[:boundary]
    functions = list(C_FUNCTION.finditer(production))
    calls = c_literal_calls(production)
    strings = [
        match for match in C_TOKEN.finditer(production) if match[0].startswith('"')
    ]
    groups: list[list[re.Match[str]]] = []
    for match in strings:
        if groups and not production[groups[-1][-1].end() : match.start()].strip():
            groups[-1].append(match)
        else:
            groups.append([match])
    result = []
    for group in groups:
        start, end = group[0].start(), group[-1].end()
        value = "".join(ast.literal_eval(match[0]) for match in group)
        function = next(
            (match[1] for match in reversed(functions) if match.start() < start), ""
        )
        enclosing = calls.get(start, ())
        prompt_piece = function in PROMPT_FUNCTIONS and "buf_puts" in enclosing
        prompt_piece = prompt_piece and not value.lstrip().startswith(("{", "["))
        diagnostic = any(
            name in {"server_log", "die", "perror", "fprintf"} for name in enclosing
        )
        visible = re.sub(
            r"%[-+#0-9. ]*(?:hh|ll|h|l|z|t)?[diuoxXfeEgGaAcspn%]", "", value
        )
        diagnostic = diagnostic and re.search(r"[A-Za-z]", visible) is not None
        if has_prose(value) or prompt_piece or diagnostic:
            result.append(ProductLiteral(start, end, value, function))
    return tuple(result)


def python_product_literals(source: str) -> tuple[ProductLiteral, ...]:
    tree = ast.parse(source)
    documents: set[int] = set()
    for node in ast.walk(tree):
        if (
            isinstance(
                node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
            )
            and node.body
        ):
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                documents.add(id(first.value))
    result = []
    seen: set[int] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in documents
            and id(node) not in seen
        ):
            value = node.value
            # HTML/JS/CSS structure is presentation syntax, not a prose phrase.
            syntax = any(
                marker in value
                for marker in (
                    "=>",
                    "return ",
                    "function(",
                    ".classes.",
                    "aria-label=",
                    "tabindex=",
                )
            )
            if has_prose(value) and not syntax:
                result.append(
                    ProductLiteral(node.lineno, node.end_lineno or node.lineno, value)
                )
                seen.add(id(node))
        if isinstance(node, ast.Call):
            literal = None
            if (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in {"logger", "logging"}
                and node.func.attr
                in {"debug", "info", "warning", "error", "exception", "critical"}
                and node.args
            ):
                literal = node.args[0]
            for keyword in node.keywords:
                if keyword.arg == "description":
                    literal = keyword.value
            if (
                isinstance(literal, ast.Constant)
                and isinstance(literal.value, str)
                and literal.value
                and id(literal) not in seen
            ):
                result.append(
                    ProductLiteral(
                        literal.lineno,
                        literal.end_lineno or literal.lineno,
                        literal.value,
                    )
                )
                seen.add(id(literal))
    return tuple(result)


def violations(root: Path) -> tuple[str, ...]:
    found = []
    server_source = (root / "ds4_server.c").read_text()
    for literal in c_product_literals(server_source):
        line = server_source.count("\n", 0, literal.start) + 1
        found.append(f"ds4_server.c:{line}: {literal.function}")
    for directory in ("agent/src/exmekh_agent", "exmekh_lib/src/exmekh_lib"):
        for path in (root / directory).rglob("*.py"):
            if "config" in path.relative_to(root / directory).parts:
                continue
            for literal in python_product_literals(path.read_text()):
                found.append(f"{path.relative_to(root)}:{literal.start}")
    return tuple(found)


def main() -> int:
    found = violations(Path(__file__).resolve().parents[1])
    for message in found:
        print(message)
    return int(bool(found))


if __name__ == "__main__":
    raise SystemExit(main())
