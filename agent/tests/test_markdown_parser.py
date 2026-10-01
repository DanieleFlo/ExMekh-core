from __future__ import annotations

from fe.ui.markdown_parser import convert_latex_to_mathml, parse_markdown_to_html


def test_user_integral_example_display_and_inline() -> None:
    prompt_answer = r"""Calcola l'integrale di $x^2$ passo passo in $dx$:

$$
\int x^2 \, dx = \frac{x^3}{3} + C
$$

Passo dopo passo:
1. Applica la formula: $\int x^n \, dx = \frac{x^{n+1}}{n+1} + C$
2. Per $n = 2$:
$$
\frac{x^{2+1}}{2+1} = \frac{x^3}{3}
$$
"""
    html_out = parse_markdown_to_html(prompt_answer)
    assert "<math" in html_out
    assert "$$" not in html_out
    assert "math-display" in html_out
    assert "math-inline" in html_out
    # Check that integral symbol is present in mathml
    assert "&#x0222B;" in html_out or "∫" in html_out


def test_brackets_notation() -> None:
    text = r"Formula di Einstein: \[ E = mc^2 \] ed energia cinetica: \( K = \frac{1}{2}mv^2 \)"
    html_out = parse_markdown_to_html(text)
    assert "<math" in html_out
    assert r"\[" not in html_out
    assert r"\(" not in html_out


def test_currency_preservation() -> None:
    text = "Il servizio costa $100 all'anno oppure $50 per sei mesi."
    html_out = parse_markdown_to_html(text)
    assert "$100" in html_out
    assert "$50" in html_out
    assert "<math" not in html_out


def test_code_block_dollar_protection() -> None:
    text = """```bash
export PATH=$PATH:/usr/local/bin
echo $USER
```
Esegui `$HOME` nel terminale."""
    html_out = parse_markdown_to_html(text)
    assert "$PATH" in html_out
    assert "$USER" in html_out
    assert "$HOME" in html_out
    assert "<math" not in html_out


def test_markdown_tables_and_formatting() -> None:
    text = """| Proprietà | Valore |
| --- | --- |
| Funzione | $f(x) = x^2$ |
| Derivata | $f'(x) = 2x$ |

- Elemento **importante**
- Elemento ~~cancellato~~
"""
    html_out = parse_markdown_to_html(text)
    assert "<table>" in html_out
    assert "<th>Proprietà</th>" in html_out
    assert "<strong>importante</strong>" in html_out
    assert "<s>cancellato</s>" in html_out or "<del>cancellato</del>" in html_out
    assert "<math" in html_out


def test_streaming_incomplete_math_no_crash() -> None:
    incomplete = r"Sto calcolando $$ \int x^2"
    html_out = parse_markdown_to_html(incomplete)
    assert len(html_out) > 0


def test_empty_string() -> None:
    assert parse_markdown_to_html("") == ""
    assert parse_markdown_to_html(None) == ""


def test_convert_latex_fallback() -> None:
    # Malformed LaTeX should produce a safe fallback without crashing
    res = convert_latex_to_mathml(r"\invalid_command{}}{{{", display="inline")
    assert "math-fallback" in res


def test_fenced_code_and_special_text_blocks_have_dedicated_copy_button() -> None:
    text = """Ecco un blocco Python:
```python
def somma(a: int, b: int) -> int:
    return a + b
```

Ed ecco una sezione di testo speciale senza linguaggio:
```
Riga 1 del testo speciale
Riga 2 con caratteri < & > " '
```

Testo inline con `singolo_apice` non deve avere bottone."""
    html_out = parse_markdown_to_html(text)
    assert html_out.count('class="code-block-wrapper"') == 2
    assert html_out.count('class="code-copy-btn"') == 2
    assert ">content_copy</i>" in html_out
    assert (
        'data-code="def somma(a: int, b: int) -&gt; int:&#10;    return a + b"'
        in html_out
    )
    assert (
        'data-code="Riga 1 del testo speciale&#10;Riga 2 con caratteri &lt; &amp; &gt; &quot; &#x27;"'
        in html_out
    )
    assert "Ecco un blocco Python" not in html_out.split('data-code="')[1].split('"')[0]
    assert "<code>singolo_apice</code>" in html_out
