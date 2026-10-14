"""Architectural regression: product text belongs to its configuration owner."""

import unittest
from pathlib import Path

from tools.check_product_text import (
    c_product_literals,
    python_product_literals,
    violations,
)


class ProductTextConfigurationTests(unittest.TestCase):
    def test_repository_production_text_has_no_inline_violations(self):
        self.assertEqual(violations(Path(__file__).resolve().parents[1]), ())

    def test_python_guard_includes_logs_schema_metadata_and_formatted_tooltips(self):
        source = '''"Module documentation."
logger.warning("Inline warning")
Field(description="Meaning")
tooltip = f"Context tokens before closing: {count}"'''
        found = python_product_literals(source)
        self.assertEqual(len(found), 3)
        self.assertEqual({item.start for item in found}, {2, 3, 4})
        self.assertFalse(python_product_literals('message = app_text("chat.context")'))

    def test_c_guard_combines_literals_but_excludes_comments_wire_and_native_fixtures(
        self,
    ):
        source = """static void render(void) {
    // Documented instruction in a comment.
    const char *wire = "<tool_call>";
    const char *notice = "Inline " "message";
}
#else

static int test_failures = 0;
const char *fixture = "Test fixture content";
"""
        found = c_product_literals(source)
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].value, "Inline message")

    def test_diagnostics_are_checked_after_long_format_arguments(self):
        source = (
            "server_log(level, "
            + "CONFIGURED_LONG_FORMAT " * 30
            + ', ready ? "yes" : "no");'
        )
        self.assertEqual(
            {item.value for item in c_product_literals(source)}, {"yes", "no"}
        )


if __name__ == "__main__":
    unittest.main()
