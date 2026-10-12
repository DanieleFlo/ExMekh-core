"""Decode measurements exclude restored output and unrelated tool turns."""
import unittest

from tests.bench_interactive_edit_live import edit_decode_measurements


class EditBenchmarkTests(unittest.TestCase):
    def test_resume_counts_only_new_tokens_and_includes_final_progress(self):
        log = """ds4-server: chat ctx=0..100:100 prompt start
ds4-server: chat ctx=100..110:10 gen=10 TOOLS decoding chunk=10 t/s
ds4-server: phase profile ctx=0..100:100 gen=10 mode=trie wall=1000ms
ds4-server: tool calls ctx=0..100:100 names=[read-file]
ds4-server: chat ctx=200..158:0 prompt start
ds4-server: chat ctx=200..208:8 gen=50 TOOLS decoding chunk=8 t/s
ds4-server: old_text drafts cycles=1 proposed=31 accepted=31 batches=32/64
ds4-server: phase profile ctx=200..158:0 gen=100 mode=trie wall=2000ms
ds4-server: chat ctx=208..258:50 gen=100 TOOLS decoding chunk=50 t/s
ds4-server: tool calls ctx=200..158:0 names=[edit-file]
"""
        records = edit_decode_measurements(log)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["new_decode_tokens"], 58)
        self.assertEqual(records[0]["restored_output_tokens"], 42)
        self.assertEqual(records[0]["decode_tokens_per_second"], 29)
        self.assertEqual(records[0]["snapshot_drafts"], {"cycles": 1, "proposed": 31, "accepted": 31})

    def test_missing_phase_is_unmeasured(self):
        self.assertEqual(edit_decode_measurements("tool calls ctx=0..0:0 names=[edit-file]"), [])

    def test_missing_decode_tokens_rejects_the_measurement(self):
        with self.assertRaises(ValueError):
            edit_decode_measurements("phase profile ctx=0..0:0 gen=42 mode=trie wall=1ms\n"
                                     "tool calls ctx=0..0:0 names=[edit-file]")
