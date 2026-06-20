import unittest
from lirien import verify, i64

class TestDiagnostics(unittest.TestCase):
    def test_no_verification_context_manager(self):
        from lirien import no_verification

        with no_verification():
            @verify
            def unsafe_div_ctx(n: i64, d: i64) -> i64:
                return n // d

        self.assertTrue(getattr(unsafe_div_ctx, "__lirien_jit__", False))
        self.assertEqual(unsafe_div_ctx(10, 2), 5)

    def test_tracing_context_manager(self):
        from lirien import tracing, BRIDGE

        class CaptureOutput:
            def __enter__(self):
                import os
                import tempfile

                self.stdout_fd = 1
                self.stderr_fd = 2
                self.saved_out = os.dup(self.stdout_fd)
                self.saved_err = os.dup(self.stderr_fd)
                self.temp_file = tempfile.TemporaryFile(mode="w+b")
                os.dup2(self.temp_file.fileno(), self.stdout_fd)
                os.dup2(self.temp_file.fileno(), self.stderr_fd)
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                import os

                os.dup2(self.saved_out, self.stdout_fd)
                os.dup2(self.saved_err, self.stderr_fd)
                os.close(self.saved_out)
                os.close(self.saved_err)
                self.temp_file.seek(0)
                self.output = self.temp_file.read().decode("utf-8")
                self.temp_file.close()

        import re

        def strip_ansi(text):
            ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
            return ansi_escape.sub("", text)

        # 1. Compile a function with tracing active, capture output
        with CaptureOutput() as captured_traced:
            with tracing({BRIDGE: "debug"}):
                @verify
                def dummy_func_1(x: i64) -> i64:
                    return x + 1

        traced_clean = strip_ansi(captured_traced.output)
        # Verify that DEBUG bridge logs are present in the output
        self.assertIn("DEBUG lirien::bridge: Struct layouts:", traced_clean)

        # 2. Compile a function without tracing active, capture output
        with CaptureOutput() as captured_untraced:
            @verify
            def dummy_func_2(x: i64) -> i64:
                return x + 1

        untraced_clean = strip_ansi(captured_untraced.output)
        # Verify that DEBUG bridge logs are NOT present in the output
        self.assertNotIn("DEBUG lirien::bridge: Struct layouts:", untraced_clean)

        # Verify both functions still work correctly
        self.assertEqual(dummy_func_1(42), 43)
        self.assertEqual(dummy_func_2(42), 43)

if __name__ == "__main__":
    unittest.main()
