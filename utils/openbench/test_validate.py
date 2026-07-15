#!/usr/bin/env python3

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("validate.py")
SPEC = importlib.util.spec_from_file_location("sykora_openbench_validate", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
validate = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = validate
SPEC.loader.exec_module(validate)


class OpenBenchValidationTests(unittest.TestCase):
    def test_repository_parameters_are_valid(self):
        parameters = validate.load_parameters(Path(__file__).with_name("spsa.txt"))
        self.assertEqual(4, len(parameters))
        self.assertTrue(all(parameter.kind == "int" for parameter in parameters))

    def test_parameter_names_must_be_openbench_safe(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "bad.txt"
            path.write_text("LMR Scale, int, 100, 50, 200, 7.5, 0.002\n")
            with self.assertRaises(validate.ValidationError):
                validate.load_parameters(path)

    def test_uci_contract_matches_parameter_file(self):
        parameters = validate.load_parameters(Path(__file__).with_name("spsa.txt"))
        lines = [
            "option name Threads type spin default 1 min 1 max 64",
            "option name Hash type spin default 128 min 1 max 4096",
        ]
        lines.extend(
            f"option name {p.name} type spin default {int(p.current)} min {int(p.minimum)} max {int(p.maximum)}"
            for p in parameters
        )
        validate.validate_uci_output("\n".join(lines), parameters)

    def test_bench_parser_uses_final_positive_values(self):
        self.assertEqual((585082, 1300000), validate.parse_bench_output("585082 nodes 1300000 nps\n"))
        with self.assertRaises(validate.ValidationError):
            validate.parse_bench_output("benchmark unavailable\n")


if __name__ == "__main__":
    unittest.main()
