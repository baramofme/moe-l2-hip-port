"""Placeholder — moe-l2 tests."""
import unittest


class TestPredictor(unittest.TestCase):
    def test_predict_codegen(self):
        from moe_l2.predictor import predict
        self.assertEqual(predict("implement a sorting algorithm"), "codegen")

    def test_predict_math(self):
        from moe_l2.predictor import predict
        self.assertEqual(predict("calculate the derivative"), "math")

    def test_predict_fallback(self):
        from moe_l2.predictor import predict
        self.assertEqual(predict("hello world"), "general_qa")


if __name__ == "__main__":
    unittest.main()
