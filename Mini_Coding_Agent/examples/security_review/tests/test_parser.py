import unittest
from parser import parse_literal


class ParserTests(unittest.TestCase):
    def test_values(self):
        self.assertEqual(parse_literal('[1, 2]'), [1, 2])
        self.assertEqual(parse_literal("'hello'"), 'hello')

    def test_blank(self):
        self.assertIsNone(parse_literal('   '))
