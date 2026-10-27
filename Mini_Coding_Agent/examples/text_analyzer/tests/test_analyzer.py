import unittest
from analyzer import top_words


class WordStatisticsTests(unittest.TestCase):
    def test_punctuation_and_frequencies(self):
        self.assertEqual(top_words('Cat, dog! cat. DOG dog?'), [('dog', 3), ('cat', 2)])

    def test_ties_and_limit(self):
        self.assertEqual(top_words('pear apple pear apple fig', 2), [('apple', 2), ('pear', 2)])

    def test_empty_and_zero(self):
        self.assertEqual(top_words('...', 5), [])
        self.assertEqual(top_words('apple apple', 0), [])

    def test_invalid_limit(self):
        with self.assertRaises(ValueError):
            top_words('apple', -1)

    def test_unicode_and_underscore(self):
        self.assertEqual(top_words('猫_猫 café CAFÉ １２'), [('café', 2), ('猫', 2), ('１２', 1)])
