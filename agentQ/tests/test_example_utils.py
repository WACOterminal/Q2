
import unittest
from agentQ.app.core.example_utils import add_numbers

class TestExampleUtils(unittest.TestCase):

    def test_add_numbers(self):
        """
        Tests the add_numbers function.
        """
        self.assertEqual(add_numbers(2, 3), 5)
        self.assertEqual(add_numbers(-1, 1), 0)
        self.assertEqual(add_numbers(0, 0), 0)

if __name__ == '__main__':
    unittest.main() 