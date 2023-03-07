import unittest
from typing import List, Tuple

def equal_slice(overall:int, num_slices:int)-> List[Tuple[int, int]]:

    res = []
    start = 0
    num_units = overall // num_slices
    for i in range(num_slices):
        end = start + num_units if i < num_slices - 1 else overall
        res.append((start, end))
        start = end
    return res


class TestUtils(unittest.TestCase):
    def test_slice(self):
        overall = 4
        num_slices= 2
        self.assertEqual(equal_slice(overall, num_slices), [(0,2), (2,4)])
        overall=5
        num_slices=2
        self.assertEqual(equal_slice(overall, num_slices), [(0,2), (2,5)])
        overall=10
        num_slices=2
        self.assertEqual(equal_slice(overall, num_slices), [(0,5), (5,10)])

if __name__ == "__main__":
    unittest.main()