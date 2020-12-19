import os
import sys
import unittest
import numpy as np
from numpy.testing import assert_almost_equal, assert_equal, assert_string_equal
from numpy.testing import assert_allclose, assert_raises, assert_raises_regex

this_dir = os.path.dirname(os.path.abspath(__file__)) + '/'
lib_dir = this_dir + '../lib'

sys.path.insert(1, lib_dir)
from recommender_data import RECCOMEND_DATA

class TestExtractDataSlice(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestExtractDataSlice, self).__init__(*args, **kwargs)
        self.r_d = RECCOMEND_DATA()

    def test_extract_data_slice_empty_params(self):
        self.assertTrue(all(self.r_d.extract_data_slice({})), "All Should be true")

    def test_extract_data_slice_invalid_params(self):
        self.assertFalse(any(self.r_d.extract_data_slice(
            {"non-key":['invalid-value']})), "All Should be false")

    def test_extract_data_slice_valid_params(self):
        self.assertTrue(sum(self.r_d.extract_data_slice(
            {"season":['Summer']})) > 0, "Matched should be true")

if __name__ == '__main__':
    unittest.main()
