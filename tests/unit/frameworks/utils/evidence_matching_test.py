# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import unittest

import numpy as np

from tbp.monty.frameworks.utils.evidence_matching import (
    ChannelMapper,
    EvidenceSlopeTracker,
)


class ChannelMapperTest(unittest.TestCase):
    def setUp(self) -> None:
        """Sets up a default ChannelMapper instance for testing."""
        self.mapper = ChannelMapper({"A": 5, "B": 10, "C": 15})

    def test_initialization(self):
        """Test initializing ChannelMapper with predefined sizes."""
        self.assertEqual(self.mapper.channels, ["A", "B", "C"])
        self.assertEqual(self.mapper.total_size, 30)

    def test_channel_range(self):
        """Test retrieving channel ranges and non-existent channels."""
        self.assertEqual(self.mapper.channel_range("A"), (0, 5))
        self.assertEqual(self.mapper.channel_range("B"), (5, 15))
        self.assertEqual(self.mapper.channel_range("C"), (15, 30))
        with self.assertRaises(ValueError):
            self.mapper.channel_range("D")

    def test_resize_channel_by_positive(self):
        """Test increasing channel sizes."""
        self.mapper.resize_channel_by("B", 5)
        self.assertEqual(self.mapper.channel_range("B"), (5, 20))
        self.assertEqual(self.mapper.total_size, 35)

    def test_resize_channel_by_negative(self):
        """Test decreasing channel sizes."""
        self.mapper.resize_channel_by("B", -5)
        self.assertEqual(self.mapper.channel_range("B"), (5, 10))
        self.assertEqual(self.mapper.total_size, 25)

        with self.assertRaises(ValueError):
            self.mapper.resize_channel_by("A", -10)

    def test_add_channel(self):
        """Test adding a new channel."""
        self.mapper.add_channel("D", 8)
        self.assertIn("D", self.mapper.channels)
        self.assertEqual(self.mapper.channel_range("D"), (30, 38))

        with self.assertRaises(ValueError):
            self.mapper.add_channel("A", 3)

    def test_add_channel_at_position(self):
        """Test inserting a channel at a specific position."""
        self.mapper.add_channel("X", 7, position=1)
        self.assertEqual(self.mapper.channels, ["A", "X", "B", "C"])
        self.assertEqual(self.mapper.channel_range("X"), (5, 12))
        self.assertEqual(self.mapper.channel_range("B"), (12, 22))
        self.assertEqual(self.mapper.channel_range("C"), (22, 37))

        with self.assertRaises(ValueError):
            self.mapper.add_channel("Y", 5, position=10)

    def test_repr(self):
        """Test string representation of the ChannelMapper."""
        expected_repr = "ChannelMapper({'A': (0, 5), 'B': (5, 15), 'C': (15, 30)})"
        self.assertEqual(repr(self.mapper), expected_repr)


class EvidenceSlopeTrackerTest(unittest.TestCase):
    def setUp(self) -> None:
        """Set up a default tracker for testing."""
        self.tracker = EvidenceSlopeTracker(window_size=3, min_age=2)

    def test_initialization(self) -> None:
        """Test that the tracker initializes correctly with default values."""
        self.assertEqual(self.tracker.window_size, 3)
        self.assertEqual(self.tracker.min_age, 2)
        self.assertEqual(self.tracker.total_size, 0)
        self.assertEqual(len(self.tracker.age), 0)

    def test_add_hyp(self) -> None:
        """Test adding new hypotheses updates data and age arrays."""
        self.tracker.add_hyp(2)
        self.assertEqual(self.tracker.total_size, 2)
        self.assertEqual(self.tracker.data.shape, (2, 3))
        self.assertTrue(np.all(np.isnan(self.tracker.data)))
        self.assertTrue(np.all(self.tracker.age == 0))

    def test_remove_hyp(self) -> None:
        """Test removing hypotheses by index."""
        # add 3 hypotheses to the space
        self.tracker.add_hyp(3)
        self.assertEqual(self.tracker.total_size, 3)

        # push values for the 3 hypotheses
        self.tracker.update([1.0, 2.0, 3.0])

        # remove hypothesis at index 1
        self.tracker.remove_hyp([1])
        self.assertEqual(self.tracker.total_size, 2)
        self.assertTrue(np.allclose(self.tracker.data[:, -1], [1.0, 3.0]))

    def test_update_valid(self) -> None:
        """Test valid update of evidence values and age increment."""
        self.tracker.add_hyp(2)
        self.tracker.update([0.1, 0.2])
        self.assertListEqual(self.tracker.data[:, -1].tolist(), [0.1, 0.2])
        self.assertListEqual(self.tracker.age.tolist(), [1, 1])

        # Second update
        self.tracker.update([0.2, 0.3])
        self.assertListEqual(self.tracker.data[:, -1].tolist(), [0.2, 0.3])
        self.assertListEqual(self.tracker.age.tolist(), [2, 2])

    def test_update_invalid_length(self) -> None:
        """Test that update raises error on mismatched input length.

        Updates should be of the same lengths as the number of hypotheses.
        """
        self.tracker.add_hyp(2)
        with self.assertRaises(ValueError):
            self.tracker.update([1.0])

    def test_get_slopes(self) -> None:
        """Test slope calculation from data."""
        self.tracker.add_hyp(1)
        self.tracker.update([1.0])
        self.tracker.update([2.0])
        self.tracker.update([3.0])

        # Expected slope is ((2.0 - 1.0) + (3.0 - 2.0)) / 2 = 1.0
        slope = self.tracker._get_slopes()
        self.assertEqual(slope[0], 1.0)

    def test_valid_and_must_keep_masks(self) -> None:
        """Test that valid and must_keep masks are computed correctly."""
        self.tracker.add_hyp(3)
        self.tracker.age[:] = [1, 2, 3]
        self.assertTrue(
            np.array_equal(self.tracker.valid_indices_mask, [False, True, True])
        )
        self.assertTrue(
            np.array_equal(self.tracker.must_keep_mask, [True, False, False])
        )

    def test_topk_hyp(self) -> None:
        """Test top-k selection in ascending and descending order."""
        self.tracker.add_hyp(3)
        self.tracker.update([1, 3, 1])
        self.tracker.update([2, 2, 1])
        self.tracker.update([3, 1, 1])

        # Expected sloped are: [1.0, -1.0, 0.0]
        self.assertListEqual(self.tracker._get_slopes().tolist(), [1.0, -1.0, 0.0])

        # Test top-2 in ascending order
        top_2_ascending = self.tracker.topk_hyp(2, order="ascending")
        self.assertListEqual(top_2_ascending.tolist(), [1, 2])

        # Test top-2 in descending order
        top_2_descending = self.tracker.topk_hyp(2, order="descending")
        self.assertListEqual(top_2_descending.tolist(), [0, 2])

    def test_topk_hyp_valid_filtering(self) -> None:
        """Test that invalid hypotheses are excluded from top-k."""
        # data = [[3]]
        self.tracker.add_hyp(1)
        self.tracker.update([3])

        # data = [[3,2],[nan,0]]
        self.tracker.add_hyp(1)
        self.tracker.update([2, 0])

        # data = [[3,2,1],[nan,0,2],[nan,nan,3]]
        self.tracker.add_hyp(1)
        self.tracker.update([1, 2, 3])

        # ages = [3,2,1]
        self.assertListEqual(self.tracker.age.tolist(), [3, 2, 1])

        # Slopes are [-1.0, 2.0, np.nan]
        # index 3 is not valid due to min_age=2
        self.assertTrue(np.array_equal(self.tracker.topk_hyp(3), [1, 0]))


if __name__ == "__main__":
    unittest.main()
