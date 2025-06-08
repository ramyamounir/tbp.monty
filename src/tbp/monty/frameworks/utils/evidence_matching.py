# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List, Optional, Tuple
from typing import OrderedDict as OrderedDictType

import numpy as np


class ChannelMapper:
    """Marks the range of hypotheses that correspond to each input channel.

    The `EvidenceGraphLM` implementation stacks the hypotheses from all input channels
    in the same array to perform efficient vector operations on them. Therefore, we
    need to keep track of which indices in the stacked array correspond to which input
    channel. This class stores only the sizes of the input channels in an ordered data
    structure (OrderedDict), and computes the range of indices for each channel. Storing
    the sizes of channels in an ordered dictionary allows us to insert or remove
    channels, as well as dynamically resize them.

    """

    def __init__(self, channel_sizes: Optional[Dict[str, int]] = None) -> None:
        """Initializes the ChannelMapper with an ordered dictionary of channel sizes.

        Args:
            channel_sizes (Optional[Dict[str, int]]): Dictionary of
                {channel_name: size}.
        """
        self.channel_sizes: OrderedDictType[str, int] = (
            OrderedDict(channel_sizes) if channel_sizes else OrderedDict()
        )

    @property
    def channels(self) -> List[str]:
        """Returns the existing channel names.

        Returns:
            List[str]: List of channel names.
        """
        return list(self.channel_sizes.keys())

    @property
    def total_size(self) -> int:
        """Returns the total number of hypotheses across all channels.

        Returns:
            int: Total size across all channels.
        """
        return sum(self.channel_sizes.values())

    def channel_size(self, channel_name: str) -> int:
        """Returns the total number of hypotheses for a specific channel.

        Returns:
            int: Size of channel

        Raises:
            ValueError: If the channel is not found.
        """
        if channel_name not in self.channel_sizes:
            raise ValueError(f"Channel '{channel_name}' not found.")

        return self.channel_sizes[channel_name]

    def channel_range(self, channel_name: str) -> Tuple[int, int]:
        """Returns the start and end indices of the given channel.

        Args:
            channel_name (str): The name of the channel.

        Returns:
            Tuple[int, int]: The start and end indices of the channel.

        Raises:
            ValueError: If the channel is not found.
        """
        if channel_name not in self.channel_sizes:
            raise ValueError(f"Channel '{channel_name}' not found.")

        start = 0
        for name, size in self.channel_sizes.items():
            if name == channel_name:
                return (start, start + size)
            start += size

    def resize_channel_by(self, channel_name: str, value: int) -> None:
        """Increases or decreases the channel by a specific amount.

        Args:
            channel_name (str): The name of the channel.
            value (int): The value used to modify the channel size.
                Use a negative value to decrease the size.

        Raises:
            ValueError: If the channel is not found or the requested size is negative.
        """
        if channel_name not in self.channel_sizes:
            raise ValueError(f"Channel '{channel_name}' not found.")
        if self.channel_sizes[channel_name] + value <= 0:
            raise ValueError(
                f"Channel '{channel_name}' size cannot be negative or zero."
            )
        self.channel_sizes[channel_name] += value

    def resize_channel_to(self, channel_name: str, new_size: int) -> None:
        """Sets the size of the given channel to a specific value.

        Args:
            channel_name (str): The name of the channel.
            new_size (int): The new size to set for the channel.

        Raises:
            ValueError: If the channel is not found or if the new size is not positive.
        """
        if channel_name not in self.channel_sizes:
            raise ValueError(f"Channel '{channel_name}' not found.")
        if new_size <= 0:
            raise ValueError(f"Channel '{channel_name}' size must be positive.")
        self.channel_sizes[channel_name] = new_size

    def add_channel(
        self, channel_name: str, size: int, position: Optional[int] = None
    ) -> None:
        """Adds a new channel at a specified position (default is at the end).

        Args:
            channel_name (str): The name of the new channel.
            size (int): The size of the new channel.
            position (Optional[int]): The index at which to insert the channel.

        Raises:
            ValueError: If the channel already exists or position is out of bounds.
        """
        if channel_name in self.channel_sizes:
            raise ValueError(f"Channel '{channel_name}' already exists.")

        if isinstance(position, int) and position >= len(self.channel_sizes):
            raise ValueError(f"Position index '{position}' is out of bounds.")

        if position is None:
            self.channel_sizes[channel_name] = size
        else:
            items = list(self.channel_sizes.items())
            items.insert(position, (channel_name, size))
            self.channel_sizes = OrderedDict(items)

    def extract(self, original: np.ndarray, channel: str) -> np.ndarray:
        """Extracts the portion of the original array corresponding to a given channel.

        Args:
            original (np.ndarray): The full hypotheses array across all channels.
            channel (str): The name of the channel to extract.

        Returns:
            np.ndarray: The extracted slice of the original array. Returns a view, not
                a copy of the original array.

        Raises:
            ValueError: If the channel is not found.
        """
        if channel not in self.channel_sizes:
            raise ValueError(f"Channel '{channel}' not found.")

        start, end = self.channel_range(channel)
        return original[start:end]

    def update(
        self, original: np.ndarray, channel: str, data: np.ndarray
    ) -> np.ndarray:
        """Inserts data into the original array at the position of the given channel.

        This function inserts new data at the index range previously associated with
        the provided channel. If the new data is of the same shape as the existing
        channel data shape, we simply replace the data at the channel range indices.
        Otherwise, We split the original array around the input channel range, then
        concatenate the before and after splits with the data to be inserted. This
        accommodates 'data' being of a different size than the current channel size.

        For example, if original has the shape (20, 3), channel start index is 10,
        channel end index is 13, and the data has the shape (5, 3). We would concatenate
        as such: (original[0:10], data, original[13:]). This will result in an array of
        the shape (22, 3), i.e., we removed 3 rows and added new 5 rows.

        Args:
            original (np.ndarray): The original array.
            channel (str): The name of the input channel.
            data (np.ndarray): The new data to insert.

        Returns:
            np.ndarray: The resulting array after insertion. Can return a new copy or a
                view, depending on whether the inserted data is of the same size as the
                existing channel.

        Raises:
            ValueError: If the channel is not found.
        """
        if channel not in self.channel_sizes:
            raise ValueError(f"Channel '{channel}' not found.")

        start, end = self.channel_range(channel)

        if self.channel_sizes[channel] == data.shape[0]:
            # returns a view not a copy
            original[start:end] = data
        else:
            # returns a copy not a view
            original = np.concatenate([original[:start], data, original[end:]], axis=0)

        return original

    def __repr__(self) -> str:
        """Returns a string representation of the current channel mapping.

        Returns:
            str: String representation of the channel mappings.
        """
        ranges = {ch: self.channel_range(ch) for ch in self.channel_sizes}
        return f"ChannelMapper({ranges})"


class EvidenceSlopeTracker:
    """Tracks the slopes of evidence streams over a sliding window.

    This class maintains a set of hypotheses, each represented by a time series of
    evidence values. It allows updating these values, computing average slopes, adding
    and removing hypotheses, and selecting the top-k hypotheses based on their slopes.

    Attributes:
        window_size (int): The number of past values to consider for each hypothesis.
        min_age (int): The minimum age required for a hypothesis to be considered valid.
        data (np.ndarray): A 2D NumPy array storing evidence values for each hypothesis.
        age (np.ndarray): A 1D NumPy array tracking the number of valid values for each
            hypothesis.
    """

    def __init__(self, window_size: int = 3, min_age: int = 5) -> None:
        """Initializes the Tracker with a given window size and minimum age.

        Args:
            window_size (int, optional): The size of the sliding window. Defaults to 3.
            min_age (int, optional): The minimum age required for a hypothesis to be
            considered valid. Defaults to 2.
        """
        self.window_size: int = window_size
        self.min_age: int = min_age
        self.data: np.ndarray = np.full(
            (0, window_size), np.nan
        )  # Initialize with NaNs
        self.age: np.ndarray = np.zeros(
            0, dtype=int
        )  # Tracks the number of valid values

    @property
    def total_size(self) -> int:
        """Returns the total number of tracked hypotheses.

        Returns:
            int: The number of hypotheses currently stored.
        """
        return self.data.shape[0]

    @property
    def valid_indices_mask(self) -> np.ndarray:
        """Returns the indices of hypotheses that meet the minimum age requirement.

        Returns:
            np.ndarray: A boolean array indicating valid hypotheses.
        """
        return self.age >= self.min_age

    @property
    def must_keep_mask(self) -> np.ndarray:
        """Returns the indices of hypotheses that must be kept.

        Returns:
            np.ndarray: A boolean array indicating hypotheses that should not be
            removed.
        """
        return self.age < self.min_age

    def valid_indices_mask_range(
        self, start: int | None, end: int | None
    ) -> np.ndarray:
        range_mask = np.zeros_like(self.valid_indices_mask, dtype=bool)

        if start is None and end is None:
            range_mask[:] = True
        elif start is None:
            range_mask[:end] = True
        elif end is None:
            range_mask[start:] = True
        else:
            range_mask[start:end] = True

        return self.valid_indices_mask & range_mask

    def update(self, values: List[float] | np.ndarray) -> None:
        """Updates all hypotheses with a list of new evidence values.

        Args:
            values (List[float] | np.ndarray): A list or NumPy array of new
            evidence values.

        Raises:
            ValueError: If the number of provided values does not match the expected
            size.
        """
        values = np.array(values, dtype=float)  # Convert input to NumPy array

        if values.shape[0] != self.total_size:
            raise ValueError(
                f"Expected {self.total_size} values, but got {len(values)}"
            )

        # Shift all rows to the left
        self.data[:, :-1] = self.data[:, 1:]

        # Append the new values, keeping NaNs where no new value is provided
        self.data[:, -1] = values

        # Increment age only for valid (non-NaN) updates
        self.age[~np.isnan(values)] += 1

    def _get_slopes(self) -> np.ndarray:
        """Computes the average slope dynamically for each hypothesis.

        Returns:
            np.ndarray: An array containing the average slope for each hypothesis.
        """
        diffs = np.diff(
            self.data, axis=1
        )  # Compute differences between consecutive values
        valid_steps = np.sum(~np.isnan(diffs), axis=1)  # Count non-NaN differences
        valid_steps = np.where(
            valid_steps == 0, np.nan, valid_steps
        )  # Avoid division by zero
        return (
            np.nansum(diffs, axis=1) / valid_steps
        )  # Compute average slope per hypothesis

    def add_hyp(self, num_new_hyp: int) -> None:
        """Adds new hypotheses initialized with NaNs and age 0.

        Args:
            num_new_hyp (int): The number of new hypotheses to add.
        """
        new_data = np.full((num_new_hyp, self.window_size), np.nan)
        new_age = np.zeros(num_new_hyp, dtype=int)

        # Append new streams to the existing ones
        self.data = np.vstack((self.data, new_data))
        self.age = np.concatenate((self.age, new_age))

    def remove_hyp(self, stream_ids: List[int]) -> None:
        """Removes specified hypotheses based on their indices.

        Args:
            stream_ids (List[int]): A list of indices representing hypotheses
            to be removed.
        """
        mask = np.ones(self.data.shape[0], dtype=bool)
        mask[stream_ids] = False  # Mark indices for removal

        # Keep only the hypotheses that are not removed
        self.data = self.data[mask]
        self.age = self.age[mask]

    def get_keep_and_remove_ids(
        self, desired_total: int, start: int | None = None, end: int | None = None
    ) -> tuple[list[int], list[int]]:
        """Determines which hypotheses to keep and which to.

        Args:
            desired_total (int): The target number of hypotheses to keep.
            start (int | None, optional): Start index for candidate removals.
            end (int | None, optional): End index for candidate removals.

        Returns:
            tuple[list[int], list[int]]: (to_keep, to_remove)
        """
        valid_mask = self.valid_indices_mask_range(start, end)
        remove_requested = (end - start) - desired_total

        # retrieve slopes
        slopes = self._get_slopes()
        valid_slopes = slopes[valid_mask]
        valid_ids = np.arange(len(slopes))[valid_mask]

        sorted_indices = np.argsort(valid_slopes)

        to_remove = valid_ids[sorted_indices[:remove_requested]]
        to_keep = np.array(
            [i for i in np.arange(self.total_size) if i not in to_remove], dtype=int
        )
        return to_keep, to_remove
