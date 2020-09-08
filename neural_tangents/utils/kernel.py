# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The `Kernel` class containing NTK and NNGP `np.ndarray`s as fields."""


import operator as op
from typing import Dict, Tuple, Optional, Callable, Any

import jax.numpy as np
from neural_tangents.utils import dataclasses
from neural_tangents.utils import utils


@dataclasses.dataclass
class Kernel:
  """Dataclass containing information about the NTK and NNGP of a model.

  Attributes:
    nngp:
      covariance between the first and second batches (NNGP). A `np.ndarray` of
      shape
      `(batch_size_1, batch_size_2, height, [height,], width, [width,], ..))`,
      where exact shape depends on `diagonal_spatial`.
    ntk:
      the neural tangent kernel (NTK). `np.ndarray` of same shape as `nngp`.
    cov1:
      covariance of the first batch of inputs. A `np.ndarray` with shape
      `(batch_size_1, [batch_size_1,] height, [height,], width, [width,], ..)`
      where exact shape depends on `diagonal_batch` and `diagonal_spatial`.
    cov2:
      optional covariance of the second batch of inputs. A `np.ndarray` with
      shape
      `(batch_size_2, [batch_size_2,] height, [height,], width, [width,], ...)`
      where the exact shape depends on `diagonal_batch` and `diagonal_spatial`.
    x1_is_x2:
      a boolean specifying whether `x1` and `x2` are the same.
    is_gaussian:
      a boolean, specifying whether the output features or channels of the layer
      / NN function (returning this `Kernel` as the `kernel_fn`) are i.i.d.
      Gaussian with covariance `nngp`, conditioned on fixed inputs to the layer
      and i.i.d. Gaussian weights and biases of the layer. For example, passing
      an input through a CNN layer with i.i.d. Gaussian weights and biases
      produces i.i.d. Gaussian random variables along the channel dimension,
      while passing an input through a nonlinearity does not.
    is_reversed:
      a boolean specifying whether the covariance matrices `nngp`, `cov1`,
      `cov2`, and `ntk` have the ordering of spatial dimensions reversed.
      Ignored unless `diagonal_spatial` is `False`. Used internally to avoid
      self-cancelling transpositions in a sequence of CNN layers that flip the
      order of kernel spatial dimensions.
    is_input:
      a boolean specifying whether the current layer is the input layer and it
      is used to avoid applying dropout to the input layer.
    diagonal_batch:
      a boolean specifying whether `cov1` and `cov2` store only the diagonal of
      the sample-sample covariance (`diagonal_batch == True`,
      `cov1.shape == (batch_size_1, ...)`), or the full covariance
      (`diagonal_batch == False`,
      `cov1.shape == (batch_size_1, batch_size_1, ...)`). Defaults to `True` as
      no current layers require the full covariance.
    diagonal_spatial:
      a boolean specifying whether all (`cov1`, `ntk`, etc.) covariance matrices
      store only the diagonals of the location-location covariances
      (`diagonal_spatial == True`,
      `nngp.shape == (batch_size_1, batch_size_2, height, width, depth, ...)`),
      or the full covariance (`diagonal_spatial == False`,
     `nngp.shape == (batch_size_1, batch_size_2, height, height, width, width,
     depth, depth, ...)`). Defaults to `False`, but is set to `True` if the
     output top-layer covariance depends only on the diagonals (e.g. when a CNN
     network has no pooling layers and `Flatten` on top).
    shape1:
      a tuple specifying the shape of the random variable in the first batch of
      inputs. These have covariance `cov1` and covariance with the second batch
      of inputs given by `nngp`.
    shape2:
      a tuple specifying the shape of the random variable in the second batch of
      inputs. These have covariance `cov2` and covariance with the first batch
      of inputs given by `nngp`.
    batch_axis:
      the batch axis of the activations.
    channel_axis:
      channel axis of the activations (taken to infinity).
    mask1:
      an optional boolean `np.ndarray` with a shape broadcastable to `shape1`
      (and the same number of dimensions). `True` stands for the input being
      masked at that position, while `False` means the input is visible. For
      example, if `shape1 == (5, 32, 32, 3)` (a batch of 5 `NHWC` CIFAR10
      images), a `mask1` of shape `(5, 1, 32, 1)` means different images can
      have different blocked columns (`H` and `C` dimensions are always either
      both blocked or unblocked). `None` means no masking.
    mask2:
      same as `mask1`, but for the second batch of inputs.
  """

  nngp: np.ndarray
  ntk: Optional[np.ndarray]

  cov1: np.ndarray
  cov2: Optional[np.ndarray]
  x1_is_x2: np.ndarray

  is_gaussian: bool = dataclasses.field(pytree_node=False)
  is_reversed: bool = dataclasses.field(pytree_node=False)
  is_input: bool = dataclasses.field(pytree_node=False)

  diagonal_batch: bool = dataclasses.field(pytree_node=False)
  diagonal_spatial: bool = dataclasses.field(pytree_node=False)

  shape1: Tuple[int, ...] = dataclasses.field(pytree_node=False)
  shape2: Tuple[int, ...] = dataclasses.field(pytree_node=False)

  batch_axis: int = dataclasses.field(pytree_node=False)
  channel_axis: int = dataclasses.field(pytree_node=False)

  mask1: Optional[np.ndarray] = None
  mask2: Optional[np.ndarray] = None

  replace = ...  # type: Callable[..., 'Kernel']
  asdict = ...  # type: Callable[[], Dict[str, Any]]
  astuple = ...  # type: Callable[[], Tuple[Any, ...]]

  def slice(self, n1_slice: slice, n2_slice: slice) -> 'Kernel':
    cov1 = self.cov1[n1_slice]
    cov2 = self.cov1[n2_slice] if self.cov2 is None else self.cov2[n2_slice]
    ntk = self.ntk

    mask1 = None if self.mask1 is None else self.mask1[n1_slice]
    mask2 = None if self.mask2 is None else self.mask2[n2_slice]

    return self.replace(
        cov1=cov1,
        nngp=self.nngp[n1_slice, n2_slice],
        cov2=cov2,
        ntk=ntk if ntk is None or ntk.ndim == 0 else ntk[n1_slice, n2_slice],
        shape1=(cov1.shape[0],) + self.shape1[1:],
        shape2=(cov2.shape[0],) + self.shape2[1:],
        mask1=mask1,
        mask2=mask2)

  def reverse(self) -> 'Kernel':
    """Reverse the order of spatial axes in the covariance matrices.

    Returns:
      A `Kernel` object with spatial axes order flipped in
      all covariance matrices. For example, if `kernel.nngp` has shape
      `(batch_size_1, batch_size_2, H, H, W, W, D, D, ...)`, then
      `reverse(kernels).nngp` has shape
      `(batch_size_1, batch_size_2, ..., D, D, W, W, H, H)`.
    """
    batch_ndim = 1 if self.diagonal_batch else 2
    cov1 = utils.reverse_zipped(self.cov1, batch_ndim)
    cov2 = utils.reverse_zipped(self.cov2, batch_ndim)
    nngp = utils.reverse_zipped(self.nngp, 2)
    ntk = utils.reverse_zipped(self.ntk, 2)

    return self.replace(cov1=cov1,
                        nngp=nngp,
                        cov2=cov2,
                        ntk=ntk,
                        is_reversed=not self.is_reversed)

  def transpose(self, axes: Tuple[int, ...] = None) -> 'Kernel':
    """Permute spatial dimensions of the `Kernel` according to `axes`.

    Follows
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.transpose.html

    Note that `axes` apply only to spatial axes, batch axes are ignored and
    remain leading in all covariance arrays, and channel axes are not present
    in a `Kernel` object. If the covariance array is of shape
    `(batch_size, X, X, Y, Y)`, and `axes == (0, 1)`, resulting array is of
    shape `(batch_size, Y, Y, X, X)`.
    """
    if axes is None:
      axes = tuple(range(len(self.shape1) - 2))

    def permute(mat: Optional[np.ndarray],
                batch_ndim: int) -> Optional[np.ndarray]:
      if mat is not None:
        _axes = tuple(batch_ndim + a for a in axes)
        if not self.diagonal_spatial:
          _axes = tuple(j for a in _axes
                        for j in (2 * a - batch_ndim,
                                  2 * a - batch_ndim + 1))
        _axes = tuple(range(batch_ndim)) + _axes
        return np.transpose(mat, _axes)
      return mat

    cov1 = permute(self.cov1, 1 if self.diagonal_batch else 2)
    cov2 = permute(self.cov2, 1 if self.diagonal_batch else 2)
    nngp = permute(self.nngp, 2)
    ntk = permute(self.ntk, 2)
    return self.replace(cov1=cov1, nngp=nngp, cov2=cov2, ntk=ntk)

  def mask(self,
           mask1: Optional[np.ndarray],
           mask2: Optional[np.ndarray]) -> 'Kernel':
    """Mask all covariance matrices according to `mask1`, `mask2`."""
    mask11, mask12, mask22 = self._get_mask_prods(mask1, mask2)

    cov1 = utils.mask(self.cov1, mask11)
    cov2 = utils.mask(self.cov2, mask22)
    nngp = utils.mask(self.nngp, mask12)
    ntk = utils.mask(self.ntk, mask12)

    return self.replace(cov1=cov1,
                        nngp=nngp,
                        cov2=cov2,
                        ntk=ntk,
                        mask1=mask1,
                        mask2=mask2)

  def _get_mask_prods(
      self,
      mask1: Optional[np.ndarray],
      mask2: Optional[np.ndarray]
  ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Gets outer products of `mask1, mask1`, `mask1, mask2`, `mask2, mask2`."""
    def get_mask_prod(m1, m2, batch_ndim):
      if m1 is None and m2 is None:
        return None

      def reshape(m):
        if m is not None:
          if m.shape[self.channel_axis] != 1:
            raise NotImplementedError(
                f'Different channel-wise masks are not supported for '
                f'infinite-width layers now (got `mask.shape == {m.shape}). '
                f'Please describe your use case at '
                f'https://github.com/google/neural-tangents/issues/new')

          m = np.squeeze(np.moveaxis(m, (self.batch_axis, self.channel_axis),
                                     (0, -1)), -1)
          if self.is_reversed:
            m = np.moveaxis(m, range(1, m.ndim), range(m.ndim - 1, 0, -1))
        return m

      m1, m2 = reshape(m1), reshape(m2)

      start_axis = 2 - batch_ndim
      end_axis = 1 if self.diagonal_spatial else m1.ndim

      mask = utils.outer_prod(m1, m2, start_axis, end_axis, op.or_)
      return mask

    mask11 = get_mask_prod(mask1, mask1, 1 if self.diagonal_batch else 2)
    mask22 = (get_mask_prod(mask2, mask2, 1 if self.diagonal_batch else 2)
              if mask2 is not None else mask11)
    mask12 = get_mask_prod(mask1, mask2, 2)
    return mask11, mask12, mask22

  def __mul__(self, other: float) -> 'Kernel':
    var = other**2
    return self.replace(cov1=var * self.cov1,
                        nngp=var * self.nngp,
                        cov2=None if self.cov2 is None else var * self.cov2,
                        ntk=None if self.ntk is None else var * self.ntk)

  __rmul__ = __mul__

  def __add__(self, other: float) -> 'Kernel':
    var = other**2
    return self.replace(cov1=var + self.cov1,
                        nngp=var + self.nngp,
                        cov2=None if self.cov2 is None else var + self.cov2)

  __sub__ = __add__

  def __truediv__(self, other: float) -> 'Kernel':
    return self.__mul__(1. / other)

  def __neg__(self) -> 'Kernel':
    return self

  __pos__ = __neg__
