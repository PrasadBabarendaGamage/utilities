import numpy as np

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def np_1_13_unique(ar, return_index=False, return_inverse=False,
           return_counts=False, axis=None):

    ar = np.asanyarray(ar)
    if axis is None:
        return _unique1d(ar, return_index, return_inverse, return_counts)
    if not (-ar.ndim <= axis < ar.ndim):
        raise ValueError('Invalid axis kwarg specified for unique')

    ar = np.swapaxes(ar, axis, 0)
    orig_shape, orig_dtype = ar.shape, ar.dtype
    # Must reshape to a contiguous 2D array for this to work...
    ar = ar.reshape(orig_shape[0], -1)
    ar = np.ascontiguousarray(ar)

    if ar.dtype.char in (np.typecodes['AllInteger'] +
                         np.typecodes['Datetime'] + 'S'):
        # Optimization: Creating a view of your data with a np.void data type of
        # size the number of bytes in a full row. Handles any type where items
        # have a unique binary representation, i.e. 0 is only 0, not +0 and -0.
        dtype = np.dtype((np.void, ar.dtype.itemsize * ar.shape[1]))
    else:
        dtype = [('f{i}'.format(i=i), ar.dtype) for i in range(ar.shape[1])]

    try:
        consolidated = ar.view(dtype)
    except TypeError:
        # There's no good way to do this for object arrays, etc...
        msg = 'The axis argument to unique is not supported for dtype {dt}'
        raise TypeError(msg.format(dt=ar.dtype))

    def reshape_uniq(uniq):
        uniq = uniq.view(orig_dtype)
        uniq = uniq.reshape(-1, *orig_shape[1:])
        uniq = np.swapaxes(uniq, 0, axis)
        return uniq

    output = _unique1d(consolidated, return_index,
                       return_inverse, return_counts)
    if not (return_index or return_inverse or return_counts):
        return reshape_uniq(output)
    else:
        uniq = reshape_uniq(output[0])
        return (uniq,) + output[1:]

def _unique1d(ar, return_index=False, return_inverse=False,
              return_counts=False):
    """
    Find the unique elements of an array, ignoring shape.
    """
    ar = np.asanyarray(ar).flatten()

    optional_indices = return_index or return_inverse
    optional_returns = optional_indices or return_counts

    if ar.size == 0:
        if not optional_returns:
            ret = ar
        else:
            ret = (ar,)
            if return_index:
                ret += (np.empty(0, np.intp),)
            if return_inverse:
                ret += (np.empty(0, np.intp),)
            if return_counts:
                ret += (np.empty(0, np.intp),)
        return ret

    if optional_indices:
        perm = ar.argsort(kind='mergesort' if return_index else 'quicksort')
        aux = ar[perm]
    else:
        ar.sort()
        aux = ar
    flag = np.concatenate(([True], aux[1:] != aux[:-1]))

    if not optional_returns:
        ret = aux[flag]
    else:
        ret = (aux[flag],)
        if return_index:
            ret += (perm[flag],)
        if return_inverse:
            iflag = np.cumsum(flag) - 1
            inv_idx = np.empty(ar.shape, dtype=np.intp)
            inv_idx[perm] = iflag
            ret += (inv_idx,)
        if return_counts:
            idx = np.concatenate(np.nonzero(flag) + ([ar.size],))
            ret += (np.diff(idx),)
    return ret

def np_1_13_in1d(ar1, ar2, assume_unique=False, invert=False):
    """
    Test whether each element of a 1-D array is also present in a second array.
    Returns a boolean array the same length as `ar1` that is True
    where an element of `ar1` is in `ar2` and False otherwise.
    We recommend using :func:`isin` instead of `in1d` for new code.
    Parameters
    ----------
    ar1 : (M,) array_like
        Input array.
    ar2 : array_like
        The values against which to test each value of `ar1`.
    assume_unique : bool, optional
        If True, the input arrays are both assumed to be unique, which
        can speed up the calculation.  Default is False.
    invert : bool, optional
        If True, the values in the returned array are inverted (that is,
        False where an element of `ar1` is in `ar2` and True otherwise).
        Default is False. ``np.in1d(a, b, invert=True)`` is equivalent
        to (but is faster than) ``np.invert(in1d(a, b))``.
        .. versionadded:: 1.8.0
    Returns
    -------
    in1d : (M,) ndarray, bool
        The values `ar1[in1d]` are in `ar2`.
    See Also
    --------
    isin                  : Version of this function that preserves the
                            shape of ar1.
    numpy.lib.arraysetops : Module with a number of other functions for
                            performing set operations on arrays.
    Notes
    -----
    `in1d` can be considered as an element-wise function version of the
    python keyword `in`, for 1-D sequences. ``in1d(a, b)`` is roughly
    equivalent to ``np.array([item in b for item in a])``.
    However, this idea fails if `ar2` is a set, or similar (non-sequence)
    container:  As ``ar2`` is converted to an array, in those cases
    ``asarray(ar2)`` is an object array rather than the expected array of
    contained values.
    .. versionadded:: 1.4.0
    Examples
    --------
    >>> test = np.array([0, 1, 2, 5, 0])
    >>> states = [0, 2]
    >>> mask = np.in1d(test, states)
    >>> mask
    array([ True, False,  True, False,  True], dtype=bool)
    >>> test[mask]
    array([0, 2, 0])
    >>> mask = np.in1d(test, states, invert=True)
    >>> mask
    array([False,  True, False,  True, False], dtype=bool)
    >>> test[mask]
    array([1, 5])
    """
    # Ravel both arrays, behavior for the first array could be different
    ar1 = np.asarray(ar1).ravel()
    ar2 = np.asarray(ar2).ravel()

    # This code is significantly faster when the condition is satisfied.
    if len(ar2) < 10 * len(ar1) ** 0.145:
        if invert:
            mask = np.ones(len(ar1), dtype=np.bool)
            for a in ar2:
                mask &= (ar1 != a)
        else:
            mask = np.zeros(len(ar1), dtype=np.bool)
            for a in ar2:
                mask |= (ar1 == a)
        return mask

    # Otherwise use sorting
    if not assume_unique:
        ar1, rev_idx = np.unique(ar1, return_inverse=True)
        ar2 = np.unique(ar2)

    ar = np.concatenate((ar1, ar2))
    # We need this to be a stable sort, so always use 'mergesort'
    # here. The values from the first array should always come before
    # the values from the second array.
    order = ar.argsort(kind='mergesort')
    sar = ar[order]
    if invert:
        bool_ar = (sar[1:] != sar[:-1])
    else:
        bool_ar = (sar[1:] == sar[:-1])
    flag = np.concatenate((bool_ar, [invert]))
    ret = np.empty(ar.shape, dtype=bool)
    ret[order] = flag

    if assume_unique:
        return ret[:len(ar1)]
    else:
        return ret[rev_idx]

def np_1_13_isin(element, test_elements, assume_unique=False, invert=False):
    """
    Calculates `element in test_elements`, broadcasting over `element` only.
    Returns a boolean array of the same shape as `element` that is True
    where an element of `element` is in `test_elements` and False otherwise.
    Parameters
    ----------
    element : array_like
        Input array.
    test_elements : array_like
        The values against which to test each value of `element`.
        This argument is flattened if it is an array or array_like.
        See notes for behavior with non-array-like parameters.
    assume_unique : bool, optional
        If True, the input arrays are both assumed to be unique, which
        can speed up the calculation.  Default is False.
    invert : bool, optional
        If True, the values in the returned array are inverted, as if
        calculating `element not in test_elements`. Default is False.
        ``np.isin(a, b, invert=True)`` is equivalent to (but faster
        than) ``np.invert(np.isin(a, b))``.
    Returns
    -------
    isin : ndarray, bool
        Has the same shape as `element`. The values `element[isin]`
        are in `test_elements`.
    See Also
    --------
    in1d                  : Flattened version of this function.
    numpy.lib.arraysetops : Module with a number of other functions for
                            performing set operations on arrays.
    Notes
    -----
    `isin` is an element-wise function version of the python keyword `in`.
    ``isin(a, b)`` is roughly equivalent to
    ``np.array([item in b for item in a])`` if `a` and `b` are 1-D sequences.
    `element` and `test_elements` are converted to arrays if they are not
    already. If `test_elements` is a set (or other non-sequence collection)
    it will be converted to an object array with one element, rather than an
    array of the values contained in `test_elements`. This is a consequence
    of the `array` constructor's way of handling non-sequence collections.
    Converting the set to a list usually gives the desired behavior.
    .. versionadded:: 1.13.0
    Examples
    --------
    >>> element = 2*np.arange(4).reshape((2, 2))
    >>> element
    array([[0, 2],
           [4, 6]])
    >>> test_elements = [1, 2, 4, 8]
    >>> mask = np.isin(element, test_elements)
    >>> mask
    array([[ False,  True],
           [ True,  False]], dtype=bool)
    >>> element[mask]
    array([2, 4])
    >>> mask = np.isin(element, test_elements, invert=True)
    >>> mask
    array([[ True, False],
           [ False, True]], dtype=bool)
    >>> element[mask]
    array([0, 6])
    Because of how `array` handles sets, the following does not
    work as expected:
    >>> test_set = {1, 2, 4, 8}
    >>> np.isin(element, test_set)
    array([[ False, False],
           [ False, False]], dtype=bool)
    Casting the set to a list gives the expected result:
    >>> np.isin(element, list(test_set))
    array([[ False,  True],
           [ True,  False]], dtype=bool)
    """
    element = np.asarray(element)
    return np_1_13_in1d(element, test_elements, assume_unique=assume_unique,
                invert=invert).reshape(element.shape)