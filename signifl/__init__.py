"""
    Main code of signifl.
    Copyright (C) 2018 Erik Quaeghebeur. All rights reserved.

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""


import numpy as _np


def _np_float_type(val):
    """Check that argument has a NumPy binary float type; return that type"""
    if not val.dtype.kind == 'f':
        raise TypeError("Argument must be a NumPy binary float type. "
                        "Your argument has type ‘{}’.".format(val.dtype.name))
    return val.dtype


def _decompose(val):
    """Decompose floats into negative, exponent, and significand

    See https://stackoverflow.com/questions/46093123 for context.

    """
    val_type = _np_float_type(val)
    finfo = _np.finfo(val_type)
    negative = _np.signbit(val)
    int_type = _np.dtype('i' + str(val_type.itemsize))
    val_int = _np.abs(val).view(int_type)  # discard sign (MSB now 0),
                                           # view bit string as int
    exponent = (val_int >> finfo.nmant)  # drop significand
    exponent -= finfo.maxexp - 1  # correct exponent offset
    significand = val_int & _np.array(2 ** finfo.nmant - 1, int_type)
                        # second factor provides mask to extract significand
    return (negative, exponent, significand)


def _bounds(val_enc, multiplier):
    """Extract tightest bounds on the original, non-encoded values"""
    val_type = _np_float_type(val_enc)
    unc_bnd = uncertainty_bound(val_enc)
    pair = _np.dtype([('lower', val_type), ('upper', val_type)])
    bounds = _np.empty(val_enc.shape, pair)
    bounds['lower'] = val_enc - multiplier * unc_bnd / 2
    bounds['upper'] = val_enc + multiplier * unc_bnd / 2
    return bounds


def encode(val, unc):
    """Encode floats into convention format

    Args:
        val:
            NumPy array of (floating-point) values
        unc:
            NumPy array with uncertainty (floating-point) values

    Returns:
        Numpy array of values encoded according to the convention

    Examples:
        Usage is illustrated below:

        >>> import numpy as np
        >>> import signifl as sf
        >>> values = np.float32([10/3, -1234, -np.inf, -np.nan])
        >>> values
        array([3.33333325, -1234., -inf, nan], dtype=float32)
        >>> uncertainties = np.float64([0.1, 3, np.nan, np.inf])
        >>> uncertainties
        array([0.1, 3.,  nan, inf])
        >>> sf.encode(values, uncertainties)
        array([3.34375, -1235., -inf, nan], dtype=float32)

        NumPy broadcasting can be used (continued from example above):

        >>> uncertainties = np.float64(0.01)
        >>> sf.encode(values, uncertainties)
        array([3.33203125, -1234.00390625, -inf, nan], dtype=float32)

        But this has not been broadly tested yet, so beware!

    """
    val_type = _np_float_type(val)
    if _np_float_type(unc) != val_type:
        unc = _np.array(unc, val_type)
    val_enc = _np.copy(val)
    unc_bnd = _np.copy(unc)
    finfo = _np.finfo(val_type)
    unc_bnd = _np.abs(unc)  # unc is assumed to be one-sided
    if _np.isscalar(unc_bnd):
        unc_bnd = unc_bnd * _np.ones(val_enc.shape)
    val_isneg = _np.signbit(val)
    val_enc[val_isneg] *= -1  # remove sign
    # we will not change NaN or infinity
    val_isnum = ~(_np.isnan(val) | _np.isinf(val))
    # avoid uncs that are too small or NaN; then use maximal precision
    unc_bnd = _np.fmax(unc_bnd[val_isnum],
                       2 * _np.fmax(finfo.tiny, val_enc[val_isnum] * finfo.eps))
    unc_bnd = 2 ** _np.floor(_np.log2(unc_bnd))
    val_enc[val_isnum] = ((2 * _np.floor(val_enc[val_isnum] / unc_bnd) + 1)
                          * unc_bnd / 2)
    val_enc[val_isneg] *= -1  # restore sign
    return val_enc


def uncertainty_bound(val_enc):
    """Extract the uncertainty bound of encoded values

    Args:
        val_enc:
            Numpy array of (floating-point) values encoded
            according to the convention

    Returns:
        NumPy array with lowers bound on the uncertainties; these are the
        largest powers of two smaller than the original uncertainties

    Raises:
        ValueError:
            when subnormal values are present in the argument array
            (subnormal values—including zero—are never generated according to
            the convention)

    Examples:
        Usage is illustrated below:

        >>> import numpy as np
        >>> import signifl as sf
        >>> values = np.float32([10/3, -1234, -np.inf, -np.nan])
        >>> uncertainties = np.float64([0.1, 3, np.nan, np.inf])
        >>> values_encoded = sf.encode(values, uncertainties)
        >>> sf.uncertainty_bound(values_encoded)
        array([0.0625, 2., nan, nan])

    """
    val_type = _np_float_type(val_enc)
    finfo = _np.finfo(val_type)
    val_isnum = ~(_np.isnan(val_enc) | _np.isinf(val_enc))
    unc_bnd = _np.nan * _np.ones(val_enc.shape)  # return NaN for NaN and inf
    negative, exponent, significand = _decompose(val_enc[val_isnum])
    if any(exponent == finfo.minexp - 1):  # subnormals (including zero)
        raise ValueError("Zero or subnormal value detected in input; "
                         "these cannot be generated under our convention.")
    unc_exponent = exponent.astype(val_type)
    b = significand != 0
    unc_exponent[b] -= _np.array(finfo.nmant, val_type)
    unc_exponent[b] += _np.log2(
                          (significand[b] & -significand[b]).astype(val_type))
            # for v & -v trick, see https://stackoverflow.com/questions/18806481
    unc_bnd[val_isnum] = 2 * 2 ** unc_exponent
    return unc_bnd


def inner_bounds(val_enc):
    """Extract tightest bounds on the non-encoded values

    Args:
        val_enc:
            Numpy array of (floating-point) values encoded
            according to the convention

    Returns:
        Numpy array with structured pairs of tightest `'lower'` and `'upper'`
        bounds on the non-encoded values
        (namely, the encoded value ± half the uncertainty bound)

    Examples:
        Usage is illustrated below:

        >>> import numpy as np
        >>> import signifl as sf
        >>> values = np.float32([10/3, -1234, -np.inf, -np.nan])
        >>> uncertainties = np.float64([0.1, 3, np.nan, np.inf])
        >>> values_encoded = sf.encode(values, uncertainties)
        >>> sf.inner_bounds(values_encoded)
        array([(3.3125, 3.375), (-1236.0, -1234.0), (nan, nan), (nan, nan)],
              dtype=[('lower', '<f4'), ('upper', '<f4')])

    """
    return _bounds(val_enc, 1)


def outer_bounds(val_enc):
    """Extract tightest bounds on the non-encoded uncertainty intervals

    Args:
        val_enc:
            Numpy array of (floating-point) values encoded
            according to the convention

    Returns:
        Numpy array with structured pairs of tightest `'lower'` and `'upper'`
        bounds on the non-encoded uncertainty intervals
        (namely, the encoded value ± five times half the uncertainty bound)

    Examples:
        Usage is illustrated below:

        >>> import numpy as np
        >>> import signifl as sf
        >>> values = np.float32([10/3, -1234, -np.inf, -np.nan])
        >>> uncertainties = np.float64([0.1, 3, np.nan, np.inf])
        >>> values_encoded = sf.encode(values, uncertainties)
        >>> sf.outer_bounds(values_encoded)
        array([(3.1875, 3.5), (-1240.0, -1230.0), (nan, nan), (nan, nan)],
              dtype=[('lower', '<f4'), ('upper', '<f4')])

    """
    return _bounds(val_enc, 5)


def greater_than(val_enc_lhs, val_enc_rhs):
    """Determine which values are greater in a significant way

    Args:
        val_enc_lhs:
            Numpy array of (floating-point) values encoded
            according to the convention
        val_enc_rhs:
            Numpy array of (floating-point) values encoded
            according to the convention

    Returns:
        Numpy boolean array

    Examples:
        Usage is illustrated below:

        >>> import numpy as np
        >>> import signifl as sf
        >>> values = np.float32([10/3, -1234, 10/3, -1234])
        >>> uncertainties = np.float32([0.1, 3, 0.1, 3])
        >>> values_perturbed = np.copy(values)
        >>> values_perturbed[:2] += uncertainties[:2] / 2
        >>> values_perturbed[2:] += 10 * uncertainties[2:]
        >>> values_encoded = sf.encode(values, uncertainties)
        >>> values_encoded
        array([3.34375, -1235., 3.34375, -1235.], dtype=float32)
        >>> values_perturbed_encoded = sf.encode(values_perturbed, uncertainties)
        >>> values_perturbed_encoded
        array([3.40625, -1233., 4.34375, -1205.], dtype=float32)
        >>> sf.greater_than(values_perturbed_encoded, values_encoded)
        array([False, False,  True,  True], dtype=bool)

    """
    return (outer_bounds(val_enc_lhs)['lower']
            > outer_bounds(val_enc_rhs)['upper'])


def less_than(val_enc_lhs, val_enc_rhs):
    """Determine which values are less in a significant way

    Args:
        val_enc_lhs:
            Numpy array of (floating-point) values encoded
            according to the convention
        val_enc_rhs:
            Numpy array of (floating-point) values encoded
            according to the convention

    Returns:
        Numpy boolean array

    Examples:
        Usage is illustrated below:

        >>> import numpy as np
        >>> import signifl as sf
        >>> values = np.float32([10/3, -1234, 10/3, -1234])
        >>> uncertainties = np.float32([0.1, 3, 0.1, 3])
        >>> values_perturbed = np.copy(values)
        >>> values_perturbed[:2] += uncertainties[:2] / 2
        >>> values_perturbed[2:] += 10 * uncertainties[2:]
        >>> values_encoded = sf.encode(values, uncertainties)
        >>> values_encoded
        array([3.34375, -1235., 3.34375, -1235.], dtype=float32)
        >>> values_perturbed_encoded = sf.encode(values_perturbed, uncertainties)
        >>> values_perturbed_encoded
        array([3.40625, -1233., 4.34375, -1205.], dtype=float32)
        >>> sf.less_than(values_encoded, values_perturbed_encoded)
        array([False, False, False, False], dtype=bool)

    """
    return greater_than(val_enc_lhs, val_enc_rhs)


def incomparable(val_enc_lhs, val_enc_rhs):
    """Determine which values are incomparable in a significant way

    Args:
        val_enc_lhs:
            Numpy array of (floating-point) values encoded
            according to the convention
        val_enc_rhs:
            Numpy array of (floating-point) values encoded
            according to the convention

    Returns:
        Numpy boolean array

    Examples:
        Usage is illustrated below:

        >>> import numpy as np
        >>> import signifl as sf
        >>> values = np.float32([10/3, -1234, 10/3, -1234])
        >>> uncertainties = np.float32([0.1, 3, 0.1, 3])
        >>> values_perturbed = np.copy(values)
        >>> values_perturbed[:2] += uncertainties[:2] / 2
        >>> values_perturbed[2:] += 10 * uncertainties[2:]
        >>> values_encoded = sf.encode(values, uncertainties)
        >>> values_encoded
        array([3.34375, -1235., 3.34375, -1235.], dtype=float32)
        >>> values_perturbed_encoded = sf.encode(values_perturbed, uncertainties)
        >>> values_perturbed_encoded
        array([3.40625, -1233., 4.34375, -1205.], dtype=float32)
        >>> sf.incomparable(values_perturbed_encoded, values_encoded)
        array([True,  True, False, False], dtype=bool)

    """
    return ~(greater_than(val_enc_rhs, val_enc_lhs) |
             less_than(val_enc_lhs, val_enc_rhs))


def round_decimal(val_enc):
    """Decimal round values so that they can be re-encoded without loss

    Args:
        val_enc:
            Numpy array of (floating-point) values encoded
            according to the convention

    Returns:
        Numpy array with structured pairs of 'value' and 'uncertainty'

    Examples:
        Usage is illustrated below:

        >>> import numpy as np
        >>> import signifl as sf
        >>> values = np.float32([10/3, -1234, -np.inf, -np.nan])
        >>> uncertainties = np.float64([0.1, 3, np.nan, np.inf])
        >>> values_encoded = sf.encode(values, uncertainties)
        >>> decvalues_uncbound = sf.round_decimal(values_encoded)
        >>> decvalues_uncbound
        array([(3.3399999141693115, 0.0625), (-1235.0, 2.0), (nan, nan), (nan, nan)],
              dtype=[('value', '<f4'), ('uncertainty', '<f4')])
        >>> values_reencoded = sf.encode(decvalues_uncbound['value'], decvalues_uncbound['uncertainty'])
        >>> np.all(values_reencoded[~np.isnan(values)] == values_encoded[~np.isnan(values)])
        True

    """
    val_type = _np_float_type(val_enc)
    pair = _np.dtype([('value', val_type), ('uncertainty', val_type)])
    unc_bnd = uncertainty_bound(val_enc)
    val_unc = _np.empty(val_enc.shape, pair)
    val_unc['uncertainty'] = unc_bnd
    dec_unc_bnd = 10 ** _np.floor(_np.log10(unc_bnd))
    # we will not change NaN or infinity
    val_isnum = ~(_np.isnan(val_enc) | _np.isinf(val_enc))
    val_unc['value'] = _np.where(val_isnum,
                                 _np.round(val_enc / dec_unc_bnd) * dec_unc_bnd,
                                 val_enc)
    return val_unc
