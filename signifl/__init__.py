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


def _bounds(val_encoded, multiplier):
    """Extract tightest bounds on the original, non-encoded values"""
    val_type = _np_float_type(val_encoded)
    err_bnd = error_bound(val_encoded)
    pair = _np.dtype([('lower', val_type), ('upper', val_type)])
    bounds = _np.empty(val_encoded.shape, pair)
    bounds['lower'] = val_encoded - multiplier * err_bnd / 2
    bounds['upper'] = val_encoded + multiplier * err_bnd / 2
    return bounds


def encode(val, err):
    """Encode floats into convention format"""
    val_type = _np_float_type(val)
    if _np_float_type(err) != val_type:
        err = _np.array(err, val_type)
    val_encoded = _np.copy(val)
    err_bnd = _np.copy(err)
    finfo = _np.finfo(val_type)
    err_bnd = _np.abs(err)  # err is assumed to be one-sided
    if _np.isscalar(err_bnd):
        err_bnd = err_bnd * _np.ones(val_encoded.shape)
    val_isneg = _np.signbit(val)
    val_encoded[val_isneg] *= -1  # remove sign
    # we will not change NaN or infinity
    val_isnum = ~(_np.isnan(val) | _np.isinf(val))
    # avoid errs that are too small or NaN
    err_bnd = _np.fmax(err_bnd[val_isnum],
                       2 * _np.fmax(finfo.tiny, val_encoded[val_isnum] * finfo.eps))
    err_bnd = 2 ** _np.floor(_np.log2(err_bnd))
    val_encoded[val_isnum] = (2 * _np.floor(val_encoded[val_isnum] / err_bnd) + 1) * err_bnd / 2
    val_encoded[val_isneg] *= -1  # restore sign
    return val_encoded


def error_bound(val_encoded):
    """Extract the error bound of encoded values"""
    val_type = _np_float_type(val_encoded)
    finfo = _np.finfo(val_type)
    val_isnum = ~(_np.isnan(val_encoded) | _np.isinf(val_encoded))
    err_bnd = _np.nan * _np.ones(val_encoded.shape)  # return NaN for NaN and infinity
    negative, exponent, significand = _decompose(val_encoded[val_isnum])
    if any(exponent == finfo.minexp - 1):  # subnormals (including zero)
        raise ValueError("Zero or subnormal value detected in input; "
                         "these cannot be generated under our convention.")
    err_exponent = exponent.astype(val_type)
    b = significand != 0
    err_exponent[b] -= _np.array(finfo.nmant, val_type)
    err_exponent[b] += _np.log2(
                          (significand[b] & -significand[b]).astype(val_type))
            # for v & -v trick, see https://stackoverflow.com/questions/18806481
    err_bnd[val_isnum] = 2 * 2 ** err_exponent
    return err_bnd


def inner_bounds(val_encoded):
    """Extract tightest bounds on the original, non-encoded values"""
    return _bounds(val_encoded, 1)


def outer_bounds(val_encoded):
    """Extract tightest bounds on the original, non-encoded error intervals"""
    return _bounds(val_encoded, 5)


def round_decimal(val_encoded):
    """Decimal round values so that they can be re-encoded without loss"""
    val_type = _np_float_type(val_encoded)
    pair = _np.dtype([('value', val_type), ('error', val_type)])
    err_bnd = error_bound(val_encoded)
    val_err = _np.empty(val_encoded.shape, pair)
    val_err['error'] = err_bnd
    dec_err_bnd = 10 ** _np.floor(_np.log10(err_bnd))
    val_err['value'] = _np.round(val_encoded / dec_err_bnd) * dec_err_bnd
    return val_err
