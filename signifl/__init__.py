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


import numpy as np


def _numpy_float_type(val):
    if not val.dtype.kind == 'f':
        raise TypeError("Argument must be a NumPy binary float type. "
                        "Your argument has type ‘{}’.".format(val.dtype.name))
    return val.dtype


def _decompose(val):
    """Decompose floats into negative, exponent, and significand

    See https://stackoverflow.com/questions/46093123 for context.

    """
    val_type = _numpy_float_type(val)
    finfo = np.finfo(val_type)
    negative = np.signbit(val)
    int_type = np.dtype('i' + str(val_type.itemsize))
    val_int = np.abs(val).view(int_type)  # discard sign (MSB now 0),
                                          # view bit string as int
    exponent = (val_int >> finfo.nmant) + finfo.minexp - 1  # drop significand
    exponent -= finfo.maxexp - 1  # correct exponent offset
    significand = val_int & np.array(2 ** finfo.nmant - 1, int_type)
                            # second factor provides mask to extract significand
    return (negative, exponent, significand)


def _bounds(val_encoded, multiplier):
    """Extract tightest bounds on the original, non-encoded values"""
    val_type = _numpy_float_type(val_encoded)
    err_bnd = error_bound(val_encoded)
    pair = np.dtype([('lower', val_type), ('upper', val_type)])
    bounds = np.array(dtype=pair, shape=val_encoded.shape)
    bounds['lower'] = val_encoded - multiplier * err_bnd / 2
    bounds['upper'] = val_encoded + multiplier * err_bnd / 2
    return bounds


def encode(val, err):
    """Encode floats into convention format"""
    val_type = _numpy_float_type(val)
    if _numpy_float_type(err) != val_type:
        err = np.array(err, val_type)
    finfo = np.finfo(val_type)
    err = np.abs(err)  # err is assumed to be one-sided
    val_isneg = np.signbit(val)
    val[val_isneg] *= -1  # remove sign
    # we will not change NaN or infinity
    val_isnum = ~(np.isnan(val) | np.isinf(val))
    # avoid errs that are too small or NaN
    err = np.fmax(err[val_isnum],
                  2 * np.fmax(finfo.tiny, val[val_isnum] * finfo.eps))
    err_bnd = 2 ** np.floor(np.log2(err))
    val[val_isnum] = (2 * np.floor(val[val_isnum] / err_bnd) + 1) * err_bnd / 2
    val[val_isneg] *= -1  # restore sign
    return val


def error_bound(val_encoded):
    """Extract the error bound of encoded values"""
    val_type = _numpy_float_type(val_encoded)
    finfo = np.finfo(val_type)
    val_isnum = ~(np.isnan(val_encoded) | np.isinf(val_encoded))
    err_bnd = np.nan * np.ones(val_encoded.shape)  # return NaN for NaN and infinity
    negative, exponent, significand = _decompose(val_encoded[val_isnum])
    if any(exponent == finfo.minexp - 1):  # subnormals (including zero)
        raise ValueError("Zero or subnormal value detected in input; "
                         "these cannot be generated under our convention.")
    significand_isnonzero = significand != 0
    nonzero_significand = significand[significand_isnonzero]
    err_exponent = exponent
    err_exponent[significand_isnonzero] -= finfo.nmant
    err_exponent[significand_isnonzero] += np.log2(nonzero_significand
                                                   & -nonzero_significand)
            # for v & -v trick, see https://stackoverflow.com/questions/18806481
    err_bnd[val_isnum] = 2 ** err_exponent
    return err_bnd


def value_bounds(val_encoded):
    """Extract tightest bounds on the original, non-encoded values"""
    return _bounds(val_encoded, 1)


def error_bounds(val_encoded):
    """Extract tightest bounds on the original, non-encoded error intervals"""
    return _bounds(val_encoded, 5)


def round_decimal(val_encoded):
    """Decimal round values so that they can be re-encoded without loss"""
    err_bnd = error_bound(val_encoded)
    pair = np.dtype([('value', val_type), ('error', val_type)])
    val_err = np.array(dtype=pair, shape=val_encoded.shape)
    val_err['error'] = err_bnd
    dec_err_bnd = 10 ** np.floor(np.log10(err_bnd))
    val_err['value'] = np.round(val_encoded / dec_err_bnd) * dec_err_bnd
    return val_err
