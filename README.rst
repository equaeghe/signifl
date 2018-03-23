signifl is a Python package for working with IEEE-753 binary floating point
numbers that carry significance information by using a specific convention


Installation
============

Currently, you have to clone this repository and then locally install the
package:

    $ cd path/to/local/copy/of/repo
    $ pip install --user .


Usage
=====

The public functions in the main code file, ``signifl/__init__.py`` contain
docstrings that should get you started. Any modern Python shell will happily
tell you which functions there are after ``import signifl as sf`` by
tab-completing ``sf.``.


Background documentation
========================

*(This is an edited excerpt from a paper in preparation.)*

Consider ``x±ε``, a floating point number with its uncertainty. Assume ``x`` is
positive, otherwise apply negation first. Define ``δ`` by ``lb(δ) = ⌊lb(ε)⌋``,
[#log]_ so that ``δ`` provides tight power-of-two bounds on ``ε``; namely
``δ ≤ ε < 2·δ``. Then we store ``y = (2·⌊x/δ⌋+1)·δ/2`` instead of ``x``; it is
the odd multiple of ``δ/2`` nearest to ``x``, [#nearest]_ so ``|x-y| ≤ δ/2``.
Because ``y`` is an odd multiple of a power of two, the denominator of ``y``
seen as an irreducable fraction is ``2/δ``, so that ``δ`` can effectively be
determined from ``y``.

Let us illustrate our convention with an example. Suppose ``x = 0.65432`` and
``ε = 0.05``. Then ``⌊lb(ε)⌋ = -5`` and thus ``δ = 1/2⁵ = 1/32 = 0.03125``, so
that ``⌊x/δ⌋ = 20`` and ``y = 0.640625``. In binary, these are written
``x = 0.10100111100000011…`` and ``y = 0.101001``. In the binary expansion of
``y``, fractional digit number ``6 = -lb(δ/2)`` is ``1`` and all further digits
are ``0``. This is a general feature of our convention that makes it practically
easy to determine ``δ``.

Our definitions imply that
``y-5·δ/2 < x-ε < y-δ/2 ≤ x < y+δ/2 ≤ x+ε < y+5·δ/2``,
so we have decent bounds on the original value and uncertainty interval.
Moreover, if the original uncertainty ``ε`` is a known (monotone) function ``f``
of ``x``, we can do much better: We know ``ε = f(x)`` and let ``ε' = f(y)``,
then we can deduce a bound on ``|ε-ε'|`` by substituting ``y±δ/2`` for ``x``.
For example, if ``α > 0`` is a relative uncertainty, then ``|ε-ε'| < α·δ/2`` and
we can use ``ε'+α·δ/2`` as a conservative estimate for ``ε``.

For human consumption, a decimal rounding ``z`` that converts back to ``y``
knowing ``δ`` is useful. A relatively short one is ``z = ⌊y/γ⌉γ`` with
``lg(γ) = ⌊lg(δ/2)⌋``, [#log]_ so that ``γ < δ/2``. This guarantees that
``|x-z| ≤ γ+δ/2 < δ ≤ ε``. For our example, ``⌊lg(δ/2)⌋ = -2`` and thus
``lg(γ) = -2 = 0.01``, so that ``z = 0.64``. Note that ``δ`` cannot be
determined from ``γ``; e.g., ``10 < 16 < 32 < 64 < 100``.

Users of our convention—actually, all users of values with limited
accuracy—should be aware that not all values can be ordered. Namely, if for two
measurement values encoded as ``x¯±ε¯`` and ``x⁺±ε⁺`` we numerically have
``x¯<x⁺``, then this ordering has meaning only if ``|x¯-x⁺|`` is sufficiently
large compared to ``ε¯+ε⁺``. This is good to keep in mind when using our
convention, as order reversal—i.e., ``y¯>y⁺``—is possible when applying it if
``ε¯+ε⁺`` is sufficiently large compared to ``|x¯-x⁺|``.

A convention such as ours is necessary because the standardized binary floating
point formats ([IEEE-753]_, §3.4) are all normalized and can therefore not
encode a number of significant binary digits—i.e., bits. To wit, apart from
some special cases, such floating point numbers are effectively of the form
``1.s·2^e``, where ``e`` is an integer exponent and ``s`` is a fixed-length
string of fractional bits. The fixed leading integer bit ‘``1``’ precludes
having an initial string of zero bits that reduces the number of remaining,
significant bits.

There do exist unnormalized decimal floating point formats that encode the
number of significant decimal digits ([IEEE-753]_, §3.5), but power-of-ten
magnitudes are rather crude. A standardized unnormalized binary floating point
format would be useful; we have contacted the standardization committee about
this (without any real reaction). Also Dufour [D17]_, to get around the
lack of standardized unnormalized binary floating point format the context of a
proposal for significance arithmetic, introduces an encoding where trailing
zeros are considered insignificant; but unlike us, he considers the final ``1``
insignificant as well, reducing compatibility with the standard binary floating
point format.


.. [#log] We use ISO logarithm notation: ``lb`` for the base-2 logarithm and
    ``lg`` for the base-10 logarithm.

.. [#nearest] When ``x`` is a multiple of ``δ``, there are two odd multiples of
    ``δ/2`` nearest to ``x``. In that case the definition selects the one with
    the largest magnitude.

.. [IEEE-753] Cowlishaw, Mike, ed. (2008). IEEE Standard for Floating-Point
    Arithmetic. IEEE Std 754-2008.
    http://dx.doi.org/10.1109/IEEESTD.2008.4610935.

.. [D17] Defour, David (2017). FP-ANR: A representation format to handle
    floating-point cancellation at run-time. Research rep. lirmm-01549601.
    Version 1. https://hal-lirmm.ccsd.cnrs.fr/lirmm-01549601.
