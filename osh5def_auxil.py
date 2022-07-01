"""
osh5def_auxil.py
==========
Define auxiliary classes to be used by the H5Data class
"""

import re
import numpy as np
import copy as cp
from fractions import Fraction as frac

fn_rule = re.compile(r'.(\d+)\.')

class DataAxis:
    def __init__(self, axis_min=0., axis_max=1., axis_npoints=1, attrs=None, data=None):
        if data is None:
            if axis_min > axis_max:
                raise Exception('illegal axis range: [ %(l)s, %(r)s ]' % {'l': axis_min, 'r': axis_max})
            self.ax = np.linspace( axis_min, axis_max, axis_npoints, endpoint=False )
        else:
            self.ax = data
        # now make attributes for axis that are required..
        if attrs is None:
            self.attrs = {'UNITS': OSUnits('a.u.'), 'LONG_NAME': "", 'NAME': ""}
        else:
            attrs = cp.deepcopy(attrs)
            self.attrs = {'LONG_NAME': attrs.pop('LONG_NAME', ""), 'NAME': attrs.pop('NAME', "")}
            try:
                u = attrs.pop('UNITS', 'a.u.')
                self.attrs['UNITS'] = OSUnits(u)
            except ValueError:
                self.attrs['UNITS'] = u
        # get other attributes for the AXIS
        if attrs:
            self.attrs.update(attrs)

    def __str__(self):
        return ''.join([str(self.attrs['NAME']), ': [', str(self.ax[0]), ', ', str(self.max), '] ',
                        str(self.attrs['UNITS'])])

    def __repr__(self):
        if len(self.ax) == 0:
            return 'None'
        return ''.join([str(self.__class__.__module__), '.', str(self.__class__.__name__), ' at ', hex(id(self)),
                        ': size=', str(self.ax.size), ', (min, max)=(', repr(self.ax[0]), ', ',
                        repr(self.max), '), ', repr(self.attrs)])

    def __getitem__(self, index):
        return self.ax[index]

    def __eq__(self, other):
        return (self.ax == other.ax).all()

    # def __getstate__(self):
    #     return self.ax[0], self.ax[-1], self.size, self.attrs
    #
    # def __setstate__(self, state):
    #     self.ax = np.linspace(state[0], state[1], state[2])
    #     self.attrs = state[3]

    @property
    def name(self):
        return self.attrs['NAME']

    @name.setter
    def name(self, s):
        self.attrs['NAME'] = str(s)

    @property
    def long_name(self):
        return self.attrs['LONG_NAME']

    @long_name.setter
    def long_name(self, s):
        self.attrs['LONG_NAME'] = str(s)

    @property
    def units(self):
        return self.attrs['UNITS']

    @property
    def min(self):
        return self.ax[0]

    @property
    def max(self):
        try:
            return self.ax[-1] + self.ax[1] - self.ax[0]
        except IndexError:
            return self.ax[-1]

    @property
    def size(self):
        return self.ax.size

    def __len__(self):
        return self.ax.size

    @property
    def increment(self):
        try:
            return self.ax[1] - self.ax[0]
        except IndexError:
            return 0

    def to_phys_unit(self, wavelength=None, density=None, **_unused):
        """
        convert this axis to physical units. note that this function won't change the actual axis.
        the copy of the axis data is returned
        :param wavelength: laser wavelength in micron
        :param density: critical plasma density in cm^-3
        :return: a converted axes, unit
        """
        fac, unit = self.punit_convert_factor(wavelength=wavelength, density=density)
        return self.ax * fac, unit

    def punit_convert_factor(self, wavelength=None, density=None, **_ununsed):
        """
        converting factor of physical units
        """
        if not wavelength:
            if not density:
                wavelength = 0.351
                density = 1.12e21
            else:
                wavelength = 1.98e10 * np.sqrt(1./density)
        elif not density:
            density = 3.93e20 * wavelength**2
        try:
            if self.attrs['UNITS'].is_frequency():
                return 2.998e2 / wavelength, 'THz'
            if self.attrs['UNITS'].is_time():
                return wavelength * 5.31e-4, 'ps'
            if self.attrs['UNITS'].is_length():
                return wavelength / (2 * np.pi), '\mu m'
            if self.attrs['UNITS'].is_density():
                return density, 'cm^{-3}'
        except AttributeError:  # self.attrs['UNITS'] is str?
            pass
        return 1.0, str(self.units)


class OSUnits:
    name = ('m_e', 'c', '\omega', 'e', 'n_0')
    disp_name = ['m_e', 'c', '\omega_p', 'e', 'n_0', 'a.u.']
    xtrnum = re.compile(r"(?<=\^)\d+|(?<=\^{).*?(?=})")

    def __init__(self, s):
        """
        :param s: string notation of the units. there should be whitespace around quantities and '/' dividing quantities
        """
        if isinstance(s, OSUnits):
            self.power = cp.deepcopy(s.power)
        else:
            self.power = np.array([frac(0), frac(0), frac(0), frac(0), frac(0)])
            if isinstance(s, bytes):
                s = s.decode("utf-8")
            if 'a.u.' != s:
                s = re.sub('/(?![^{]*})', ' / ', s)
                sl = s.split()
                nominator = True
                while sl:
                    ss = sl.pop(0)
                    if ss == '/':
                        nominator = False
                        continue
                    for p, n in enumerate(OSUnits.name):
                        if n == ss[0:len(n)]:
                            res = OSUnits.xtrnum.findall(ss)  # extract numbers
                            if res:
                                self.power[p] = frac(res[0]) if nominator else -frac(res[0])
                            else:
                                self.power[p] = frac(1, 1) if nominator else frac(-1, 1)
                            break
                        elif ss in ['1', '2', '\pi', '2\pi']:
                            break
                    else:
                        raise ValueError('Unknown unit: ' + re.findall(r'\w+', ss)[0])

    def tex(self):
        return '$' + self.__str__() + '$' if self.__str__() else self.__str__()

    def limit_denominator(self, max_denominator=64):
        """call fractions.Fraction.limit_denominator method for each base unit"""
        self.power = np.array([u.limit_denominator(max_denominator=max_denominator) for u in self.power])

    def is_time(self):
        return (self.power == np.array([frac(0), frac(0), frac(-1), frac(0), frac(0)])).all()

    def is_frequency(self):
        return (self.power == np.array([frac(0), frac(0), frac(1), frac(0), frac(0)])).all()

    def is_velocity(self):
        return (self.power == np.array([frac(0), frac(1), frac(0), frac(0), frac(0)])).all()

    def is_length(self):
        return (self.power == np.array([frac(0), frac(1), frac(-1), frac(0), frac(0)])).all()

    def is_density(self):
        return (self.power == np.array([frac(0), frac(0), frac(0), frac(0), frac(1)])).all()

    def __mul__(self, other):
        res = OSUnits('a.u.')
        res.power = self.power + other.power
        return res

    def __truediv__(self, other):
        res = OSUnits('a.u.')
        res.power = self.power - other.power
        return res

    __floordiv__ = __truediv__
    __div__ = __truediv__

    def __pow__(self, other, modulo=1):
        res = OSUnits('a.u.')
        res.power = self.power * frac(other)
        return res

    def __eq__(self, other):
        return (self.power == other.power).all()

    def __str__(self):
        disp = ''.join(['' if p == 0 else n + " " if p == 1 else n + '^{' + str(p) + '} '
                        for n, p in zip(OSUnits.disp_name[:-1], self.power)])
        if not disp:
            return OSUnits.disp_name[-1]
        return disp

    def __repr__(self):
        return ''.join([str(self.__class__.__module__), '.', str(self.__class__.__name__), ' at ', hex(id(self)),
                        ': ', repr(self.name), '=(', ', '.join([str(fr) for fr in self.power]), ')'])

    def encode(self, *args, **kwargs):
        return self.__str__().encode(*args, **kwargs)
