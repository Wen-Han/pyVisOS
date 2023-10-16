#!/usr/bin/env python

"""
osh5def.py
==========
Define the OSIRIS HDF5 data class and basic functions.
    The basic idea is to make the data unit and axes consistent with the data
    itself. Therefore users should only modify the unit and axes by modifying
    the data or by dedicated functions (unit conversion for example).
"""

import numpy as np
import copy as cp
from itertools import product
from osh5io_backend import open_zdf_and_extract_data, open_h5_and_extract_data
import warnings
from osh5def_auxil import fn_rule, DataAxis, OSUnits
try:
    import xarray as xr
    _has_xarray_support = True
except ImportError:
    _has_xarray_support = False


# TODO: this list is incomplete, it lacks line/slice diagnostics
_known_fields = product(('', 'ext_', 'part_'), ('e','b'))
_known_fields = tuple(''.join(s) for s in _known_fields) + ('j', 's')
_known_fields = product(_known_fields, ('1','2','3'), ('', '-savg', '-senv'), ('', '-tavg'))
_known_fields = tuple(''.join(s) for s in _known_fields) + ('ene_e', 'ene_b', 'ene_emf', 'div_e', 'div_b', 'chargecons', 'psi')


class _LocIndexer(object):
    def __init__(self, h5data):
        self.__data = h5data

    def label_slice2int_slice(self, i, slc):
        bnd = (slc.start, slc.stop, slc.step)
        return slice(*H5Data.get_index_slice(self.__data.axes[i], bnd))

    def label2int(self, i, label):
        ax = self.__data.axes[i]
        ind = int(round(max(label - ax.min, 0) / ax.increment)) if ax.increment > 0 else -1
        return min(ind, self.__data.shape[i] - 1)

    def iterable2int_list(self, i, iterable):
        return H5Data.get_index_list(self.__data.axes[i], iterable)

    def __convert_index(self, index):
        try:
            iter(index)
            idxl = index
        except TypeError:
            idxl = [index]
        converted, nn, dn = [], self.__data.ndim - len(idxl) + idxl.count(None) + 1, 0
        for i, idx in enumerate(idxl):
            if isinstance(idx, slice):
                converted.append(self.label_slice2int_slice(dn, idx))
                dn += 1
            elif isinstance(idx, (int, float)):
                converted.append(self.label2int(dn, idx))
                dn += 1
            elif idx is Ellipsis:
                [converted.append(slice(None,)) for _ in range(nn)]
                dn += nn
            elif idx is None:
                converted.append(None)
            else:
                try:
                    iter(idx)
                    converted.append(self.iterable2int_list(dn, idx))
                    dn += 1
                except TypeError:
                    # It is something we don't recognize, now hopefully idx know how to convert to int
                    converted.append(int(idx))
                    dn += 1
        return converted

    def __getitem__(self, index):
        return self.__data[tuple(self.__convert_index(index))]

    def __setitem__(self, index, value):
        self.__data.values[tuple(self.__convert_index(index))] = value


# Important: the first occurrence of serial numbers before '.' must be the time stamp information


class H5Data(np.ndarray):

    def __new__(cls, input_array, timestamp=None, data_attrs=None, run_attrs=None, axes=None, runtime_attrs=None):
        """wrap input_array into our class, and we don't copy the data!"""
        obj = np.asarray(input_array).view(cls)
        if timestamp:
            obj.timestamp = timestamp
        # if name:
        #     obj.name = name
        if data_attrs:
            obj.data_attrs = cp.deepcopy(data_attrs)  # there is OSUnits obj inside
        if run_attrs:
            obj.run_attrs = cp.deepcopy(run_attrs)
        if runtime_attrs:
            obj.runtime_attrs = cp.deepcopy(runtime_attrs)
        if axes:
            obj.axes = cp.deepcopy(axes)   # the elements are numpy arrays
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.timestamp = getattr(obj, 'timestamp', '0' * 6)
        # self.name = getattr(obj, 'name', 'data')
#         if self.base is obj:
#             self.data_attrs = getattr(obj, 'data_attrs', {})
#             self.run_attrs = getattr(obj, 'run_attrs', {})
#             self.axes = getattr(obj, 'axes', [])
#         else:
#             self.data_attrs = cp.deepcopy(getattr(obj, 'data_attrs', {}))
#             self.run_attrs = cp.deepcopy(getattr(obj, 'run_attrs', {}))
#             self.axes = cp.deepcopy(getattr(obj, 'axes', []))
        self.data_attrs, self.run_attrs, self.axes, self.runtime_attrs = (cp.deepcopy(getattr(obj, 'data_attrs', {})),
                                                                          cp.deepcopy(getattr(obj, 'run_attrs', {})),
                                                                          cp.deepcopy(getattr(obj, 'axes', [])),
                                                                          cp.deepcopy(getattr(obj, 'runtime_attrs', {})))

    @property
    def T(self):
        return self.transpose()

    @property
    def name(self):
        return self.data_attrs.get('NAME', '')

    @name.setter
    def name(self, s):
        self.data_attrs['NAME'] = str(s)

    @property
    def label(self):
        return self.data_attrs.get('LONG_NAME', '')

    @label.setter
    def label(self, s):
        self.data_attrs['LONG_NAME'] = str(s)

    @property
    def units(self):
        return self.data_attrs.get('UNITS', OSUnits('a.u.'))

    @property
    def values(self):
        return self.view(np.ndarray)

    @values.setter
    def values(self, numpy_ndarray):
        v = self.view(np.ndarray)
        v[()] = numpy_ndarray[()]

    # need the following two function for mpi4py high level function to work correctly
    def __setstate__(self, state, *args):
        self.__dict__ = state[-1]
        super(H5Data, self).__setstate__(state[:-1], *args)

    # It looks like mpi4py/ndarray use reduce for pickling. One would think setstate/getstate pair should also work but
    # it turns out the __getstate__() function is never called!
    # Luckily ndarray doesn't use __dict__ so we can pack everything in it.
    def __reduce__(self):
        ps = super(H5Data, self).__reduce__()
        ms = ps[2] + (self.__dict__,)
        return ps[0], ps[1], ms

    def __getstate__(self):
        return self.__reduce__()

    def __str__(self):
        if not self.shape:
            return str(self.values)
        else:
            return ''.join([self.name, '-', self.timestamp, ', shape: ', str(self.shape), ', time:',
                            str(self.run_attrs['TIME']), ' [', str(self.run_attrs['TIME UNITS']), ']\naxis:\n  ',
                            '\n  '.join([str(ax) for ax in self.axes]) if len(self.axes) else 'None'])

    def __repr__(self):
        return ''.join([str(self.__class__.__module__), '.', str(self.__class__.__name__), ' at ', hex(id(self)),
                        ', shape', str(self.shape), ',\naxis:\n  ',
                        '\n  '.join([repr(ax) for ax in self.axes]) if len(self.axes) else 'None',
                        '\ndata_attrs: ', repr(self.data_attrs), '\nrun_attrs:', repr(self.run_attrs)])

    def __getitem__(self, index):
        """I am inclined to support only basic indexing/slicing. Otherwise it is too difficult to define the axes.
             However we would return an ndarray if advance indexing is invoked as it might help things floating...
        """
        ndim = self.ndim
        try:
            v = super(H5Data, self).__getitem__(index)
        except IndexError:
            return self.view(np.ndarray)  # maybe it is a scalar
        try:
            iter(index)
            idxl = index
        except TypeError:
            idxl = [index]
        try:
            converted, nn, dn = [], ndim - len(idxl) + idxl.count(None) + 1, 0
            for idx in idxl:
                if isinstance(idx, int):  # idx is a trivial dimension now, record its name and value before it is gone
                    try:
                        v.runtime_attrs.setdefault(v.axes[dn].name,
                                                   (v.axes[dn][idx], v.axes[dn].units))
                        del v.axes[dn]
                    except AttributeError:
                        break
                elif isinstance(idx, slice):  # also slice the axis
                    v.axes[dn].ax = v.axes[dn].ax[idx]
                    dn += 1
                elif idx is Ellipsis:  # let's fast forward to the next explicitly referred axis
                    dn += nn
                elif idx is None:  # in numpy None means newAxis
                    v.axes.insert(dn, DataAxis(0., 1., 1))
                    dn += 1
                else:  # type not supported
                    return v.view(np.ndarray)
        except:
            return v.view(np.ndarray)
        return v

    def __getattr__(self, label):
        try:
            return getattr(self.values, label)
        except AttributeError:  # maybe it is an axis name
            try:
                ind = self.index_of(label)
            except ValueError:
                raise AttributeError()
            axes = np.meshgrid(*reversed([x.ax for x in self.axes]), sparse=True)
            return axes[self.ndim-1-ind].copy()

    def meta2dict(self):
        """return a deep copy of the meta data as a dictionary"""
        return cp.deepcopy(self.__dict__)

    def transpose(self, *axes):
        v = super(H5Data, self).transpose(*axes)
        if axes == () or axes[0] is None:  # axes is none, numpy default is to reverse the order
            axes = range(len(v.axes)-1, -1, -1)
        try:                               # called like transpose(2, 1, 0)
            v.axes = [self.axes[i] for i in axes]
        except TypeError:                  # called like transpose([2, 1, 0])
            v.axes = [self.axes[i] for i in axes[0]]
        return v

    def __del_axis(self, axis):
        # axis cannot be None
        if isinstance(axis, int):
            del self.axes[axis]
        else:
            # remember axis index can be negative
            self.axes = [v for i, v in enumerate(self.axes) if i not in axis and i - self.ndim not in axis]

    def __ufunc_with_axis_handled(self, func, *args, **kwargs):
        try:
            iter(kwargs['axis'])
            if isinstance(kwargs['axis'][0], str):
                kwargs['axis'] = self.index_of(kwargs['axis'])
        except TypeError:
            if isinstance(kwargs['axis'], str):
                kwargs['axis'] = self.index_of(kwargs['axis'])
        if not kwargs['keepdims']:
            # default is to reduce over all axis, return a value
            _axis = range(0, self.ndim) if kwargs['axis'] is None else kwargs['axis']

        # we need a better way
        o = func(*args, **kwargs)

    # def __ufunc_del_axis(self, axis, ):
        if not kwargs['keepdims']:
            o.__del_axis(_axis)
            if kwargs['out']:
                kwargs['out'].__del_axis(_axis)
        return o

    def mean(self, axis=None, dtype=None, out=None, keepdims=False):
        return self.__ufunc_with_axis_handled(super(H5Data, self).mean,
                                              axis=axis, dtype=dtype, out=out, keepdims=keepdims)

    def sum(self, axis=None, out=None, dtype=None, keepdims=False):
        return self.__ufunc_with_axis_handled(super(H5Data, self).sum,
                                              axis=axis, dtype=dtype, out=out, keepdims=keepdims)

    def min(self, axis=None, out=None, keepdims=False):
        return self.__ufunc_with_axis_handled(super(H5Data, self).min, axis=axis, out=out, keepdims=keepdims)

    def max(self, axis=None, out=None, keepdims=False):
        return self.__ufunc_with_axis_handled(super(H5Data, self).max, axis=axis, out=out, keepdims=keepdims)

    def std(self, axis=None, dtype=None, out=None, ddof=0, keepdims=False):
        return self.__ufunc_with_axis_handled(super(H5Data, self).std,
                                              axis=axis, dtype=dtype, out=out, ddof=0, keepdims=keepdims)

    def argmax(self, axis=None, out=None):
        return np.argmax(self.view(np.ndarray), axis=axis, out=out)

    def argmin(self, axis=None, out=None):
        return np.argmin(self.view(np.ndarray), axis=axis, out=out)

    def ptp(self, axis=None, out=None):
        return self.__ufunc_with_axis_handled(super(H5Data, self).ptp, axis=axis, out=out)

    def swapaxes(self, axis1, axis2):
        o = super(H5Data, self).swapaxes(axis1, axis2)
        o.axes[axis1], o.axes[axis2] = o.axes[axis2], o.axes[axis1]
        return o

    def var(self, axis=None, dtype=None, out=None, ddof=0, keepdims=False):
        return self.__ufunc_with_axis_handled('var', axis=axis, dtype=dtype, out=out, ddof=0, keepdims=keepdims)

    def squeeze(self, axis=None):
        v = super(H5Data, self).squeeze(axis=axis)
        if axis is None:
            axis = [i for i, d in enumerate(self.shape) if d <= 1]
        for i in reversed(axis):
            del v.axes[i]
        return v

    def __array_wrap__(self, out, context=None):
        """Here we handle the unit attribute
        We do not check the legitimacy of ufunc operating on certain unit. We hard code a few unit changing
        rules according to what ufunc is called
        """
        # the document says that __array_wrap__ could be deprecated in the future but this is the most direct way...
        div, mul = 1, 2
        op, __ufunc_mapping = None, {'sqrt': '1/2', 'cbrt': '1/3', 'square': '2', 'power': '', 'divide': div,
                                     'true_divide': div, 'floor_divide': div, 'reciprocal': '-1', 'multiply': mul}
        if context:
            op = __ufunc_mapping.get(context[0].__name__)
        if op is not None:
            try:
                if isinstance(op, str):
                    if not op:  # op is 'power', get the second operand
                        op = context[1][1]
                    out.data_attrs['UNITS'] **= op
                elif op == div:  # op is divide
                    if not isinstance(context[1][0], H5Data):  # nominator has no unit
                        out.data_attrs['UNITS'] **= '-1'
                    else:
                        out.data_attrs['UNITS'] = context[1][0].data_attrs['UNITS'] / context[1][1].data_attrs['UNITS']
                else:  # op is multiply
                    out.data_attrs['UNITS'] = context[1][0].data_attrs['UNITS'] * context[1][1].data_attrs['UNITS']
            except (AttributeError, KeyError):  # .data_attrs['UNITS'] failed
                pass
        return np.ndarray.__array_wrap__(self, out, context)

    @staticmethod
    def get_index_slice(ax, bd):
        """
            given a list-like bd that corresponds to a slice, return a list of index along axis ax that can be casted
            to slice, clip to boundary if the value is out of bound. Note that this method does not account for reverse
            indexing, aka bd[0] must be no bigger than bd[1].
        """
        if ax.increment <= 0:
            return (None,)
        # clip lower bound
        tmp = [int(round(max(co - ax.min, 0) / ax.increment)) if co is not None else None for co in bd[:2]]
        # clip upper bound
        if tmp[1] is not None:
            tmp[1] = min(ax.size, tmp[1])
        # determine step, step must be >= 1
        if len(bd) == 3:
            tmp.append(None if bd[2] is None else max(int(round(bd[2] / ax.increment)), 1))
        return tmp

    @staticmethod
    def get_index_list(ax, lst):
        """
            given a list-like lst, return a list of index along axis ax, clip to boundary if the value is out of bound.
        """
        if ax.increment <= 0:
            return None,
        # use min max to clip to legal bound
        tmp = list(min(int(round(max(co - ax.min, 0) / ax.increment)), ax.size - 1) for co in lst)
        return tmp

    @staticmethod
    def __get_axes_bound(axes, bound):  # bound should have depth of 2
        # i keeps track of the axes we are working on
        ind, i = [], 0
        for bnd in bound:
            if bnd is Ellipsis:
                # fill in all missing axes
                for j in range(len(axes) - len(bound) + 1):
                    ind.append(slice(None))
                    i += 1
                continue
            elif not bnd:  # None or empty list/tuple or 0
                ind.append(slice(None))
            elif len(bnd) == 3:  # slice with step, None can appear at any places
                sgn = int(np.sign(bnd[2]))
                if sgn == 1:
                    se = H5Data.get_index_slice(axes[i], bnd[0:2])
                else:
                    se = H5Data.get_index_slice(axes[i], reversed(bnd[0:2]))
                    se = list(reversed(se))
                step = int(round(bnd[2] / axes[i].increment))
                se.append(step if abs(step) > 0 else sgn)
                ind.append(slice(*se))
            else:  # len(bnd) == 1 or 2
                ind.append(slice(*H5Data.get_index_slice(axes[i], bnd)))
            i += 1
        return ind

    @property
    def loc(self):
        return _LocIndexer(self)

    # get the depth of list/tuple recursively, empty list/tuple will raise ValueError
    @staticmethod
    def __check_bound_depth(bnd):
        return max(H5Data.__check_bound_depth(b) for b in bnd) + 1 if isinstance(bnd, (tuple, list, np.ndarray)) else 0

    @staticmethod
    def __get_symmetric_bound(axes, index):
        return [slice(*[None if idx.stop is None else ax.size - idx.stop,
                        None if idx.start is None else ax.size - idx.start,
                        idx.step]) for ax, idx in zip(axes, index)]

    def index_of(self, axis_name):
        """return the index of the axis given its name. raise ValueError if not found"""
        axn = [ax.name for ax in self.axes]
        try:
            if isinstance(axis_name, str):
                return axn.index(axis_name)
            else:
                return tuple(axn.index(a) for a in axis_name)
        except ValueError:
            raise ValueError('one or more of axis names not found in axis list ' + str(axn))

    def has_axis(self, axis_name):
        """check if H5Data has axis with name axis_name"""
        return axis_name in [ax.name for ax in self.axes]

    def sel(self, new=False, **bound):
        """
            indexing H5Data object by axis name
        :param bound: keyword dict specifying the axis name and range
        :param new: if True return a copy of the object
        :return: a copy or a view of the H5Data object depending on what bound looks like
        Examples:
            # use a dictionary
            a.sel({'x1': slice(0.4, 0.7, 0.02), 'p1': 0.5}) will return an H5Data oject whose x1 axis range
                is (0.4, 0.7) with 0.02 spacing and p1 axis equal to 0.5. aka the return will be one dimension
                less than a
            # use keyword format to do the same thing
            a.sel(x1=slice(0.4, 0.7, 0.02), p1=0.5)
            # use index other than silce or int will return a numpy ndarray (same as the numpy array advanced
            # indexing rule). The following return a numpy ndarray
            a.sel(x1=[0.2,0.5,0.8])
        """
        # early termination
        if not bound:
            return self

        ind = [slice(None,)] * self.ndim
        for axn, bnd in bound.items():
            ind[self.index_of(axn)] = bnd if isinstance(bnd, (slice, int, float)) else slice(*bnd)
        res = self.loc[tuple(ind)]
        if new:
            return res.copy() if res.base is self else res
        return res

    def subrange(self, bound=None, new=False):
        """
        use .axes[:] data to index the array
        :param bound: see bound as using slice indexing with floating points. [1:3, :, 2:9:2, ..., :8] for a 6-D array
            with dx=0.1 in all directions would look like bound=[(0.1, 0.3), None, (0.2, 0.9, 0.2), ..., (None, 0.8)]
        :param new: if true return a new array instead of a view
        """
        warnings.warn(".subrange will be removed from future version. "
                      "Please use .loc or .sel instead (They are also more intuitive)", DeprecationWarning, stacklevel=2)
        if not bound:  # None or empty list/tuple or 0
            return self
        if H5Data.__check_bound_depth(bound) == 1:
            bound = (bound,)
        index = self.__get_axes_bound(self.axes, bound)
        if new:
            return cp.deepcopy(self[index])
        else:
            return self[index]

    def set_value(self, bound=None, val=(0,), symmetric=True, inverse_select=False, method=None):
        """
        set values inside bound using val
        :param bound: lists of triples of elements marking the lower and higher bound of the bound, e.g.
                [(l1,u1), (l2,u2), (l3,u3)] in 3d data marks data[l1<z<u1, l2<y<u2, l3<x<u3]. See bound parameter
                in function subrange for more detail. There can be multiple lists like this marking multiple regions.
                Note that if ln or un is out of bound they will be clipped to the array boundary.
        :param val: what value to set, default is (0,). val can also be list of arrays with the same size
                as each respective bound.
        :param inverse_select: set array elements OUTSIDE specified bound to val. This will used more memory
                as temp array will be created. If true val can only be a number.
        :param symmetric: if true then the mask will also apply to the center-symmetric regions where center is define
                as nx/2 for each dimension
        :param method: how to set the value. if None the val will be assigned. User can provide function that accept
                two parameters, and the result is self[bound[i]] = user_func(self[bound[i]], val[i]). Note that
                method always acts on the specified bound regardless of inverse_select being true or not.
        """
        if not bound:
            return
        # convert to ndarray for maximum compatibility
        v = self.view(np.ndarray)
        bdp = H5Data.__check_bound_depth(bound)
        if bdp == 1:  # probably 1D data
            bound = ((bound,),)
        elif bdp == 2:  # only one region is set
            bound = (bound,)
        elif bdp > 3:  # 3 level max: list of regions; list of bounds in a region; start, end, step in one bound
            raise ValueError('Too many levels in the bound parameter')
        try:
            iter(val)
        except TypeError:
            val = (val,)
        rec, vallen = [], len(val)
        for i, bnd in enumerate(bound):
            index = [self.__get_axes_bound(self.axes, bnd)]
            if symmetric:
                index.append(H5Data.__get_symmetric_bound(self.axes, index[0]))
            for idx in index:
                idx = tuple(idx)
                if inverse_select:  # record original data for later use
                    if method:
                        rec.append((idx, cp.deepcopy(method(v[idx], val[i % vallen]))))
                    else:
                        rec.append((idx, cp.deepcopy(v[idx])))
                else:
                    if method:
                        v[idx] = method(v[idx], val[i % vallen])
                    else:
                        v[idx] = val[i % vallen]
        if inverse_select:
            # set everything to val and restore the original data in regions
            self.fill(val[0])
            for r in rec:
                v[r[0]] = r[1]


    def meanwhile_get(self, qname, convert=1.0, file_prefix=None, data_dir_prefix='/MS/'):
        """
        return the quantity specified by qname in the same simdir and
          with the same timestamp as the (convert*timestamp) of this h5data.
        :param qname: a string specifying the data file to read.
                      For example, 'e1' for '/MS/FLD/e1'.
                      If this h5data is phasespace data, say 'p1x1/eletrons', one can specify part of the data path:
                      'p2p1' will return 'p2p1/electrons' and 'ions' will return 'p1x1/ions'.
        :param convert: read the data file with timestamp of (convert*h5data's timestamp)
        :param file_prefix: a string for the file prefix right before the timestamp, if omitted it will be inferred from qname
        :param data_dir_prefix: a string specifying where the 'FLD', 'PHA', 'RAW' etc reside
        :return: an h5Data object
        """
        ts = self.timestamp
        ts = str(round(convert*int(ts))).zfill(len(ts))
        fpath, ext = self.runtime_attrs['simdir'] + data_dir_prefix, self.runtime_attrs['extension']
        if '/' in qname:  # user has been very specific, so do exactly as they ask
            if qname[-1] == '/':
                qname = qname[:-1]
            # See if user is asking for a field, if not we assume they want phasespace
            # TODO: in pricipal we can also try to read RAW data
            if 'FLD/' not in qname:
                qname = 'PHA/' + qname
            fp = qname[4:].replace('/', '-') if file_prefix == None else file_prefix
            fname = fpath + qname + '/' + fp + '-' + ts + ext
        elif qname in _known_fields:  # it is a physical field
            fp = qname if file_prefix == None else file_prefix
            fname = fpath + '/FLD/' + qname + '/' + fp + '-' + ts + ext
        else:  # looks like it is phasespace data
            species = path.basename(self.runtime_attrs['dirname'])
            pha = path.basename(path.dirname(self.runtime_attrs['dirname']))
            fpath += '/PHA/'
            # fuzzy matching the phase space name, false positive and false negative are possible!
            if qname == 'g' or qname == 'gl':
                fp = qname + '-' + species if file_prefix == None else file_prefix
                fname = fpath + qname + '/' + species + '/' + fp + '-' + ts + ext
            else:
                index = 2 if qname[0:2] == 'gl' else 0
                index = 1 if qname[0] == 'g' else 0
                if qname[index] in 'xpl' and qname[index+1] in '123': # qname is a phase space name
                    fp = qname + '-' + species if file_prefix == None else file_prefix
                    fname = fpath + qname + '/' + species + '/' + fp + '-' + ts + ext
                else: # qname is a species name
                    fp = pha + '-' + qname if file_prefix == None else file_prefix
                    fname = fpath + pha + '/' + qname + '/' + fp + '-' + ts + ext
        info = ( open_zdf_and_extract_data(fname)
                if ext=='.zdf'
                else open_h5_and_extract_data(fname) )
        return H5Data(info[0], timestamp=info[1], data_attrs=info[2], run_attrs=info[3],
                      axes=info[4], runtime_attrs=info[5])

    if _has_xarray_support:
        # experimental
        def to_xarray(self, copy=False):
            """
            convert H5Data format to an xarray.DataArray
            :param copy: if True the array data will be copied instead of being taken a view
            :return: xarray.DataArray
            """
            dim_dict = {}
            data = cp.deepcopy(self) if copy else self

            # fill out the axes dict form
            for ax in data.axes:
                ax_data = {'dims': ax.name, 'data': ax.ax.copy(), 'attrs': {'units': str(ax.units)}}
                for k, v in ax.attrs.items():
                    ax_data['attrs'].setdefault(k, v)
                dim_dict[ax.name] = ax_data

            # self.units will be converted to str by xarray, no special treatment needed
            data_attrs_dict = {}
            data_attrs_dict.update(data.run_attrs)
            data_attrs_dict.update(data.data_attrs)

            dims_name = tuple(k for k, v in dim_dict.items())

            data_dict = {'coords': dim_dict, 'attrs': data_attrs_dict,
                         'dims': dims_name, 'data': data.view(np.ndarray), 'name': data.name}
            return xr.DataArray.from_dict(data_dict)


# basically a copy of the numpy array subclassing tutorial
class PartData(np.ndarray):
    """
    A modified numpy structured array storing particles raw data. See numpy documents on structured array for detailed examples.
    The only modification is that the meta data of the particles are stored in .attrs attributes.

    Simple indexing examples (assuming part is the PartData instance):
    part[123]: return the raw data (coordinates, momenta, charge etc) of particle 123.
    part['x1']: return the 'x1' coordinate of all particles.
    """
    def __new__(subtype, shape, dtype=float, buffer=None, offset=0,
                strides=None, order=None, attrs=None):
        obj = super(PartData, subtype).__new__(subtype, shape, dtype,
                                                buffer, offset, strides,
                                                order)
        obj.attrs = attrs
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.attrs = getattr(obj, 'attrs', None)

    def __find_attrs_by_named_id(self, attr_name, quant=None):
        if quant:
            ind = self.attrs['QUANTS'].index(quant)
            return self.attrs[attr_name][ind]
        else:
            return tuple(self.attrs[attr_name])

    def label(self, quant=None):
        warnings.warn(".label will return the label of the species in the future. Use .label_of to return the label of quantities",
                      DeprecationWarning, stacklevel=2)
        return self.label_of(quant=quant)

    def units(self, quant=None):
        warnings.warn(".units will remove or change behavior in the future. Use .units_of to return the units of quantities",
                      DeprecationWarning, stacklevel=2)
        return self.units_of(quant=quant)


    def label_of(self, quant=None):
        return self.__find_attrs_by_named_id('LABELS', quant)

    def units_of(self, quant=None):
        return self.__find_attrs_by_named_id('UNITS', quant)

    @property
    def timestamp(self):
        try:
            return self.attrs['TIMESTAMP']
        except KeyError:
            return '0' * 6

    @timestamp.setter
    def timestamp(self, ts):
        try:
            int(ts)
            self.attrs['TIMESTAMP'] = str(ts)
        except ValueError:
            raise ValueError('Illigal timestamp format, must be integer of base 10')
