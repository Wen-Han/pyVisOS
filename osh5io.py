#!/usr/bin/env python

"""
osh5io.py
=========
Disk IO for the OSIRIS data.
"""

import osh5io_backend
from os.path import basename
from osh5def import H5Data, PartData
from osh5def_auxil import fn_rule, DataAxis, OSUnits


def read_grid(filename, path=None, axis_name="AXIS/AXIS"):
    """
    Read grid data from Osiris/OSHUN output. Data can be in hdf5 or zdf format
    """
    ext = basename(filename).split(sep='.')[-1]

    if ext == 'h5':
        return read_h5(filename, path=path, axis_name=axis_name)
    elif ext == 'zdf':
        return read_zdf(filename, path=path)
    else:
        # the file extension may not be specified, trying all supported formats
        try:
            return read_h5(filename+'.h5', path=path, axis_name=axis_name)
        except OSError:
            return read_zdf(filename+'.zdf', path=path)


def read_zdf(filename, path=None):
    """
    HDF reader for Osiris/Visxd compatible ZDF files.
    Returns: H5Data object.
    """
    info = osh5io_backend.open_zdf_and_extract_data(filename, path=path)
    return H5Data(info[0], timestamp=info[1], data_attrs=info[2], run_attrs=info[3],
                  axes=info[4], runtime_attrs=info[5])


def read_h5(filename, path=None, axis_name="AXIS/AXIS"):
    """
    HDF reader for Osiris/Visxd compatible HDF files... This will slurp in the data
    and the attributes that describe the data (e.g. title, units, scale).

    Usage:
            diag_data = read_hdf('e1-000006.h5')      # diag_data is a subclass of numpy.ndarray with extra attributes

            print(diag_data)                          # print the meta data
            print(diag_data.view(numpy.ndarray))      # print the raw data
            print(diag_data.shape)                    # prints the dimension of the raw data
            print(diag_data.run_attrs['TIME'])        # prints the simulation time associated with the hdf5 file
            diag_data.data_attrs['UNITS']             # print units of the dataset points
            list(diag_data.data_attrs)                # lists all attributes related to the data array
            list(diag_data.run_attrs)                 # lists all attributes related to the run
            print(diag_data.axes[0].attrs['UNITS'])   # prints units of X-axis
            list(diag_data.axes[0].attrs)             # lists all variables of the X-axis

            diag_data[slice(3)]
                print(rw.view(np.ndarray))

    We will convert all byte strings stored in the h5 file to strings which are easier to deal with when writing codes
    see also write_h5() function in this file

    """
    info_list = osh5io_backend.open_h5_and_extract_data(filename, path=path, axis_name=axis_name)
    data_list = [ H5Data(info[0], timestamp=info[1], data_attrs=info[2],
                         run_attrs=info[3], axes=info[4], runtime_attrs=info[5])
                 for info in info_list ]

    if len(data_list) == 1:
        return data_list[0]
    else:
        return data_list


def read_h5_openpmd(filename, path=None):
    """
    HDF reader for OpenPMD compatible HDF files... This will slurp in the data
    and the attributes that describe the data (e.g. title, units, scale).

    Usage:
            diag_data = read_hdf_openpmd('EandB000006.h5')      # diag_data is a subclass of numpy.ndarray with extra attributes

            print(diag_data)                          # print the meta data
            print(diag_data.view(numpy.ndarray))      # print the raw data
            print(diag_data.shape)                    # prints the dimension of the raw data
            print(diag_data.run_attrs['TIME'])        # prints the simulation time associated with the hdf5 file
            diag_data.data_attrs['UNITS']             # print units of the dataset points
            list(diag_data.data_attrs)                # lists all attributes related to the data array
            list(diag_data.run_attrs)                 # lists all attributes related to the run
            print(diag_data.axes[0].attrs['UNITS'])   # prints units of X-axis
            list(diag_data.axes[0].attrs)             # lists all variables of the X-axis

            diag_data[slice(3)]
                print(rw.view(np.ndarray))

    We will convert all byte strings stored in the h5 file to strings which are easier to deal with when writing codes
    see also write_h5() function in this file

    """
    fld_dict = osh5io_backend.open_openpmd_and_extract_data(filename, path=path)

    for k, v in fld_dict.items():
        fld_dict[k] = H5Data(v[0], timestamp=v[0], data_attrs=v[1], run_attrs=v[2], axes=v[3], runtime_attrs=v[4])

    return fld_dict


def write_h5(data, filename=None, path=None, dataset_name=None, overwrite=True, axis_name=None):
    """
    Usage:
        write(diag_data, '/path/to/filename.h5')    # writes out Visxd compatible HDF5 data.

    Since h5 format does not support python strings, we will convert all string data (units, names etc)
    to bytes strings before writing.

    see also read_h5() function in this file

    """
    if isinstance(data, H5Data):
        data_object = data
    elif isinstance(data, np.ndarray):
        data_object = H5Data(data)
    else:
        try:  # maybe it's something we can wrap in a numpy array
            data_object = H5Data(np.array(data))
        except:
            raise Exception(
                "Invalid data type.. we need a 'hdf5_data', numpy array, or somehitng that can go in a numy array")
    osh5io_backend.write_h5(data, filename=filename, path=path, dataset_name=dataset_name,
                            overwrite=overwrite, axis_name=axis_name)


def write_h5_openpmd(data, filename=None, path=None, dataset_name=None, overwrite=True, axis_name=None,
    time_to_si=1.0, length_to_si=1.0, data_to_si=1.0 ):
    """
    Usage:
        write_h5_openpmd(diag_data, '/path/to/filename.h5')    # writes out Visxd compatible HDF5 data.

    Since h5 format does not support python strings, we will convert all string data (units, names etc)
    to bytes strings before writing.

    see also read_h5() function in this file

    """
    if isinstance(data, H5Data):
        data_object = data
    elif isinstance(data, np.ndarray):
        data_object = H5Data(data)
    else:
        try:  # maybe it's something we can wrap in a numpy array
            data_object = H5Data(np.array(data))
        except:
            raise Exception(
                "Invalid data type.. we need a 'hdf5_data', numpy array, or something that can go in a numy array")
    osh5io_backend.write_h5_openpmd(data, filename=filename, path=path, dataset_name=dataset_name,
                                    overwrite=overwrite, axis_name=axis_name,
                                    time_to_si=time_to_si, length_to_si=length_to_si, data_to_si=data_to_si )

def read_raw(filename, path=None):
    """
    Read particle raw data into a numpy sturctured array.
    See numpy documents for detailed usage examples of the structured array.
    The only modification is that the meta data of the particles are stored in .attrs attributes.

    Usage:
            part = read_raw("raw-electron-000000.h5")   # part is a subclass of numpy.ndarray with extra attributes

            print(part.shape)                           # should be a 1D array with # of particles
            print(part.attrs)                           # print all the meta data
            print(part.attrs['TIME'])                   # prints the simulation time associated with the hdf5 file
    """
    ext = basename(filename).split(sep='.')[-1]

    if ext == 'h5':
        data, quants, attrs = osh5io_backend.read_h5_raw(filename, path=path, axis_name=axis_name)
    elif ext == 'zdf':
        data, quants, attrs = osh5io_backend.read_zdf_raw(filename, path=path)
    else:
        # the file extension may not be specified, trying all supported formats
        try:
            data, quants, attrs = osh5io_backend.read_h5_raw(filename+'.h5', path=path, axis_name=axis_name)
        except OSError:
            data, quants, attrs = osh5io_backend.read_zdf_raw(filename+'.zdf', path=path)

    dtype = [(q, data[q].dtype) for q in quants]
    r = PartData(data[dtype[0][0]].shape, dtype=dtype, attrs=attrs)
    for dt in dtype:
        r[dt[0]] = data[dt[0]]

    return r


if __name__ == '__main__':
    import osh5utils as ut
    a = np.arange(6.0).reshape(2, 3)
    ax, ay = DataAxis(0, 3, 3, attrs={'UNITS': '1 / \omega_p'}), DataAxis(10, 11, 2, attrs={'UNITS': 'c / \omega_p'})
    da = {'UNITS': 'n_0', 'NAME': 'test', }
    h5d = H5Data(a, timestamp='123456', data_attrs=da, axes=[ay, ax], runtime_attrs={})
    write_h5(h5d, './test-123456.h5')
    rw = read_h5('./test-123456.h5')
    h5d = read_h5('./test-123456.h5')  # read from file to get all default attrs
    print("rw is h5d: ", rw is h5d, '\n')
    print(repr(rw))

    # let's read/write a few times and see if there are mutations to the data
    # you should also diff the output h5 files
    for i in range(5):
        write_h5(rw, './test' + str(i) + '-123456.h5')
        rw = read_h5('./test' + str(i) + '-123456.h5')
        assert (rw == a).all()
        for axrw, axh5d in zip(rw.axes, h5d.axes):
            assert axrw.attrs == axh5d.attrs
            assert (axrw == axh5d).all()
        assert h5d.timestamp == rw.timestamp
        assert h5d.name == rw.name
        assert h5d.data_attrs == rw.data_attrs
        assert h5d.run_attrs == rw.run_attrs
        print('checking: ', i+1, 'pass completed')

    # test some other functionaries
    print('\n meta info of rw: ', rw)
    print('\nunit of rw is ', rw.data_attrs['UNITS'])
    rw **= 3
    print('unit of rw^3 is ', rw.data_attrs['UNITS'])
    print('contents of rw^3: \n', rw.view(np.ndarray))
