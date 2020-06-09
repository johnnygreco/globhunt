# Taken and modified from the LSST software stack
# LSST Data Management System
# Copyright 2008, 2009, 2010, 2011 LSST Corporation.
#
# This product includes software developed by the
# LSST Project (http://www.lsst.org/).
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the LSST License Statement and
# the GNU General Public License along with this program.  If not,
# see <http://www.lsstcorp.org/LegalNotices/>.
#
# **** JPG update:
# Added __getitem__ and __setitem__ attributes so structs will 
# also behave as dictionaries if desired.

from __future__ import print_function, division

__all__ = ["LsstStruct"]


class LsstStruct(object):
    """A container to which you can add fields as attributes.

    Parameters
    ----------
    keyArgs
        keyword arguments specifying fields and their values.

    Examples
    --------
    >>> myStruct = Struct(
    >>>     strVal = 'the value of the field named "strVal"',
    >>>     intVal = 35,
    >>> )
    """

    def __init__(self, **keyArgs):
        object.__init__(self)
        for name, val in keyArgs.items():
            self.__safeAdd(name, val)

    def __safeAdd(self, name, val):
        """
	Add a field if it does not already exist and name does not 
	start with ``__`` (two underscores).

        Parameters
        ----------
        name : `str`
            Name of field to add.
        val : object
            Value of field to add.

        Raises
        ------
        RuntimeError
            Raised if name already exists or starts with ``__`` 
            (two underscores).
        """
        if hasattr(self, name):
            raise RuntimeError("Item %s already exists" % (name,))
        if name.startswith("__"):
            raise RuntimeError("Item name %r invalid; must not begin with __" % (name,))
        setattr(self, name, val)

    def getDict(self):
        """
        Get a dictionary of fields in this struct.

        Returns
        -------
        structDict : `dict`
            Dictionary with field names as keys and field values as values. 
            The values are shallow copies.
        """
        return self.__dict__.copy()

    def mergeItems(self, struct, *nameList):
        """
        Copy specified fields from another struct, provided they don't 
        already exist.

        Parameters
        ----------
        struct : `Struct`
            `Struct` from which to copy.
        *nameList : `str`
            All remaining arguments are names of items to copy.

        Raises
        ------
        RuntimeError
            Raised if any item in nameList already exists in self 
            (but any items before the conflicting item
            in nameList will have been copied).

        Examples
        --------
        For example::
            foo.copyItems(other, "itemName1", "itemName2")
        copies ``other.itemName1`` and ``other.itemName2`` into self.
        """
        for name in nameList:
            self.__safeAdd(name, getattr(struct, name))

    def copy(self):
        """Make a one-level-deep copy (values are not copied).
        Returns
        -------
        copy : `Struct`
            One-level-deep copy of this Struct.
        """
        return Struct(**self.getDict())

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return self.__dict__[key]

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __repr__(self):
        itemList = ["%s=%r" % (name, val) for name, val in self.getDict().items()]
        return "%s(%s)" % (self.__class__.__name__, "; ".join(itemList))
