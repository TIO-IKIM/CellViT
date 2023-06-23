# -*- coding: utf-8 -*-
# Exceptions
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen


class WrongParameterException(Exception):
    """
    Exceptions that occur when the given parameters are not supported.
    """

    pass


class OverwriteException(WrongParameterException):
    """
    Exceptions that occur when the data may be overwritten but there is a missing parameter.
    """

    pass


class UnalignedDataException(Exception):
    """
    Exceptions that occur when the given data is not aligned.
    """

    pass
