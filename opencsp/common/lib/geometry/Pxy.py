import numpy as np
import numpy.typing as npt

from opencsp.common.lib.geometry.Vxy import Vxy


class Pxy(Vxy):
    def __init__(self, data, dtype=float):
        # Instantiate vector
        super().__init__(data, dtype)

    def __repr__(self):
        return '2D Point:\n' + self._data.__repr__()

    def distance(self, data_in: "Pxy") -> npt.NDArray[np.float_]:
        """Calculates the euclidian distance between this point and the data_in point."""
        self._check_is_Vxy(data_in)
        return (self - data_in).magnitude()

    def angle_from(self, origin: "Pxy") -> npt.NDArray[np.float_]:
        """
        Returns the rotation angle in which this point lies relative to the
        given origin point.
        """
        self._check_is_Vxy(origin)
        return (self - origin).angle()

    def as_Vxy(self):
        return Vxy(self._data, self.dtype)
