import operator
from packaging.version import parse as parse_version
from PyQt6 import QtCore, QtGui, QtWidgets, sip, QtSvg
from PySide6 import QtCore, QtGui, QtWidgets, QtSvg, __version__
import shiboken6
from PyQt5 import QtCore, QtGui, QtWidgets, QtSvg
import sip
from PySide2 import QtCore, QtGui, QtWidgets, QtSvg, __version__
from PySide2 import shiboken2
import shiboken2

def _setup_pyqt5plus():
    global QtCore, QtGui, QtWidgets, QtSvg, __version__
    global _isdeleted, _to_int

    if QT_API == QT_API_PYQT6:
        from PyQt6 import QtCore, QtGui, QtWidgets, sip, QtSvg
        __version__ = QtCore.PYQT_VERSION_STR
        QtCore.Signal = QtCore.pyqtSignal
        QtCore.Slot = QtCore.pyqtSlot
        QtCore.Property = QtCore.pyqtProperty
        _isdeleted = sip.isdeleted
        _to_int = operator.attrgetter('value')
    elif QT_API == QT_API_PYSIDE6:
        from PySide6 import QtCore, QtGui, QtWidgets, QtSvg, __version__
        import shiboken6
        def _isdeleted(obj): return not shiboken6.isValid(obj)
        if parse_version(__version__) >= parse_version('6.4'):
            _to_int = operator.attrgetter('value')
        else:
            _to_int = int
    elif QT_API == QT_API_PYQT5:
        from PyQt5 import QtCore, QtGui, QtWidgets, QtSvg
        import sip
        __version__ = QtCore.PYQT_VERSION_STR
        QtCore.Signal = QtCore.pyqtSignal
        QtCore.Slot = QtCore.pyqtSlot
        QtCore.Property = QtCore.pyqtProperty
        _isdeleted = sip.isdeleted
        _to_int = int
    elif QT_API == QT_API_PYSIDE2:
        from PySide2 import QtCore, QtGui, QtWidgets, QtSvg, __version__
        try:
            from PySide2 import shiboken2
        except ImportError:
            import shiboken2
        def _isdeleted(obj):
            return not shiboken2.isValid(obj)
        _to_int = int
    else:
        raise AssertionError(f"Unexpected QT_API: {QT_API}")
