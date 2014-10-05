from PyQt4.QtCore import *
from PyQt4.QtGui import *

from mitsuba.core import Thread, Bitmap, Point2i, Vector2i, \
    Spectrum, InterpolatedSpectrum, Bitmap, FileStream, \
    ReconstructionFilter, Appender, Formatter, EDebug

from mitsuba.render import RenderListener

from util import toSRGB, fromSRGB

from time import time

import os, sys

Signal = pyqtSignal


class ChoiceWidget(QWidget):
    """
    Offers a choice between different (e.g. simple/advanced)
    GUI widgets via a '+' push button
    """
    def __init__(self, parent):
        QWidget.__init__(self, parent)
        self.widgets = None
        self.setMinimumWidth(200)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.index = 0

    def init(self, widgets, index = 0):
        self.widgets = widgets
        height = max([w.sizeHint().height() for w in widgets])
        self.setMinimumHeight(height)
        self.setMaximumHeight(height)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        for idx, w in enumerate(widgets):
            layout.addWidget(w, 0, Qt.AlignTop)
        layout.addSpacerItem(QSpacerItem(1, 1, QSizePolicy.Expanding))
        button = QPushButton('+', self)
        button.setFlat(True)
        button.setFocusPolicy(Qt.NoFocus)
        button.setMaximumWidth(20)
        button.setMaximumHeight(25)
        button.clicked.connect(self._toggled)
        layout.addWidget(button)
        self.set_selected(index)
        self.setLayout(layout)

    def set_selected(self, index):
        for i, w in enumerate(self.widgets):
            w.setVisible(i == index)
        self.index = index

    def _toggled(self):
        self.index = (self.index + 1) % len(self.widgets)
        self.set_selected(self.index)


class SlidingSpinner(QWidget):
    """
    Combined QSpinBox / QSlider widget
    """
    value_changed = Signal(float)

    def __init__(self, parent, minVal, maxVal, value):
        QWidget.__init__(self, parent)

        self.spinbox = QDoubleSpinBox(self)
        self.spinbox.setMinimum(minVal)
        self.spinbox.setValue(value)
        self.spinbox.setMaximum(maxVal)
        self.spinbox.setSingleStep((maxVal-minVal)/100.0)
        self.slider = QSlider(Qt.Horizontal, self)
        self.slider.setMinimum(minVal*100)
        self.slider.setMaximum(maxVal*100)
        self.slider.setValue(value*100)
        self.slider.setSingleStep((maxVal-minVal))
        self.slider.setPageStep((maxVal-minVal)*10.0)
        self.slider.setTickInterval((maxVal-minVal)*10.0)
        self.slider.setMinimumWidth(90)
        self.slider.valueChanged.connect(lambda val: self._update(val/100.0, True))
        self.spinbox.valueChanged.connect(lambda val: self._update(val, False))
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.slider)
        layout.addWidget(self.spinbox)
        self.setLayout(layout)

    def _update(self, value, from_slider):
        if from_slider:
            self.spinbox.setValue(value)
        else:
            self.slider.setValue(value*100)
        self.value_changed.emit(value)

    def set_value(self, value):
        state = self.slider.blockSignals(True)
        if type(value) == list:
            value = value[0]
        self.slider.setValue(value*100)
        self.slider.blockSignals(state)

    def get_value(self):
        return [self.slider.value()]*3


class FileWidget(QWidget):
    """
    Combined QLineEdit / QPushButton + QFileDialog widget
    """
    value_changed = Signal(str)

    def __init__(self, parent, value):
        QWidget.__init__(self, parent)

        self.lineedit = QLineEdit(self)
        self.lineedit.setText(value)
        self.button = QPushButton("..", self)

        def clicked(checked):
            try:
                fname = self.lineedit.text()
                if fname != "" and os.path.exists(fname):
                    dirname = os.path.dirname(fname)
                else:
                    dirname = ""

                fname = QFileDialog.getOpenFileName(self, 'Open file',
                                                    filter='Matusik BRDF file, *.binary',
                                                    directory = dirname)
                if fname == '':
                    return
                self.set_value(fname)
                self.value_changed.emit(fname)
            except:
                return

        def textChanged(text):
            self.value_changed.emit(text)

        self.lineedit.textChanged.connect(textChanged)
        self.button.clicked.connect(clicked)
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.lineedit)
        layout.addWidget(self.button)
        self.setLayout(layout)

    def set_value(self, value):
        state = self.lineedit.blockSignals(True)
        self.lineedit.setText(value)
        self.lineedit.blockSignals(state)


class AdvancedColorWidget(QWidget):
    value_changed = Signal(list)

    def __init__(self, parent, value, minValue = 0, maxValue = 1):
        QWidget.__init__(self, parent)
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)
        spinboxes = []
        if type(value) is not list:
            value = [value]*3

        for i in range(3):
            spinbox = QDoubleSpinBox(self)
            spinbox.setMinimum(minValue)
            spinbox.setMaximum(maxValue)
            spinbox.setValue(value[i])
            spinbox.setSingleStep(0.01)
            layout.addWidget(spinbox)
            spinboxes.append(spinbox)
            spinbox.valueChanged.connect(self._send_signal)

        self.setLayout(layout)

    def _send_signal(self):
        l = [self.children()[i].value() for i in range(3)]
        self.value_changed.emit(l)

    def set_value(self, value):
        for i in range(3):
            self.children()[i].setValue(value[i])

    def get_value(self):
        return [self.children()[i].value() for i in range(3)]


class SimpleColorWidget(QFrame):
    value_changed = Signal(list)

    def __init__(self, parent, value):
        QFrame.__init__(self, parent)
        if type(value) is not list:
           value = [value]*3
        self.setFrameStyle(QFrame.Sunken | QFrame.Box)
        self.setLineWidth(1)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.setAutoFillBackground(True)
        self.set_value(value)

    def sizeHint(self):
        return QSize(100, 30)

    def set_value(self, value):
        palette = self.palette()
        color = QColor(toSRGB(value[0]), toSRGB(value[1]), toSRGB(value[2]))
        palette.setColor(self.backgroundRole(), color)
        self.setPalette(palette)

    def get_value(self):
        color = self.palette().color(self.backgroundRole())
        return [fromSRGB(color.red()), fromSRGB(color.green()), fromSRGB(color.blue())]

    def mousePressEvent(self, evt):
        dialog = QColorDialog(self.palette().color(self.backgroundRole()), self)

        def callback(color):
            color = [fromSRGB(color.red()), fromSRGB(color.green()), fromSRGB(color.blue())]
            self.set_value(color)
            self.value_changed.emit(color)

        dialog.colorSelected.connect(callback)
        dialog.exec_()


class TextureWidget(QWidget):
    value_changed = Signal(list)

    def __init__(self, parent, value):
        QWidget.__init__(self, parent)

        label1 = QLabel('Texture &nbsp;<span style="font-size:16pt">[</span>', self)
        label2 = QLabel('samples <span style="font-size:16pt">]</span>', self)

        self.min_value = 0
        self.max_value = 1
        self.spinbox = QSpinBox(self)
        self.spinbox.setMinimum(2)
        self.spinbox.setValue(2)
        self.spinbox.setMaximum(10)
        self.spinbox.setSingleStep(1)
        self.spinbox.setMaximumWidth(50)
        if value is not None:
            self.set_value(value)
        self.spinbox.valueChanged.connect(self._send_signal)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(label1)
        layout.addWidget(self.spinbox)
        layout.addWidget(label2)
        layout.addSpacerItem(QSpacerItem(1, 1, QSizePolicy.Expanding))
        self.setLayout(layout)

    def _send_signal(self):
        self.value_changed.emit(self.get_value())

    def get_value(self):
        return [self.min_value, self.max_value, self.spinbox.value()]

    def set_value(self, value):
        self.min_value = value[0]
        self.max_value = value[1]
        self.spinbox.setValue(value[2])


class ColorWidget(ChoiceWidget):
    value_changed = Signal(list)

    def __init__(self, parent, value):
        ChoiceWidget.__init__(self, parent)
        widget1 = SimpleColorWidget(parent, value)
        widget2 = AdvancedColorWidget(parent, value)
        widget1.value_changed.connect(self._update)
        widget2.value_changed.connect(self._update)
        self.init([widget1, widget2])

    def _update(self, value):
        if self.sender() == self.widgets[1]:
            self.widgets[0].set_value(value)
        else:
            self.widgets[1].set_value(value)
        self.value_changed.emit(value)


class TexturableColorWidget(ChoiceWidget):
    value_changed = Signal(list, bool)

    def __init__(self, parent, value, value_tex):
        ChoiceWidget.__init__(self, parent)
        widget1 = SimpleColorWidget(parent, value)
        widget2 = AdvancedColorWidget(parent, value)
        widget3 = TextureWidget(parent, value_tex)
        widget1.value_changed.connect(self._update)
        widget2.value_changed.connect(self._update)
        widget3.value_changed.connect(self._update)
        self.init([widget1, widget2, widget3], 0 if value_tex is None else 2)

    def _update(self, value):
        if self.sender() == self.widgets[1]:
            self.widgets[0].set_value(value)
            self.value_changed.emit(value, False)
        elif self.sender() == self.widgets[0]:
            self.widgets[1].set_value(value)
            self.value_changed.emit(value, False)
        else:
            self.value_changed.emit(value, True)

    def _toggled(self):
        super(TexturableColorWidget, self)._toggled()
        self.value_changed.emit(self.widgets[self.index].get_value(), self.index == 2)


class OpticalThicknessWidget(ChoiceWidget):
    value_changed = Signal(list, bool)

    def __init__(self, parent, value, value_tex):
        ChoiceWidget.__init__(self, parent)

        if type(value) is not list:
            value = [value]*3

        widget1 = SlidingSpinner(self, 0, 5, value[0])
        widget2 = AdvancedColorWidget(parent, value, 0, 100)
        widget3 = TextureWidget(parent, value_tex if value_tex is not None else [0, 1.5, 1])
        widget1.value_changed.connect(self._update)
        widget2.value_changed.connect(self._update)
        widget3.value_changed.connect(self._update)

        if value_tex is None:
            if value[0] == value[1] and value[0] == value[2]:
                index = 0
            else:
                index = 1
        else:
            index = 2

        self.init([widget1, widget2, widget3], index)

    def _update(self, value):
        if type(value) is not list:
            value = [value]*3
        if self.sender() == self.widgets[1]:
            self.widgets[0].set_value(value)
            self.value_changed.emit(value, False)
        elif self.sender() == self.widgets[0]:
            self.widgets[1].set_value(value)
            self.value_changed.emit(value, False)
        else:
            self.value_changed.emit(value, True)

    def _toggled(self):
        super(OpticalThicknessWidget, self)._toggled()
        self.value_changed.emit(self.widgets[self.index].get_value(), self.index == 2)


class ConductorIORComboBox(QComboBox):
    value_changed = Signal(str, list, list)

    def __init__(self, parent, value):
        QComboBox.__init__(self, parent)
        self.addItem("None", "none")
        self.addItem("Aluminium", "Al")
        self.addItem("Chrome", "Cr")
        self.addItem("Copper", "Cu")
        self.addItem("Gold", "Au")
        self.addItem("Lithium", "Li")
        self.addItem("Mercury", "Hg")
        self.addItem("Nickel", "Ni_palik")
        self.addItem("Silver", "Ag")
        self.addItem("Tungsten", "W")
        self.setMaximumWidth(160)
        self.set_value(value)

        def activated(index):
            preset = self.itemData(index)
            if type(preset) is QVariant and preset.toString is not None:
                preset = preset.toString()
            preset = str(preset)
            if preset != "none":
                fResolver = Thread.getThread().getFileResolver()
                eta = Spectrum()
                eta.fromContinuousSpectrum(InterpolatedSpectrum(
                    fResolver.resolve("data/ior/" + preset + ".eta.spd")))
                k = Spectrum()
                k.fromContinuousSpectrum(InterpolatedSpectrum(
                    fResolver.resolve("data/ior/" + preset + ".k.spd")))
                eta = [eta[i] for i in range(3)]
                k = [k[i] for i in range(3)]
            else:
                eta = [0]*3
                k = [1]*3
            self.value_changed.emit(preset, eta, k)

        self.activated.connect(activated)

    def set_value(self, value):
        for i in range(self.count()):
            if value == str(self.itemData(i)):
                self.setCurrentIndex(i)
                return
        self.setCurrentIndex(self.count()-1)

class PhaseFunctionTypeComboBox(QComboBox):
    value_changed = Signal(str)

    def __init__(self, parent, value):
        QComboBox.__init__(self, parent)
        self.addItem("Henyey-Greenstein", "hg")
        self.addItem("von Mises Fisher", "vmf")
        self.set_value(value)

        def activated(index):
            data = self.itemData(index)
            if type(data) is QVariant and data.toString is not None:
                data = data.toString()
            if data is not None:
                self.value_changed.emit(str(data))

        self.activated.connect(activated)

    def set_value(self, value):
        for i in range(self.count()):
            if value == self.itemData(i):
                self.setCurrentIndex(i)
                return
        self.setCurrentIndex(self.count()-1)

class DielectricIORComboBox(QComboBox):
    value_changed = Signal(float)

    def __init__(self, parent, value):
        QComboBox.__init__(self, parent)
        self.addItem("Vacuum", 1.0)
        self.addItem("Air", 1.00028)
        self.addItem("Water ice", 1.31)
        self.addItem("Fused quartz", 1.458)
        self.addItem("Pyrex", 1.470)
        self.addItem("Acrylic glass", 1.49)
        self.addItem("Polypropylene", 1.4901)
        self.addItem("Water", 1.3330)
        self.addItem("Acetone", 1.36)
        self.addItem("Ethanol", 1.361)
        self.addItem("Carbon tetrachloride", 1.461)
        self.addItem("Glycerol", 1.4729)
        self.addItem("Benzene", 1.501)
        self.addItem("BK7 Glass", 1.5046)
        self.addItem("Sodium chloride", 1.544)
        self.addItem("Amber", 1.55)
        self.addItem("PET", 1.575)
        self.addItem("Silicone oil", 1.52045)
#       self.addItem("Diamond", 2.419)
        self.addItem("< manual >", None)
        self.setMaximumWidth(160)
        self.set_value(value)

        def activated(index):
            data = self.itemData(index)
            if data is not None:
                self.value_changed.emit(data)

        self.activated.connect(activated)

    def set_value(self, value):
        for i in range(self.count()):
            if value == self.itemData(i):
                self.setCurrentIndex(i)
                return
        self.setCurrentIndex(self.count()-1)

    def is_preset(self):
        return self.currentIndex() != self.count()-1


class AdvancedAnisotropyWidget(QWidget):
    value_changed = Signal(list)

    def __init__(self, parent, value):
        QWidget.__init__(self, parent)

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)
        spinboxes = []
        if type(value) is not list:
            value = [value]*3

        for i in range(3):
            spinbox = QDoubleSpinBox(self)
            spinbox.setMinimum(-0.99)
            spinbox.setMaximum(0.99)
            spinbox.setValue(value[i])
            spinbox.setSingleStep(0.01)
            layout.addWidget(spinbox)
            spinboxes.append(spinbox)
            spinbox.valueChanged.connect(self._send_signal)

        self.setLayout(layout)

    def _send_signal(self):
        l = [self.children()[i].value() for i in range(3)]
        self.value_changed.emit(l)

    def set_value(self, value):
        for i in range(3):
            self.children()[i].setValue(value[i])

    def get_value(self):
        return [self.children()[i].value() for i in range(3)]

class AnisotropyWidget(ChoiceWidget):
    value_changed = Signal(list, bool)

    def __init__(self, parent, value):
        ChoiceWidget.__init__(self, parent)

        if type(value) is not list:
            value = [value]*3

        widget1 = SlidingSpinner(self, -0.99, 0.99, value[0])
        widget2 = AdvancedAnisotropyWidget(parent, value)
        widget1.value_changed.connect(self._update)
        widget2.value_changed.connect(self._update)

        if value[0] == value[1] and value[0] == value[2]:
            index = 0
        else:
            index = 1

        self.init([widget1, widget2], index)

    def _update(self, value):
        if type(value) is not list:
            value = [value]*3
        if self.sender() == self.widgets[1]:
            self.widgets[0].set_value(value)
            self.value_changed.emit(value, False)
        elif self.sender() == self.widgets[0]:
            self.widgets[1].set_value(value)
            self.value_changed.emit(value, False)
        else:
            self.value_changed.emit(value, True)


class IORWidget(ChoiceWidget):
    value_changed = Signal(float)

    def __init__(self, parent, value):
        ChoiceWidget.__init__(self, parent)
        self.widget1 = DielectricIORComboBox(parent, value)
        self.widget2 = SlidingSpinner(self, 1, 2.5, value)
        self.widget1.value_changed.connect(self._update)
        self.widget2.value_changed.connect(self._update)
        self.init([self.widget1, self.widget2], 0 if self.widget1.is_preset() else 1)

    def _update(self, value):
        if self.sender() == self.widget2:
            self.widget1.set_value(value)
        else:
            self.widget2.set_value(value)
        self.value_changed.emit(value)


class MitsubaRenderBuffer(RenderListener):
    """
    Implements the Mitsuba callback interface to capture notifications about
    rendering progress. Partially completed image blocks are efficiently
    tonemapped into a local 8-bit Mitsuba Bitmap instance and exposed as a QImage.
    """
    RENDERING_FINISHED  = 0
    RENDERING_CANCELLED = 1
    RENDERING_UPDATED   = 2

    def __init__(self, queue, callback):
        super(MitsubaRenderBuffer, self).__init__()
        self.bitmap = self.qimage = None
        self.backup = self.qimage_backup = None
        self.callback = callback
        self.time = 0
        self.size = Vector2i(0, 0)
        self.busy = False
        queue.registerListener(self)

    def workBeginEvent(self, job, wu, thr):
        """ Callback: a worker thread started rendering an image block.
            Draw a rectangle to highlight this """
        _ = self._get_film_ensure_initialized(job)
        if not self.busy:
            self.backup.copyFrom(self.bitmap)
            self.busy = True
        self.backup.copyFrom(self.bitmap, wu.getOffset(), wu.getOffset(), wu.getSize())
        self.bitmap.drawWorkUnit(wu.getOffset(), wu.getSize(), thr)
        self._potentially_send_update()

    def workEndEvent(self, job, wr, cancelled):
        """ Callback: a worker thread finished rendering an image block.
            Tonemap the associated pixels and store them in 'self.bitmap' """
        film = self._get_film_ensure_initialized(job)
        if not cancelled:
            film.develop(wr.getOffset(), wr.getSize(), wr.getOffset(), self.bitmap)
        else:
            self.bitmap.copyFrom(self.backup, wr.getOffset(), wr.getOffset(), wr.getSize())
        self._potentially_send_update()

    def refreshEvent(self, job):
        """ Callback: the entire image changed (some rendering techniques
            do this occasionally). Hence, tonemap the full film. """
        film = self._get_film_ensure_initialized(job)
        film.develop(Point2i(0), self.size, Point2i(0), self.bitmap)
        self._potentially_send_update(force=True)

    def finishJobEvent(self, job, cancelled):
        """ Callback: the rendering job has finished or was cancelled.
            Re-develop the image once more for the final result. """
        film = self._get_film_ensure_initialized(job)
        if not cancelled:
            film.develop(Point2i(0), self.size, Point2i(0), self.bitmap)
        self.busy = False
        self.callback(MitsubaRenderBuffer.RENDERING_CANCELLED if cancelled
                      else MitsubaRenderBuffer.RENDERING_FINISHED)

    def _get_film_ensure_initialized(self, job):
        """ Ensure that all internal data structure are set up to deal
            with the given rendering job """
        film = job.getScene().getFilm()
        size = film.getSize()

        if self.size != size:
            self.size = size

            # Round the buffer size to the next power of 4 to ensure 32-bit
            # aligned scanlines in the underlying buffer. This is needed so
            # that QtGui.QImage and mitsuba.Bitmap have exactly the same
            # in-memory representation.
            bufsize = Vector2i((size.x + 3) // 4 * 4, (size.y + 3) // 4 * 4)

            # Create an 8-bit Mitsuba bitmap that will store tonemapped pixels
            self.bitmap = Bitmap(Bitmap.ERGB, Bitmap.EUInt8, bufsize)
            self.backup = Bitmap(Bitmap.ERGB, Bitmap.EUInt8, bufsize)
            self.bitmap.clear()

            # Create a QImage that is backed by the Mitsuba Bitmap instance
            # (i.e. without doing unnecessary bitmap copy operations)
            self.qimage = QImage(self.bitmap.buffer(), self.size.x,
                                 self.size.y, QImage.Format_RGB888)
            self.qimage_backup = QImage(self.backup.buffer(), self.size.x,
                                        self.size.y, QImage.Format_RGB888)
        return film

    def _potentially_send_update(self, force=False):
        """ Send an update request to any attached widgets, but not too often """
        now = time()
        if now - self.time > .25 or force:
            self.time = now
            self.callback(MitsubaRenderBuffer.RENDERING_UPDATED)

    def set_bitmap(self, bitmap):
        self.size = bitmap.getSize()
        self.bitmap = bitmap
        self.backup = bitmap.clone()
        self.qimage = QImage(self.bitmap.buffer(), self.size.x,
                             self.size.y, QImage.Format_RGB888)
        self.qimage_backup = QImage(self.backup.buffer(), self.size.x,
                                    self.size.y, QImage.Format_RGB888)

class RenderWidget(QWidget):
    """ This simple widget attaches itself to a Mitsuba RenderQueue instance
        and displays the progress of everything that's being rendered """
    rendering_updated = Signal(int)
    pixel_clicked = Signal(int, int)

    def __init__(self, parent, queue, ref_bitmap):
        QWidget.__init__(self, parent)
        self.buffer = MitsubaRenderBuffer(queue, self.rendering_updated.emit)
        # Need a queued conn. to avoid threading issues between Qt and Mitsuba
        self.rendering_updated.connect(self._handle_update, Qt.QueuedConnection)
        self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        self.show_last = False
        if ref_bitmap is not None:
            size = ref_bitmap.getSize()
            new_size = Vector2i((size.x + 3) // 4 * 4, (size.y + 3) // 4 * 4)

            if size != new_size:
                ref_bitmap = ref_bitmap.convert(Bitmap.ERGB, Bitmap.EFloat32, -1)
                ref_bitmap = ref_bitmap.resample(None,
                    ReconstructionFilter.EClamp,
                    ReconstructionFilter.EClamp, new_size, -float('inf'), float('inf')
                )

            self.ref_bitmap = ref_bitmap.convert(Bitmap.ERGB, Bitmap.EUInt8, -1)
            self.ref_qimage = QImage(self.ref_bitmap.buffer(),
                    ref_bitmap.getWidth(), ref_bitmap.getHeight(), QImage.Format_RGB888)
        else:
            self.ref_bitmap = None
            self.ref_qimage = None

    def swap(self):
        self.show_last = not self.show_last
        self.repaint()

    def sizeHint(self):
        return QSize(self.buffer.size.x, self.buffer.size.y)

    def get_bitmap(self):
        return self.buffer.bitmap

    def maybe_load_preview(self, filename):
        try:
            fs = FileStream(filename, FileStream.EReadOnly)
            self.buffer.set_bitmap(Bitmap(Bitmap.EAuto, fs))
            fs.close()
        except:
            pass

    def _handle_update(self, event):
        image = self.buffer.qimage
        # Detect when an image of different resolution is being rendered
        if image.width() < self.width() or image.height() < self.height():
            self.updateGeometry()
        self.repaint()

    def mousePressEvent(self, event):
        image = self.buffer.qimage if not self.show_last else self.buffer.qimage_backup
        if not image:
            returnG
        offset = QPoint((self.width() - image.width()) / 2,
                        (self.height() - image.height()) / 2)
        x, y = event.x()-offset.x(), event.y()-offset.y()
        if 0 <= x < image.width() and 0 <= y < image.height():
            self.pixel_clicked.emit(x, y)

    def paintEvent(self, event):
        """ When there is more space then necessary, display the image centered
            on a black background, surrounded by a light gray border """
        QWidget.paintEvent(self, event)
        qp = QPainter(self)
        qp.setPen(Qt.lightGray)
        qp.fillRect(self.rect(), Qt.black)

        image = self.buffer.qimage if not self.show_last else \
            (self.buffer.qimage_backup if self.ref_qimage is None else self.ref_qimage)

        if image is not None:
            offset = QPoint((self.width() - image.width()) / 2,
                            (self.height() - image.height()) / 2)
            qp.drawRect(QRect(offset - QPoint(1, 1), image.size() + QSize(1, 1)))
            qp.drawImage(offset, image)
        if self.show_last:
            qp.drawText(QPoint(15, 27), "< Previous rendering >" \
                if self.ref_bitmap is None else " < Reference image >")
        qp.end()

class LogWidget(QTextEdit):
    message_reported = Signal(int, str)

    def __init__(self, parent):
        QTextEdit.__init__(self, parent)

        font = QFont("Monospace", 12 if 'darwin' in sys.platform else 8)
        font.setStyleHint(QFont.TypeWriter)
        self.setFont(font)
        self.setReadOnly(True)
        palette = QPalette()
        palette.setColor(QPalette.Base, Qt.black)
        self.setPalette(palette)
        self.setMaximumHeight(150)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        class MyAppender(Appender):
            def append(self2, log_level, message):
                self.message_reported.emit(message)

        #logger = Thread.getThread().getLogger()
        #logger.clearAppenders()
        #logger.addAppender(MyAppender())
        #logger.setLogLevel(EDebug)
        self.message_reported.connect(self.log)

    def log(self, message):
        self.setTextColor(Qt.lightGray)
        self.append(message)
        c = self.textCursor()
        c.movePosition(QTextCursor.End)
        self.setTextCursor(c)
        self.ensureCursorVisible()
        qApp.processEvents()

