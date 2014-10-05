from PyQt4.QtCore import *
from PyQt4.QtGui import *
from PyQt4.QtSvg import *

from widgets import (SlidingSpinner, ColorWidget, IORWidget, FileWidget,
                     OpticalThicknessWidget, ConductorIORComboBox,
                     TexturableColorWidget, AnisotropyWidget, PhaseFunctionTypeComboBox)

from material import LayeredMaterial

Signal = pyqtSignal


def make_setter(target, key):
    def setter(value, texture=False):
        if not texture:
            target[key] = value
            if key+'_tex' in target:
                del target[key+'_tex']
        else:
            target[key+'_tex'] = value
    return setter


class LayerView(QFrame):
    # Signals
    layer_selected     = Signal(int)
    interface_selected = Signal(int)

    # Various constants
    LAYER_HEIGHT      = 50
    DEFAULT_COLOR     = QColor(180, 180, 180)
    SELECTED_COLOR    = QColor(180, 180, 250)
    SELECTED2_COLOR   = QColor(100, 100, 255)
    BLACK_COLOR       = QColor(0, 0, 0)
    TEXT_COLOR        = QColor(70,70,70)
    BORDER_OFFSET     = 8
    LAYER_TYPE_MAP    = {
        LayeredMaterial.DIFFUSE_LAYER :   'Diffuse',
        LayeredMaterial.CONDUCTOR_LAYER:  'Conductor',
        LayeredMaterial.MEASURED_LAYER:   'Measured',
        LayeredMaterial.DIELECTRIC_LAYER: 'Dielectric'
    }

    def __init__(self, parent, datamodel):
        QFrame.__init__(self, parent)
        self.setMinimumSize(200, 100)
        self.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        self.setLineWidth(2)
        self.datamodel = datamodel
        self.poly_layer = []
        self.poly_interface = []
        self.update()

    def resizeEvent(self, event):
        self.update()

    def export_svg(self, filename):
        generator = QSvgGenerator()
        generator.setFileName(filename)
        generator.setSize(self.size())
        generator.setViewBox(QRect(0, 0, self.width(), self.height()))
        generator.setTitle("Layer structure")
        qp = QPainter()
        qp.begin(generator)
        self._do_paint(qp)
        qp.end()

    def mousePressEvent(self, event):
        mat = self.datamodel.selected_material
        if not (event.buttons() & Qt.LeftButton) or mat is None:
            return

        def dist2(v, w):
            return float((v.x()-w.x())**2 + (v.y()-w.y())**2)

        def dist2_line_point(v, w, p):
            l2 = dist2(v, w)
            if l2 == 0:
                return dist2(p, v)
            t = ((p.x() - v.x()) * (w.x() - v.x()) +
                 (p.y() - v.y()) * (w.y() - v.y())) / l2
            if t < 0:
                return dist2(p, v)
            if t > 1:
                return dist2(p, w)
            return dist2(p, QPointF(v.x() + t * (w.x() - v.x()),
                    v.y() + t * (w.y() - v.y())))

        for index, poly in enumerate(self.poly_interface):
            for i in range(0, poly.size()):
                idx0, idx1 = i, (i+1) % poly.size()
                if dist2_line_point(poly[idx0], poly[idx1], event.pos()) < 4*4:
                    mat.selected_index = index
                    mat.selected_layer = False
                    self.interface_selected.emit(index)
                    self.repaint()
                    return

        for index, poly in enumerate(self.poly_layer):
            if poly.containsPoint(event.pos(), Qt.OddEvenFill):
                mat.selected_index = index
                mat.selected_layer = True
                self.layer_selected.emit(index)
                self.repaint()

    def update(self):
        mat = self.datamodel.selected_material
        if mat is None:
            self.repaint()
            return

        def make_poly(interface, yoffset, reverse=False):
            qsrand(id(interface) % (4096*1024))
            poly = []
            roughness = interface['roughness'] if 'roughness' in interface else LayeredMaterial.ROUGHNESS_DEFAULT
            if not interface['enabled']:
                roughness = 0
            for i in range(0, 51):
                poly.append(QPoint(LayerView.BORDER_OFFSET + (self.width()-2 *
                                                              LayerView.BORDER_OFFSET)*i/50.0,
                                   yoffset+((qrand() % 11)-5) * roughness**0.4))
            if reverse:
                poly.reverse()
            return poly

        del self.poly_layer[:]
        del self.poly_interface[:]

        ypos = (self.height() - len(mat.layers) * LayerView.LAYER_HEIGHT)/2
        for index, layer in enumerate(mat.layers):
            top = make_poly(mat.interfaces[index], ypos)
            bot = make_poly(mat.interfaces[index+1], ypos+LayerView.LAYER_HEIGHT, True)
            self.poly_layer.append(QPolygon(top+bot))
            self.poly_interface.append(QPolygon(top))
            if index == len(mat.layers)-1:
                self.poly_interface.append(QPolygon(bot))
            ypos += LayerView.LAYER_HEIGHT
        self.repaint()

    def paintEvent(self, event):
        QFrame.paintEvent(self, event)
        qp = QPainter(self)
        self._do_paint(qp)
        qp.end()

    def _do_paint(self, qp):
        mat = self.datamodel.selected_material
        if mat is None:
            return
        layers = mat.layers
        font = QFont("Gill Sans")
        font.setPixelSize(25)
        qp.setFont(font)

        ypos = (self.height() - len(mat.layers) * LayerView.LAYER_HEIGHT)/2
        for index, layer in enumerate(layers):
            fill = Qt.SolidPattern if mat.layers[index]['type'] == LayeredMaterial.DIELECTRIC_LAYER \
                else Qt.Dense2Pattern

            if mat.selected_layer and \
               mat.selected_index == index:
                color = LayerView.SELECTED_COLOR
            else:
                color = LayerView.DEFAULT_COLOR

            enabled = layer['enabled'] if 'enabled' in layer else True
            if not enabled:
                h, s, v, a = color.getHsvF()
                color = QColor.fromHsvF(h, s/2, min(1, v*1.2))
                fill = Qt.Dense3Pattern

            qp.setBrush(QBrush(color, fill))
            qp.drawPolygon(self.poly_layer[index])

            qp.setPen(LayerView.TEXT_COLOR)
            rect = QRect(LayerView.BORDER_OFFSET, ypos, self.width()-2*LayerView.BORDER_OFFSET, LayerView.LAYER_HEIGHT)
            qp.drawText(rect, Qt.AlignCenter | Qt.AlignVCenter, LayerView.LAYER_TYPE_MAP[mat.layers[index]['type']])
            ypos += LayerView.LAYER_HEIGHT

        for i in range(len(mat.interfaces)):
            enabled = mat.interfaces[i]['enabled'] if 'enabled' in mat.interfaces[i] else True
            if not enabled:
                qp.setPen(QPen(LayerView.DEFAULT_COLOR, 1, Qt.DotLine))
                qp.drawPolyline(self.poly_interface[i])

        if not mat.selected_layer:
            sel_itf = mat.interfaces[mat.selected_index]
            enabled = sel_itf['enabled'] if 'enabled' in sel_itf else True

            qp.setPen(QPen(LayerView.SELECTED2_COLOR, 5))
            qp.drawPolyline(self.poly_interface[mat.selected_index])
            if enabled:
                qp.setPen(QPen(LayerView.BLACK_COLOR, 1))
            else:
                qp.setPen(QPen(LayerView.BLACK_COLOR, 1, Qt.DotLine))
            qp.drawPolyline(self.poly_interface[mat.selected_index])


class LayerEditor(QWidget):
    # Signals
    structure_changed = Signal()

    def __init__(self, parent, datamodel):
        QWidget.__init__(self, parent)
        self.datamodel = datamodel
        self.layout = QFormLayout()
        self.layout.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.layout.setLabelAlignment(Qt.AlignRight | Qt.AlignTop)
        self.setLayout(self.layout)
        self.layout.setSpacing(8)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.layout.setContentsMargins(10, 10, 10, 0)

        self.titlefont = QFont()
        self.titlefont.setBold(True)
        self.titlefont.setPointSize(15)

        def type_updated(value):
            mat = self.datamodel.selected_material
            index = mat.selected_index
            layer = mat.layers[index]
            layer['type'] = value
            mat.init_interface_status(index)
            mat.init_interface_status(index+1)
            self.structure_changed.emit()
            self.layer_selected(index)

        self.enabled_label = QLabel("Enabled :", self)
        self.enabled_label.setMinimumWidth(185)
        self.enabled_label.setAlignment(Qt.AlignRight | Qt.AlignTop)
        self.enabled_label.hide()
        self.type_widget = QComboBox(self)
        self.type_widget.addItem('Dielectric')
        self.type_widget.addItem('Conductor')
        self.type_widget.addItem('Diffuse')
        self.type_widget.addItem('Measured')
        self.type_widget.currentIndexChanged.connect(type_updated)
        self.type_widget.hide()
        self.layer_selected(None)

    def _clear_layout(self, layout):
        if layout is not None:
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    if widget != self.type_widget and widget != self.enabled_label:
                        widget.deleteLater()
                else:
                    self.clearLayout(item.layout())

    def _clear(self, title, addSeparator = True):
        self._clear_layout(self.layout)
        titlewidget = QLabel(title, self)
        titlewidget.setAlignment(Qt.AlignHCenter)
        titlewidget.setFont(self.titlefont)
        self.layout.addRow(titlewidget)
        if addSeparator:
            separator = QFrame(self)
            separator.setFrameStyle(QFrame.HLine | QFrame.Sunken)
            separator.setMinimumSize(10, 10)
            self.layout.addRow(separator)

    def layer_selected(self, index):
        if index == None:
            self.type_widget.hide()
            self.enabled_label.hide()
            self._clear('Click on an object to begin.', False)
            self.repaint()
            return
        self._clear('Layer properties')
        mat = self.datamodel.selected_material
        layer = mat.layers[index]

        def enabled_updated(checked):
            layer['enabled'] = checked
            self.structure_changed.emit()

        self.type_widget.blockSignals(True)
        self.type_widget.setCurrentIndex(layer['type'])
        self.type_widget.blockSignals(False)
        self.type_widget.show()
        self.enabled_label.show()
        type_label = QLabel("Type :", self)
        type_label.setMinimumHeight(26)
        self.layout.addRow(type_label, self.type_widget)

        enabled = layer['enabled'] if 'enabled' in layer else True
        enabled_widget = QCheckBox(self)
        enabled_widget.setChecked(enabled)
        enabled_widget.toggled.connect(enabled_updated)
        self.layout.addRow(self.enabled_label, enabled_widget)

        if layer['type'] == LayeredMaterial.DIELECTRIC_LAYER:
            eta = layer['eta'] if 'eta' in layer else LayeredMaterial.DIELECTRIC_ETA_DEFAULT
            sigma_t = layer['sigma_t'] if 'sigma_t' in layer else [LayeredMaterial.SIGMA_T_DEFAULT]*3
            sigma_t_tex = layer['sigma_t_tex'] if 'sigma_t_tex' in layer else None
            albedo = layer['albedo'] if 'albedo' in layer else [LayeredMaterial.ALBEDO_DEFAULT]*3
            albedo_tex = layer['albedo_tex'] if 'albedo_tex' in layer else None
            g = layer['g'] if 'g' in layer else LayeredMaterial.G_DEFAULT
            sigma_t_scale = layer['sigma_t_scale'] if 'sigma_t_scale' in layer else LayeredMaterial.SIGMA_T_SCALE_DEFAULT
            phase_type = layer['phase_type'] if 'phase_type' in layer else LayeredMaterial.PHASE_TYPE_DEFAULT

            eta_widget = IORWidget(self, eta)
            eta_widget.value_changed.connect(make_setter(layer, 'eta'))
            self.layout.addRow(u'&Index of refraction (<em>\u03b7</em>) :', eta_widget)

            phase_type_widget = PhaseFunctionTypeComboBox(self, phase_type)
            self.layout.addRow(u'Phase function type :', phase_type_widget)
            phase_type_widget.value_changed.connect(make_setter(layer, 'phase_type'))
            sigma_t_widget = OpticalThicknessWidget(self, sigma_t, sigma_t_tex)
            self.layout.addRow(u'Optical thickness (<em>\u03c3<sub>t</sub></em>) :', sigma_t_widget)
            sigma_t_widget.value_changed.connect(make_setter(layer, 'sigma_t'))

            if type(sigma_t) is list and (sigma_t[0] != sigma_t[1] or sigma_t[0] != sigma_t[2]):
                sigma_t_scale_widget = SlidingSpinner(self, 0, 5, sigma_t_scale)
                self.layout.addRow(u'Optical thickness scale factor :', sigma_t_scale_widget)
                sigma_t_scale_widget.value_changed.connect(make_setter(layer, 'sigma_t_scale'))

            albedo_widget = TexturableColorWidget(self, albedo, albedo_tex)
            self.layout.addRow(u'Single scattering albedo (<em>\u03b1</em>) :', albedo_widget)
            albedo_widget.value_changed.connect(make_setter(layer, 'albedo'))


            g_widget = AnisotropyWidget(self, g)
            self.layout.addRow(u'Scattering anisotropy (<em>g</em>) :', g_widget)
            g_widget.value_changed.connect(make_setter(layer, 'g'))

        elif layer['type'] == LayeredMaterial.CONDUCTOR_LAYER:
            def conductor_ior_preset_updated(preset, eta, k):
                layer['conductor_ior_preset'] = str(preset)
                layer['eta_cond'] = eta
                layer['k_cond'] = k
            conductor_ior_preset = layer['conductor_ior_preset'] if 'conductor_ior_preset' in layer else "none"
            conductor_ior_preset_widget = ConductorIORComboBox(self, conductor_ior_preset)
            conductor_ior_preset_widget.value_changed.connect(conductor_ior_preset_updated)
            self.layout.addRow(u'&Material preset (<em>\u03b7</em>, <em>k</em>) :', conductor_ior_preset_widget)

        elif layer['type'] == LayeredMaterial.MEASURED_LAYER:
            filename=layer['filename'] if 'filename' in layer else ""
            filename_widget = FileWidget(self, filename)
            filename_widget.value_changed.connect(make_setter(layer, 'filename'))
            self.layout.addRow(u'Filename :', filename_widget)

        elif layer['type'] == LayeredMaterial.DIFFUSE_LAYER:
            diffuse_albedo = layer['diffuse_albedo'] if 'diffuse_albedo' in layer else [.5]*3
            diffuse_albedo_tex = layer['diffuse_albedo_tex'] if 'diffuse_albedo_tex' in layer else None
            diffuse_albedo_widget = TexturableColorWidget(self, diffuse_albedo, diffuse_albedo_tex)
            self.layout.addRow(u'Diffuse albedo :', diffuse_albedo_widget)
            diffuse_albedo_widget.value_changed.connect(make_setter(layer, 'diffuse_albedo'))

    def interface_selected(self, index):
        self._clear('Interface properties')
        mat = self.datamodel.selected_material
        interface = mat.interfaces[index]

        def roughness_updated(value):
            interface['roughness'] = value
            self.structure_changed.emit()

        def enabled_updated(checked):
            interface['enabled'] = checked
            self.structure_changed.emit()

        self.type_widget.hide()
        enabled_widget = QCheckBox(self)
        enabled_widget.setChecked(interface['enabled'])
        enabled_widget.toggled.connect(enabled_updated)
        self.layout.addRow(self.enabled_label, enabled_widget)

        conserve = interface['conserve'] if 'conserve' in interface else LayeredMaterial.CONSERVE_DEFAULT
        conserve_widget = QCheckBox(self)
        conserve_widget.setChecked(conserve)
        conserve_widget.toggled.connect(make_setter(interface, 'conserve'))
        self.layout.addRow('Conserve energy :', conserve_widget)

        roughness = interface['roughness'] if 'roughness' in interface else LayeredMaterial.ROUGHNESS_DEFAULT
        roughness_widget = SlidingSpinner(self, 0, 1, roughness)
        roughness_widget.value_changed.connect(roughness_updated)
        self.layout.addRow(u'&Roughness (\u03b1) :', roughness_widget)
