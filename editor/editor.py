import sys, os, json

from PyQt4.QtCore import *
from PyQt4.QtGui import *

from mitsuba.core import Scheduler, Thread, Point2, PluginManager, Properties, \
    Bitmap, FileStream

from mitsuba.render import RenderQueue, RenderJob, Scene, SceneHandler

from layer import LayerView, LayerEditor
from widgets import RenderWidget, MitsubaRenderBuffer, LogWidget
from material import DataModel
import material

def get_shapes(shapes):
    result = set()
    for shape in shapes:
        sg = shape.getShapeGroup()
        if sg is None:
            result.add(shape)
        else:
            result |= get_shapes(sg.getKDTree().getShapes())
    return result

class MaterialEditor(QMainWindow):
    def __init__(self, scene_filename, reference_filename, parent=None):
        QMainWindow.__init__(self, parent)
        self.scene = None
        self.datamodel = DataModel()
        self.fresolver = Thread.getThread().getFileResolver()
        self.scene_filename = scene_filename
        self.scene_preview = os.path.splitext(scene_filename)[0]+".png"
        self.scene = SceneHandler.loadScene(self.fresolver.resolve(scene_filename))
        self.scene.initialize()
        self.spp = 256

        if reference_filename is not None:
            self.reference_bitmap = Bitmap(Bitmap.EAuto,
                FileStream(self.fresolver.resolve(reference_filename), FileStream.EReadOnly))
        else:
            self.reference_bitmap = None

        for shape in get_shapes(self.scene.getShapes()):
            matname = shape.getBSDF().getProperties().getID()
            if self.datamodel.try_import_material(matname):
                print('Successfully imported material \"%s\" from previous run.' % matname)

        self.add_action = QAction(QIcon(':/resources/add.png'), 'Add', self)
        self.add_action.triggered.connect(self._add_layer)
        self.remove_action = QAction(QIcon(':/resources/remove.png'), 'Remove', self)
        self.remove_action.triggered.connect(self._remove_layer)
        self.open_action = QAction(QIcon(':/resources/open.png'), 'Open', self)
        self.open_action.setShortcuts(QKeySequence.Open)
        self.open_action.setShortcutContext(Qt.ApplicationShortcut)
        self.open_action.triggered.connect(self._open)
        self.saveas_action = QAction(QIcon(':/resources/saveas.png'), 'Save as..', self)
        self.saveas_action.setShortcuts(QKeySequence.SaveAs)
        self.saveas_action.setShortcutContext(Qt.ApplicationShortcut)
        self.saveas_action.triggered.connect(self._saveas)

        icon = QIcon(':/resources/bookmark.png')
        preset_menu = QMenu()
        preset_button = QToolButton()
        preset_button.setMenu(preset_menu)
        preset_button.setPopupMode(QToolButton.InstantPopup)
        preset_button.setIcon(icon)

        presets_dir = os.path.join(os.path.dirname(__file__), 'presets')
        try:
            presets = [ f for f in os.listdir(presets_dir) if os.path.isfile(os.path.join(presets_dir, f)) ]
            for preset in presets:
                action = QAction("Preset \"%s\"" % preset, self)
                action.setData(os.path.join(presets_dir, preset))
                action.triggered.connect(self._load_preset)
                preset_menu.addAction(action)
        except:
            pass

        self.preset_action = QWidgetAction(self)
        self.preset_action.setDefaultWidget(preset_button)

        self.reset_action = QAction('Reset state', self)
        self.reset_action.triggered.connect(self._reset)

        self.save_action = QAction(QIcon(':/resources/save.png'), 'Save', self)
        self.save_action.setShortcutContext(Qt.ApplicationShortcut)
        self.save_action.setShortcuts(QKeySequence.Save)
        self.save_action.triggered.connect(self._save)
        self.exportimg_action = QAction('Export rendered image ..', self)
        self.exportimg_action.triggered.connect(self._export_image)
        self.exportsvg_action = QAction('Export structure as SVG ..', self)
        self.exportsvg_action.triggered.connect(self._export_svg)
        self.exportscrshot_action = QAction('Export screenshot ..', self)
        self.exportscrshot_action.triggered.connect(self._export_scrshot)
        self.render_action = QAction(QIcon(':/resources/go.png'), 'Go', self)
        self.render_action.setShortcut('Space')
        self.render_action.setShortcutContext(Qt.ApplicationShortcut)
        self.render_action.triggered.connect(self.render)

        self.quit_action = QAction("Quit", self)
        self.quit_action.setShortcut("Escape")
        self.quit_action.setShortcutContext(Qt.ApplicationShortcut)
        self.quit_action.triggered.connect(QApplication.instance().exit)

        self.swap_action = QAction("Show previous rendering", self)
        self.swap_action.setShortcut("\\")
        self.swap_action.setShortcutContext(Qt.ApplicationShortcut)
        self.swap_action.setCheckable(True)

        self.setsamples_action = QAction("Set samples per pixel", self)
        self.setsamples_action.triggered.connect(self._set_samples_per_pixel)

        def extrapolate_changed(value):
            self.datamodel.selected_material.extrapolate = value

        self.extrapolate_action = QAction("Harmonic extrapolation", self)
        self.extrapolate_action.setShortcut("H")
        self.extrapolate_action.setShortcutContext(Qt.ApplicationShortcut)
        self.extrapolate_action.triggered.connect(extrapolate_changed)
        self.extrapolate_action.setCheckable(True)

        fileMenu = self.menuBar().addMenu("&File")
        fileMenu.addAction(self.open_action)
        fileMenu.addSeparator()
        fileMenu.addAction(self.save_action)
        fileMenu.addAction(self.saveas_action)
        fileMenu.addSeparator()
        fileMenu.addAction(self.reset_action)
        fileMenu.addSeparator()
        fileMenu.addAction(self.exportimg_action)
        fileMenu.addAction(self.exportscrshot_action)
        fileMenu.addAction(self.exportsvg_action)
        fileMenu.addSeparator()
        fileMenu.addAction(self.quit_action)
        optionsMenu = self.menuBar().addMenu("&Options")
        optionsMenu.addAction(self.swap_action)
        optionsMenu.addAction(self.extrapolate_action)
        optionsMenu.addAction(self.setsamples_action)

        hsplitter = QSplitter(Qt.Horizontal, self)
        vsplitter = QSplitter(Qt.Vertical, self)
        scroller1 = QScrollArea(self)
        scroller2 = QScrollArea(hsplitter)
        scroller1.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroller2.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroller1.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroller2.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.layer_view_parent = QWidget(vsplitter)
        self.layer_view = LayerView(self.layer_view_parent, self.datamodel)
        self.layer_editor = LayerEditor(scroller1, self.datamodel)
        self.layer_editor.structure_changed.connect(self.layer_view.update)

        toolbar = QToolBar(self.tr('Toolbar'), self.layer_view_parent)
        toolbar.addAction(self.add_action)
        toolbar.addAction(self.remove_action)
        toolbar.addSeparator()
        toolbar.addAction(self.open_action)
        toolbar.addAction(self.save_action)
        toolbar.addAction(self.saveas_action)
        toolbar.addAction(self.preset_action)
        toolbar.setFloatable(False)
        toolbar.setMovable(False)
        toolbar.setStyleSheet("QToolBar { border: 0px; }")

        spacer = QWidget(toolbar)
        spacer.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Preferred)
        toolbar.addWidget(spacer)
        toolbar.addAction(self.render_action)

        layout = QVBoxLayout()
        layout.addWidget(toolbar)
        layout.addWidget(self.layer_view)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        self.layer_view_parent.setLayout(layout)

        self.setContentsMargins(10, 10, 10, 10)
        self.layer_view.layer_selected.connect(self.layer_editor.layer_selected)
        self.layer_view.interface_selected.connect(self.layer_editor.interface_selected)
        self.layer_view.layer_selected.connect(self._layer_selection_changed)
        self.layer_view.interface_selected.connect(self._layer_selection_changed)

        self.queue = RenderQueue()
        self.scheduler = Scheduler.getInstance()
        self.rwidget = RenderWidget(scroller2, self.queue,
            self.reference_bitmap)
        self.rwidget.maybe_load_preview(self.scene_preview)
        self.rwidget.pixel_clicked.connect(self._pixel_clicked)
        self.rwidget.rendering_updated.connect(self._rendering_updated)
        self.swap_action.triggered.connect(self.rwidget.swap)
        self.job = None

        scroller1.setWidget(self.layer_editor)
        scroller1.setWidgetResizable(True)
        scroller2.setWidget(self.rwidget)
        scroller2.setWidgetResizable(True)

        hsplitter.addWidget(scroller2)
        hsplitter.addWidget(vsplitter)
        vsplitter.addWidget(self.layer_view_parent)
        vsplitter.addWidget(scroller1)

        vsplitter2 = QSplitter(Qt.Vertical, self)
        self.log_widget = LogWidget(vsplitter2)
        setattr(material, 'log', self.log_widget.log)
        vsplitter2.addWidget(hsplitter)
        vsplitter2.addWidget(self.log_widget)

        self.setCentralWidget(vsplitter2)

        self._material_selection_changed()
        self.setWindowTitle('Layered material editor')
        size = self.scene.getFilm().getSize()
        total_width = 480 + size.x

        hsplitter.setSizes([int(size.x*1.2), total_width-size.x])
        vsplitter.setSizes([220, 300])
        vsplitter2.setSizes([600, 100])
        self.resize(total_width, 180 + size.y + (-20 if 'darwin' in sys.platform else 0))

    def center(self):
        screen_rect = QApplication.desktop().screenGeometry()
        self.move(screen_rect.center() - self.rect().center())

    def render(self):
        if self.job is not None:
            self.job.cancel()
            self.queue.waitLeft(0)

        changed = self.datamodel.build()
        if self.scene is None or changed:
            self.scene = SceneHandler.loadScene(self.fresolver.resolve(self.scene_filename))
            self.scene.initialize()
            self.scene.setDestinationFile(os.path.splitext(self.scene_filename)[0])

            pmgr = PluginManager.getInstance()

            materials = {}
            for matname in self.datamodel.materials.keys():
                props = Properties('tabulated')
                props['filename'] = str(self.datamodel.get_filename_for_material(matname))
                props.setID(str(matname))
                bsdf = pmgr.createObject(props)
                for i in range(self.datamodel.get_texture_count(matname)):
                    texProps = Properties('bitmap')
                    texProps['filename'] = 'wood.jpg'
                    texProps['uscale'] = float(4)
                    texProps['vscale'] = float(2)
                    texture = pmgr.createObject(texProps)
                    texture.configure()
                    bsdf.addChild(texture)
                    texture.setParent(bsdf)

                bsdf.configure()
                materials[matname] = bsdf

            for shape in get_shapes(self.scene.getShapes()):
                props = shape.getBSDF().getProperties()
                if props.getID() in materials and props.getPluginName() != 'tabulated':
                    shape.setBSDF(materials[props.getID()])
                    shape.configure()

        self.scene.setBlockSize(16)
        film = self.scene.getSensor()
        props = Properties('halton')
        props['sampleCount'] = self.spp
        sampler = PluginManager.getInstance().createObject(props)
        self.scene.setSampler(sampler)
        film.addChild(sampler)
        self.scene.configure()
        self.job = RenderJob('rjob', self.scene, self.queue)
        self.job.start()

    def shutdown(self):
        if self.job is not None:
            self.job.cancel()
            self.queue.waitLeft(0)
        self.queue.join()
        self.scheduler.stop()

    def _remove_layer(self):
        mat = self.datamodel.selected_material
        del mat.layers[mat.selected_index]
        del mat.interfaces[mat.selected_index]
        mat.selected_layer = False
        mat.init_interface_status(mat.selected_index)
        self.layer_view.update()
        self._layer_selection_changed()
        self.layer_editor.interface_selected(mat.selected_index)

    def _add_layer(self):
        mat = self.datamodel.selected_material
        layer = mat.create_layer()
        interface = mat.interfaces[mat.selected_index].copy()
        mat.layers.insert(mat.selected_index, layer)
        mat.interfaces.insert(mat.selected_index, interface)
        mat.selected_layer = True
        mat.init_interface_status(mat.selected_index)
        mat.init_interface_status(mat.selected_index+1)
        self.layer_editor.layer_selected(mat.selected_index)
        self.layer_view.update()
        self._layer_selection_changed()

    def _load_preset(self):
        mat = self.datamodel.selected_material
        layer = mat.layers[mat.selected_index]
        filename = self.sender().data()
        if type(filename) is QVariant and filename.toString is not None:
            filename = filename.toString()
        with open(filename, 'r') as infile:
            layer.update(json.load(infile))
            self.layer_editor.layer_selected(mat.selected_index)
            self.layer_view.update()
            self._layer_selection_changed()

    def _export_image(self):
        bitmap = self.rwidget.get_bitmap()
        if bitmap is None:
            return
        try:
            fname = str(QFileDialog.getSaveFileName(self, 'Export image', filter='PNG image, *.png;;JPG image, *.jpg'))
            if fname == '':
                return
        except:
            return
        fs = FileStream(fname, FileStream.ETruncReadWrite)
        if fname.endswith('jpg'):
            bitmap.write(Bitmap.EJPEG, fs)
        else:
            bitmap.write(Bitmap.EPNG, fs)
        fs.close()

    def _export_svg(self):
        try:
            fname = str(QFileDialog.getSaveFileName(self, 'Export SVG', filter='Scalable Vector Graphics, *.svg'))
            if fname == '':
                return
        except:
            return
        self.layer_view.export_svg(fname)

    def _reset(self):
        print('resetting')
        for name, material in self.datamodel.materials.items():
            if os.path.exists(material.filename):
                try:
                    print('Deleting %s' % material.filename)
                    os.remove(material.filename)
                except:
                    print('Warning: could not delete the file!')
        self.datamodel.selected_material = None
        self.datamodel.materials.clear()
        self.layer_view.update()
        self.layer_editor.layer_selected(None)

    def _export_scrshot(self):
        try:
            fname = str(QFileDialog.getSaveFileName(self, 'Export screenshot', filter='PNG image, *.png;;JPG image, *.jpg'))
            if fname == '':
                return
        except:
            return
        pixmap = QPixmap.grabWidget(self)
        file = QFile(fname)
        file.open(QIODevice.WriteOnly)
        if fname.endswith('jpg'):
            pixmap.save(file, "JPG")
        else:
            pixmap.save(file, "PNG")
        file.close()

    def _saveas(self):
        try:
            fname = str(QFileDialog.getSaveFileName(self, 'Save file', filter='Layer file, *.json'))
            if fname == '':
                return
        except:
            return
        self.datamodel.filename = fname + '.json' if not fname.endswith('.json') else fname
        self.datamodel.save(self.datamodel.filename)

    def _save(self):
        if self.datamodel.filename is not None:
            self.datamodel.save(self.datamodel.filename)
        else:
            self._saveas()

    def _open(self):
        try:
            fname = str(QFileDialog.getOpenFileName(self, 'Open file', filter='Layer file, *.json'))
            if fname == '':
                return
        except:
            return
        self.datamodel.load(fname)
        self.datamodel.selected_material = None
        self.layer_view.update()
        self.layer_editor.layer_selected(None)

    def _set_samples_per_pixel(self):
        result, success = QInputDialog.getInteger(self, "Set samples per pixel", "Value", self.spp, 1, 8192, 1)
        if success:
            self.spp = result

    def _pixel_clicked(self, x, y):
        weight, ray = self.scene.getSensor().sampleRay(Point2(x, y), Point2(.5), 0)
        intersection = self.scene.rayIntersect(ray)
        if intersection is None:
            return
        name = intersection.shape.getBSDF().getProperties().getID()
        if name == 'unnamed':
            return

        prev_selection = self.datamodel.selected_material
        cur_selection = self.datamodel.select_material(name)
        if prev_selection is not cur_selection:
            self._material_selection_changed()

        print('Selected material \"%s\".' % name)

    def _material_selection_changed(self):
        mat = self.datamodel.selected_material
        self.layer_view.update()
        if mat is not None:
            self.layer_editor.layer_selected(0)
        self._layer_selection_changed()
        self.extrapolate_action.setEnabled(mat is not None)
        self.extrapolate_action.setChecked(mat.extrapolate if mat is not None else False)

    def _layer_selection_changed(self):
        mat = self.datamodel.selected_material
        self.remove_action.setEnabled(mat is not None and mat.selected_layer and len(mat.layers) > 1)
        self.preset_action.setEnabled(mat is not None and mat.selected_layer and len(mat.layers) > 0)
        self.add_action.setEnabled(mat is not None and not mat.selected_layer)

    def _rendering_updated(self, flag):
        if flag == MitsubaRenderBuffer.RENDERING_FINISHED:
            fs = FileStream(self.scene_preview, FileStream.ETruncReadWrite)
            bitmap = self.rwidget.get_bitmap()
            bitmap.write(Bitmap.EPNG, fs)
            fs.close()
