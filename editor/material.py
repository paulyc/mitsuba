import json, os, math

from mitsuba.layer import Layer, initializeQuadrature, BSDFStorage, \
        parameterHeuristicMicrofacet, parameterHeuristicHG, \
        parameterHeuristicVMF

from mitsuba.core import Timer, timeString, memString, Log, EInfo

import PyQt4.QtCore

def json_fix(value):
    if type(value) is dict:
        return { json_fix(k) : json_fix(v) for k, v in value.items() }
    elif type(value) is list:
        return [ json_fix(v) for v in value ]
    if hasattr(PyQt4.QtCore, 'QString'):
        from PyQt4.QtCore import QString
        if type(value) is QString:
            return str(value)
    return value

class LayeredMaterial:
    """ Keeps track of all information about a layered material """

    DIELECTRIC_LAYER       = 0
    CONDUCTOR_LAYER        = 1
    DIFFUSE_LAYER          = 2
    MEASURED_LAYER         = 3

    DIELECTRIC_ETA_DEFAULT = 1.5046
    CONDUCTOR_ETA_DEFAULT  = 0
    CONDUCTOR_K_DEFAULT    = 1
    SIGMA_T_DEFAULT        = 0
    ALBEDO_DEFAULT         = 0
    G_DEFAULT              = 0
    CONSERVE_DEFAULT       = True
    ROUGHNESS_DEFAULT      = 0.1
    SIGMA_T_SCALE_DEFAULT  = 1
    PHASE_TYPE_DEFAULT     = "hg"

    def __init__(self, filename, fail_if_missing=False, json_data=None, structure=None):
        self.filename = filename
        self.selected_index = 0
        self.selected_layer = True
        self.filesize = -1

        if structure is None:
            try:
                if json_data is None:
                    json_data = BSDFStorage.readMetadata(self.filename)
                    self.filesize = os.path.getsize(self.filename)
                structure = json.loads(json_data)
            except:
                if fail_if_missing:
                    raise
                print('Unable to load material structure from \"%s\", json_data=\"%s\"' % (filename, str(json_data)))

        if structure is not None:
            self.layers = structure['layers']
            self.interfaces = structure['interfaces']
            self.extrapolate = structure['extrapolate'] if 'extrapolate' in structure else False
        else:
            # Default material: simple diffuse slab
            self.layers = [dict(type=LayeredMaterial.DIFFUSE_LAYER, enabled=True)]
            self.interfaces = [dict(enabled=True), dict(enabled=False)]
            self.extrapolate = False

    def to_dict(self):
        result = dict(layers=self.layers, interfaces=self.interfaces, extrapolate=self.extrapolate)
        return json_fix(result)

    def _build_layer(self, index, nodes, weights, mTarget, mActual):
        """ Compute scattering matrices for a Henyey-Greenstein/diffuse/measured layer """
        if index < 0 or index >= len(self.layers):
            return [None]

        layer = self.layers[index]
        enabled = layer['enabled'] if 'enabled' in layer else True

        if not enabled:
            return [None]

        log('===   Layer %i   ===' % index)
        channels = 1
        result = []

        if layer['type'] == LayeredMaterial.DIELECTRIC_LAYER:
            sigma_t = layer['sigma_t'] if 'sigma_t' in layer else LayeredMaterial.SIGMA_T_DEFAULT
            albedo = layer['albedo'] if 'albedo' in layer else LayeredMaterial.ALBEDO_DEFAULT
            g = layer['g'] if 'g' in layer else LayeredMaterial.G_DEFAULT
            phase_type = layer['phase_type'] if 'phase_type' in layer else LayeredMaterial.PHASE_TYPE_DEFAULT
            sigma_t_scale = layer['sigma_t_scale'] if 'sigma_t_scale' in layer else LayeredMaterial.SIGMA_T_SCALE_DEFAULT

            if type(sigma_t) != list:
                sigma_t = [sigma_t] * 3
            if type(albedo) != list:
                albedo = [albedo] * 3
            if type(g) != list:
                g = [g] * 3

            for i in range(1, 3):
                if albedo[i] != albedo[0] or sigma_t[i] != sigma_t[0] or g[i] != g[0]:
                    channels = 3

            if 'diffuse_albedo_tex' in layer or 'sigma_t_tex' in layer:
                channels = 3

            for ch in range(channels):
                if sigma_t[ch] == 0:
                    result.append(None)
                    continue

                layer = Layer(nodes, weights, mActual)
                sigma_t_value = sigma_t[ch] * (1 if sigma_t[0] == sigma_t[1] and sigma_t[0] == sigma_t[2] else sigma_t_scale)

                if albedo[ch] == 0:
                    log('  ch%i: Creating absorbing layer with tau=%f ..' % (ch, sigma_t_value))
                    layer.setAbsorbing(sigma_t_value)
                elif phase_type == 'hg':
                    log('  ch%i: Creating HG layer with albedo=%f, g=%f ..' % (ch, albedo[ch], g[ch]))
                    layer.setHenyeyGreenstein(albedo[ch], g[ch])
                elif phase_type == 'vmf':
                    kappa = vmfKappa(g[ch])
                    log('  ch%i: Creating vMF layer with albedo=%f, g=%f (-> kappa=%f)..' % (ch, albedo[ch], g[ch], kappa))
                    layer.setVMF(albedo[ch], kappa)
                else:
                    raise Exception('Unknown phase function type %s' % phase_type)

                if albedo[ch] != 0:
                    log('  ch%i: Expanding to optical density %f ..' % (ch, sigma_t_value))
                    layer.solveAddingDoubling(sigma_t_value)
                result.append(layer)
        elif layer['type'] == LayeredMaterial.DIFFUSE_LAYER:
            diffuse_albedo = layer['diffuse_albedo'] if 'diffuse_albedo' in layer else .5
            if type(diffuse_albedo) != list:
                diffuse_albedo = [diffuse_albedo] * 3

            for i in range(1, 3):
                if diffuse_albedo[i] != diffuse_albedo[0]:
                    channels = 3

            if 'diffuse_albedo_tex' in layer:
                channels = 3

            for ch in range(channels):
                log('  ch%i: Creating diffuse interface with albedo=%f ..' % (ch, diffuse_albedo[ch]))
                layer = Layer(nodes, weights, mActual)
                layer.setDiffuse(diffuse_albedo[ch])
                result.append(layer)
        elif layer['type'] == LayeredMaterial.MEASURED_LAYER:
            filename = layer['filename'] if 'filename' in layer else None

            if filename is None:
                raise Exception('Measured material filename must be specified!')

            channels = 3
            for ch in range(channels):
                log('  ch%i: running FFT on \"%s\" ..' % (ch, os.path.basename(filename)))
                layer = Layer(nodes, weights, mActual)
                layer.setMatusik(filename, ch, mTarget)
                result.append(layer)
        else:
            result.append(None)

        return result

    def _get_parameters_interface(self, index):
        if index < 0 or index >= len(self.interfaces):
            return None, None, None

        itf = self.interfaces[index]
        prevLayer = self.layers[index - 1] if index > 0 else None
        nextLayer = self.layers[index] if index < len(self.layers) else None

        enabled = itf['enabled'] if 'enabled' in itf else True
        prevEnabled = prevLayer['enabled'] if prevLayer is not None and 'enabled' in prevLayer else True
        nextEnabled = nextLayer['enabled'] if nextLayer is not None and 'enabled' in nextLayer else True

        if not enabled or not prevEnabled or not nextEnabled:
            return None, None, None

        alpha = itf['roughness'] if 'roughness' in itf else LayeredMaterial.ROUGHNESS_DEFAULT
        conserve = itf['conserve'] if 'conserve' in itf else LayeredMaterial.CONSERVE_DEFAULT
        prevEta, prevK = [1]*3, [0]*3
        nextEta, nextK = [1]*3, [0]*3

        if prevLayer is not None:
            if prevLayer['type'] == LayeredMaterial.DIELECTRIC_LAYER:
                prevEta = [prevLayer['eta']]*3 if 'eta' in prevLayer else [LayeredMaterial.DIELECTRIC_ETA_DEFAULT]*3
            if prevLayer['type'] == LayeredMaterial.CONDUCTOR_LAYER:
                prevEta = prevLayer['eta_cond'] if 'eta_cond' in prevLayer else [LayeredMaterial.CONDUCTOR_ETA_DEFAULT]*3
                prevK = prevLayer['k_cond'] if 'k_cond' in prevLayer else [LayeredMaterial.CONDUCTOR_K_DEFAULT]*3
        if nextLayer is not None:
            if nextLayer['type'] == LayeredMaterial.DIELECTRIC_LAYER:
                nextEta = [nextLayer['eta']]*3 if 'eta' in nextLayer else [LayeredMaterial.DIELECTRIC_ETA_DEFAULT]*3
            if nextLayer['type'] == LayeredMaterial.CONDUCTOR_LAYER:
                nextEta = nextLayer['eta_cond'] if 'eta_cond' in nextLayer else [LayeredMaterial.CONDUCTOR_ETA_DEFAULT]*3
                nextK = nextLayer['k_cond'] if 'k_cond' in nextLayer else [LayeredMaterial.CONDUCTOR_K_DEFAULT]*3

        eta = [complex(nextEta[ch], nextK[ch]) / complex(prevEta[ch], prevK[ch]) for ch in range(3)]

        return eta, alpha, conserve

    def _build_interface(self, index, nodes, weights, mTarget, mActual, config_only=False):
        eta, alpha, conserve = self._get_parameters_interface(index)

        if eta is None:
            return [None], 1

        log('=== Interface %i ===' % index)

        channels = 1 if eta[1] == eta[0] and eta[2] == eta[0] else 3
        result = []

        for ch in range(channels):
            if eta[ch].imag != 0:
                if eta[ch].imag < 0:
                    #log('Skipping bottom of opaque layer..')
                    result.append(None)
                    continue
                log('  ch%i: Creating rough conductor interface with alpha=%f, eta=%s ..'
                      % (ch, alpha, str(eta[ch])))
            elif eta[ch].real != 1:
                log('  ch%i: Creating rough dielectric interface with alpha=%f, eta=%f ..'
                      % (ch, alpha, eta[ch].real))
            else:
                log('  ch%i: Interface is index-matched.' % ch)
                result.append(None)
                continue

            layer = Layer(nodes, weights, mActual)
            layer.setMicrofacet(eta[ch].real, eta[ch].imag, alpha, conserve, mTarget)
            result.append(layer)

        return result, eta[0].real

    def _get_mincoeffs_layer(self, index):
        m = n = 0
        if index < 0 or index >= len(self.layers):
            return 0, 0

        layer = self.layers[index]
        enabled = layer['enabled'] if 'enabled' in layer else True
        if not enabled:
            return 0, 0

        if layer['type'] == LayeredMaterial.DIELECTRIC_LAYER:
            g = layer['g'] if 'g' in layer else LayeredMaterial.G_DEFAULT
            if type(g) != list:
                g = [g] * 3
            for i in range(3):
                np, mp = parameterHeuristicHG(g[i])
                n = max(n, np)
                m = max(m, mp)
        elif layer['type'] == LayeredMaterial.DIFFUSE_LAYER:
            m, n = 1, 20
        elif layer['type'] == LayeredMaterial.MEASURED_LAYER:
            m = n = 200  # Unknown...
        return n, m

    def _get_mincoeffs_interface(self, index):
        eta, alpha, conserve = self._get_parameters_interface(index)
        if eta is None:
            return 0, 0

        channels = 1 if eta[1] == eta[0] and eta[2] == eta[0] else 3
        n = m = 0

        for ch in range(channels):
            if eta[ch].imag < 0: # Skip bottom of opaque layer, skip ior=1 layers
                continue
            if eta[ch].imag == 0 and eta[ch].real == 1:
                continue
            np, mp = parameterHeuristicMicrofacet(alpha, eta[ch].real, eta[ch].imag)
            n = max(np, n)
            m = max(mp, m)
        return n, m

    def _get_mincoeffs(self):
        n = []; m = []
        for i in range(len(self.interfaces)):
            n1, m1 = self._get_mincoeffs_interface(i)
            n2, m2 = self._get_mincoeffs_layer(i)
            n.append(n1); n.append(n2); m.append(m1); m.append(m2)
        n = max(n)
        m = [i for i in m if i!=0]
        m = m[0] if self.is_opaque() else max(m[0], m[len(m)-1])
        return n, m

    def is_opaque(self):
        opaque = False
        for layer in self.layers:
            enabled = layer['enabled'] if 'enabled' in layer else True
            if not enabled:
                continue
            if layer['type'] != LayeredMaterial.DIELECTRIC_LAYER:
                opaque = True
        return opaque

    def _build_material(self):
        n, mTarget = self._get_mincoeffs()
        log('Building material \"%s\" with n=%i, m=%i' % (self.filename, n, mTarget))

        nodes, weights = initializeQuadrature(n)
        mActual = 3 if self.extrapolate else mTarget

        opaque = False
        total_eta = 1  # Relative ior through the entire stack
        result = [None]

        def merge_into(a, b):
            while len(a) < len(b):
                if a[0] is not None:
                    log('  ch%i: Cloning from channel 0' % (len(a) + 1))
                    a.append(Layer(a[0]))
                else:
                    a.append(None)
            while len(b) < len(a):
                b.append(b[0])
            for i in range(len(a)):
                if a[i] is None and b[i] is not None:
                    a[i] = b[i]
                elif a[i] is not None and b[i] is not None:
                    log('  ch%i: Merging layers' % i)
                    a[i].addToBottom(b[i], False)

        for i in range(len(self.interfaces)):
            itf, eta = self._build_interface(i, nodes, weights, mTarget, mActual)
            layer = self._build_layer(i, nodes, weights, mTarget, mActual)
            total_eta *= eta

            merge_into(result, itf)
            merge_into(result, layer)

        if self.is_opaque():
            total_eta = 1
            for layer in result:
                layer.clearBottomAndTransmission()

        return result, total_eta

    def _parameter_samples(self, minValue, maxValue, nSamples):
        mean = .5 * (minValue+maxValue)
        radius = .5 * (maxValue-minValue)
        return [mean - radius * math.cos(math.pi * (i + .5) / nSamples) for i in range(nSamples)]

    def get_texture_count(self):
        result = 0
        for obj in self.layers + self.interfaces:
            for key, config in obj.items():
                if key.endswith('_tex'):
                    result += 1
        return result

    def init_interface_status(self, index):
        check = []
        if index >= 1:
            check.append(self.layers[index - 1]['type'])
        if index < len(self.layers):
            check.append(self.layers[index]['type'])
        enabled = True
        for t in check:
            if t == LayeredMaterial.MEASURED_LAYER or \
               t == LayeredMaterial.DIFFUSE_LAYER:
                enabled = False
        self.interfaces[index]['enabled'] = enabled

    def build(self):
        jsonData = json.dumps(self.to_dict(), indent=4)

        # Check if there were any changes. If not, leave the file as is
        if os.path.exists(self.filename):
            try:
                jsonDataFile = BSDFStorage.readMetadata(self.filename)
                if json.loads(jsonData) == json.loads(jsonDataFile):
                    return False
            except:
                pass

        nBases = 1
        parameters = []
        paramSampleCounts = []
        paramSamplePositions = []
        paramSamplePositionsFile = []

        for obj in self.layers + self.interfaces:
            for key, config in obj.items():
                if key.endswith('_tex'):
                    minValue, maxValue, nSamples = config[0], config[1], config[2]
                    nBases = nBases * nSamples
                    paramSampleCounts.append(nSamples)
                    paramSamplePositions.append(self._parameter_samples(minValue, maxValue, nSamples))
                    paramSamplePositionsFile.append(self._parameter_samples(0, 1, nSamples))
                    parameters.append(dict(obj=obj, key=key[:-4]))

        timer = Timer()
        total_eta = 0
        result = []

        for i in range(nBases):
            idx = i
            for j in range(len(parameters)):
                param = parameters[j]
                param['obj'][param['key']] = paramSamplePositions[j][idx % paramSampleCounts[j]]
                idx //= paramSampleCounts[j]

            result_instance, total_eta = self._build_material()
            result += result_instance

        if result[0] is None:
            raise Exception('There must be at least one active layer/interface!')

        nChannels = len(result)//nBases
        precomputationTime = timer.getMilliseconds() / 1000.0
        timer.reset()
        storage = BSDFStorage.fromLayerGeneral(str(self.filename), result, nChannels, nBases, len(parameters),
                                               paramSampleCounts, paramSamplePositionsFile, self.extrapolate,
                                               True, str(jsonData))
        log(storage.stats())

        storage.setEta(total_eta)
        size = storage.size()
        storage.close()
        storageTime = timer.getMilliseconds() / 1000.0

        log(' Done (precomputation took %s, saving took %s).' % (timeString(precomputationTime, True), timeString(storageTime, True)))


        if self.filesize != size:
            self.filesize = size
            return True

        return False

    def create_layer(self):
        return dict(
            type=LayeredMaterial.DIELECTRIC_LAYER
        )


class DataModel:
    def __init__(self):
        self.materials = {}
        self.selected_material = None
        self.changed = False
        self.filename = None

    def to_dict(self):
        return {name: value.to_dict() for name, value in self.materials.items()}

    def get_filename_for_material(self, name):
        return os.path.join(os.path.dirname(os.path.realpath(__file__)), os.path.join('materials', name + '.bsdf'))

    def get_texture_count(self, name):
        return self.materials[name].get_texture_count()

    def select_material(self, name):
        if name in self.materials:
            material = self.materials[name]
        else:
            material = LayeredMaterial(self.get_filename_for_material(name))
            self.materials[name] = material
            self.changed = True

        self.selected_material = material
        return material

    def try_import_material(self, name):
        if name in self.materials:
            return False
        try:
            material = LayeredMaterial(self.get_filename_for_material(name), fail_if_missing=True)
            self.materials[name] = material
            self.changed = True
            self.selected_material = material
            return True
        except:
            return False

    def save(self, filename):
        log('Writing to \"%s\" ..' % filename)
        with open(filename, 'w') as outfile:
            json.dump(self.to_dict(), outfile, indent=4)

    def load(self, filename):
        log('Loading from \"%s\" ..' % filename)
        with open(filename, 'r') as infile:
            self.materials = {name: LayeredMaterial(self.get_filename_for_material(name), structure=structure) for name, structure in json.load(infile).items()}
        log('Loaded %i materials: %s' % (len(self.materials), str(self.materials.keys())))
        self.filename = filename

    def build(self):
        try:
            os.mkdir(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'materials'))
        except OSError as e:
            pass
        changed = self.changed
        for material in self.materials.values():
            changed |= material.build()
        self.changed = False
        return changed
