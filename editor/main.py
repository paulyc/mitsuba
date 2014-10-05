#!/usr/bin/env python3

def search_file(search_path, filename):
    for path in search_path.split(os.pathsep):
        if os.path.exists(os.path.join(path, filename)):
            return os.path.abspath(os.path.join(path, filename))
    return None

def try_add_network_node(hostname):
    from mitsuba.core import SocketStream, RemoteWorker, Scheduler, Log, EInfo
    try:
        socketStream = SocketStream(hostname, 7554)
        remoteWorker = RemoteWorker('netWorker', socketStream)
        Scheduler.getInstance().registerWorker(remoteWorker)
        Log(EInfo, 'Successfully connected to \"%s\": added %i cores' % (hostname, remoteWorker.getCoreCount()))
        return True
    except:
        return False

if __name__ == '__main__':
    import sys, os, signal, platform
    from resources import resources

    # Stop the program upon Ctrl-C (SIGINT)
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    mitsuba_plugin = None
    mitsuba_libpath = None

    searchpath = os.getenv('PATH') + os.pathsep + os.path.expanduser("~/mitsuba-ad/dist")

    if 'darwin' in sys.platform:
        mitsuba_plugin = search_file(searchpath, '../../python/'
            + str(sys.version_info[0]) + '.' + str(sys.version_info[1]) + '/mitsuba.so')
    elif 'win' in sys.platform:
        mitsuba_plugin = search_file(searchpath, 'python/' + str(sys.version_info[0]) +
            '.' + str(sys.version_info[1]) + '/mitsuba.pyd')
    elif 'linux' in sys.platform:
        mitsuba_plugin = search_file(searchpath, 'python/' + str(sys.version_info[0]) +
            '.' + str(sys.version_info[1]) + '/mitsuba.so')

    if mitsuba_plugin is not None:
        # Load the Mitsuba plugin
        sys.path.append(os.path.dirname(mitsuba_plugin))
        print('Loading Mitsuba plugin from \"%s\"' % mitsuba_plugin)
    else:
        print("Warning: could not locate the Mitsuba plugin. " +
              "The next command will likely fail.")

    import mitsuba, multiprocessing
    from mitsuba.core import Scheduler, LocalWorker, Appender, Thread, EInfo, Statistics

    scheduler = Scheduler.getInstance()
    for i in range(0, multiprocessing.cpu_count()):
        scheduler.registerWorker(LocalWorker(i, 'wrk%i' % i))

    scheduler.start()

    from PyQt4.QtGui import QApplication, QIcon
    from editor import MaterialEditor

    app = QApplication(sys.argv)
    QApplication.setWindowIcon(QIcon(':/resources/appicon.png'))

    class CustomAppender(Appender):
        def append(self, logLevel, message):
            print(message)

    logger = Thread.getThread().getLogger()
    logger.setLogLevel(EInfo)
    #logger.clearAppenders()
    #logger.addAppender(CustomAppender())

    fresolver = Thread.getThread().getFileResolver()

    scene_filename = sys.argv[1] if len(sys.argv) > 1 else 'matpreview/matpreview.xml'
    reference_filename = sys.argv[2] if len(sys.argv) > 2 else None
    fresolver.appendPath(os.path.abspath(os.path.pardir))
    scene_filename = fresolver.resolve(scene_filename)
    fresolver.appendPath(os.path.dirname(scene_filename))

    window = MaterialEditor(scene_filename, reference_filename)
    window.center()
    window.show()
    window.raise_()
    retval = app.exec_()
    logger = Thread.getThread().getLogger()
    logger.clearAppenders()
    window.shutdown()
    Statistics.getInstance().printStats()
    sys.exit(retval)

