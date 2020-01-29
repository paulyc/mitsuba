export CFLAGS="-DBOOST_ALL_NO_LIB=0 -I/usr/include/glew-1.13.0" #NOPENOPE-I/opt/qt5/include -I/opt/qt5/include
export CXXFLAGS="-DBOOST_ALL_NO_LIB=0 -I/usr/include/glew-1.13.0" #NOPENOPE-I/opt/qt5/include

#NOPENOPENOPE you have to do these ONLY for the ONE object file and executable that uses QT5 apparently
#export LDFLAGS=-L/opt/qt5/lib -lQt5XmlPatterns

#ok fine, lol
ln -sf /usr/include/qt/QtXmlPatterns src/mtsgui/QtXmlPatterns
