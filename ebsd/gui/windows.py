import icons
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import *
from PyQt5 import QtGui
import numpy as np
from ..plots import widgets
from . tableview import QtTable, QTableView
Binstances = list()
color_dict = {
    'red': ['#ff0000', (255, 0, 0)],
    'green': ['#00ff00', (0, 255, 0)],
    'blue': ['#0000ff', (0, 0, 255)],
    'yellow': ['#ffff00', (255, 255, 0)],
    'gold': ['#ffd700', (255, 215, 0)],
    'pink': ['#ffc0cb', (255, 192, 203)],
    'bisque': ['#ffe4c4', (255, 228, 196)],
    'ivory': ['#fffff0', (255, 255, 240)],
    'black': ['#000000', (0, 0, 0)],
    'white': ['#ffffff', (255, 255, 255)],
    'violet': ['#ee82ee', (238, 130, 238)],
    'silver': ['#c0c0c0', (192, 192, 192)],
    'forestgreen': ['#228b22', (34, 139, 34)],
    'brown': ['#a52a3a', (165, 42, 58)],
    'chocolate': ['#d2691e', (210, 105, 30)],
    'azure': ['#fffff0', (255, 255, 240)],
    'orange': ['#ffa500', (255, 165, 0)]
}


class CustomDialogFilter(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.selected = None
        self.setWindowTitle("Kernel Size")
        QBtn = QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        self.buttonBox = QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        self.layout = QVBoxLayout()
        self.message = QLabel("Choose Median Filter Kernel Size ")
        self.layout.addWidget(self.message)

        three = QRadioButton(self)
        three.setText("3 x 3")
        self.layout.addWidget(three)
        five = QRadioButton(self)
        five.setText("5 x 5")
        self.layout.addWidget(five)
        seven = QRadioButton(self)
        seven.setText("7 x 7")
        self.layout.addWidget(seven)
        nine = QRadioButton(self)
        nine.setText("9 x 9")
        self.layout.addWidget(nine)
        eleven = QRadioButton(self)
        eleven.setText("11 x 11")
        self.layout.addWidget(eleven)

        self.layout.addWidget(self.buttonBox)
        self.setLayout(self.layout)
        three.toggled.connect(self.onClicked)
        five.toggled.connect(self.onClicked)
        seven.toggled.connect(self.onClicked)
        nine.toggled.connect(self.onClicked)
        eleven.toggled.connect(self.onClicked)

    def onClicked(self):
        radioBtn = self.sender()
        if radioBtn.isChecked():
            self.selected = radioBtn.text()


class SettingsWindow(QWidget):
    """
    This "window" is a Settings Window, Values are preset, but
    can be modified to desired values.
    """

    def __init__(self):
        super().__init__()
        self.grid = QGridLayout()
        self.b_values_list = list()
        self.ang_list = list()
        self.setWindowIcon(QtGui.QIcon('Icon.ico'))

        self.setWindowTitle('SETTINGS')
        self._setup()
        self.set_default_vals()
        self.setLayout(self.grid)
        self.setFixedSize(500, 600)

    def _setup(self):
        # Burgers Vector value
        units = '\u00C5'  # angstrom's unicode
        bVal = QLabel(f"Burgers Vector ({units})")
        self.BLine = QLineEdit()
        self.grid.addWidget(bVal, 1, 1)
        self.grid.addWidget(self.BLine, 1, 2)
        self.BLine.textChanged.connect(self.sync_lineEdit)
        # Filters
        # self.filter_choice = QLabel('Choose Filter')
        # self.filters = QComboBox()
        # _filters_list = ['Median Filter', 'Mean Filter', 'A']
        # self.filters.addItems(_filters_list)
        # self.grid.addWidget(self.filter_choice, 2, 1)
        # self.grid.addWidget(self.filters, 2, 2)
        # Segmentation Choice
        self.seg_choice = QLabel('Segementation Method')
        seg_list = ['KAM Based', 'Watershed', "Felzenszwalb Segmentation"]
        self.seg_methods = QComboBox()
        self.seg_methods.addItems(seg_list)
        self.grid.addWidget(self.seg_choice, 2, 1)
        self.grid.addWidget(self.seg_methods, 2, 2)

        # pole figure axis choice
        self.pfdd_list = ["{100}", "{010}", "{001}"]
        pfch = QLabel('Pole axis :')
        self.pfdd = QComboBox()
        self.pfdd.addItems(self.pfdd_list)
        self.grid.addWidget(pfch, 3, 1)
        self.grid.addWidget(self.pfdd, 3, 2)
        # ipf axis choice (Z/Y/X)
        self.ipfdd_list = ["X", "Y", "Z"]
        ipfch = QLabel('IPF axis :')
        self.ipfdd = QComboBox()
        self.ipfdd.addItems(self.ipfdd_list)
        self.grid.addWidget(ipfch, 4, 1)
        self.grid.addWidget(self.ipfdd, 4, 2)
        # ok button
        self.okbtn = QPushButton('OK')
        self.okbtn.clicked.connect(self.closewin)
        self.grid.addWidget(self.okbtn, 5, 2)

    def closewin(self):
        self.close()

    def sync_lineEdit(self, text):
        self.b_values_list.append(text)

    def set_default_vals(self):
        # default value of B value (235)
        if self.b_values_list != []:
            self.BLine.setText(self.b_values_list[-1])
        else:
            self.BLine.setText('235')
        # default filter - median filter
        # self.filters.setCurrentText(0)
        # default segmentation method - kam based
        self.seg_methods.setCurrentIndex(0)
        # default value of PF axis {001}
        self.pfdd.setCurrentIndex(2)
        # ipf Z is displayed by default
        self.ipfdd.setCurrentIndex(2)

    # get functions

    def getBValue(self):
        return self.BLine.text()

    # def getFilter(self):
    #     return str(self.filters.currentText())

    def getSegmentation(self):
        return str(self.seg_methods.currentText())

    def getPF(self):
        choice = str(self.pfdd.currentText())
        return choice

    def getIPF(self):
        choice = str(self.ipfdd.currentText())
        return choice


class Boundary(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowIcon(QtGui.QIcon('Icon.ico'))

        Binstances.append(self)
        self.grid = QGridLayout()
        self.setGeometry(320, 200, 340, 220)
        self.combo = QComboBox()

        self.frame = QFrame()
        self.frame.setFrameStyle(QFrame.Box | QFrame.Sunken)
        self.ang = QLineEdit()
        self.lab = QLabel('Min. Angle (°)')

        self.remove = QPushButton(QIcon(":bin.png"), '')
        self.remove.clicked.connect(self.removeB)

        self.grid.addWidget(self.remove, 1, 2, 1, 1)
        self.grid.addWidget(self.lab, 2, 1, 1, 1)
        self.grid.addWidget(self.ang, 2, 2, 1, 1)
        self.grid.addWidget(self.combo, 3, 1, 1, 1)
        self.grid.addWidget(self.frame, 3, 2, 1, 1)

        self.color_dict = color_dict
        # create a sorted color list
        self.color_list = sorted(color_dict.keys())
        # load the combobox
        for color in self.color_list:
            self.combo.addItem(color)
        # bind/connect selection to an action method
        self.setLayout(self.grid)
        self.combo.currentTextChanged.connect(self.onchange)
        self._default_init()

    def onchange(self, color):
        style_str = "QWidget {background-color: %s}"
        self.frame.setStyleSheet(style_str % self.color_dict[color][0])

    def removeB(self):
        for items in Binstances:
            if items == self:
                Binstances.remove(items)
                self.deleteLater()

    def getAngle(self):
        return int(self.ang.text())

    def getColor(self):
        tp = color_dict[self.combo.currentText()][-1]
        return np.array(tp) * 1/255

    def _default_init(self):
        self.ang.setText('5')
        self.combo.setCurrentText('yellow')


class KamWindow(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        # setGeometry(x_pos, y_pos, width, height)
        self.setGeometry(320, 200, 340, 220)
        self.setWindowTitle('Boundaries')
        self.setWindowIcon(QtGui.QIcon('Icon.ico'))

        self.box = QVBoxLayout()

        self.addBound = QPushButton(
            QIcon(":plus.png"), 'Add Boundaries')
        self.box.addWidget(self.addBound)
        self.addBound.clicked.connect(self.addmore)

        if not Binstances:
            self.b1 = Boundary()
            self.b2 = Boundary()
            self.box.addWidget(self.b1)
            self.box.addWidget(self.b2)
        else:
            for items in Binstances:
                self.box.addWidget(items)

        self.QBtn = QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        self.buttonBox = QDialogButtonBox(self.QBtn)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

        self.box.addWidget(self.buttonBox)

        self.setLayout(self.box)

    def addmore(self):
        self.b = Boundary()
        self.box.insertWidget(self.box.count()-1, self.b)

    def isSet(self):
        for items in Binstances:
            if items.getAngle():
                pass
            else:
                return False
        widgets.w['Binstances'] = Binstances
        return True


class PFSelect(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.selected = None
        self.setWindowTitle("Pole Figure View Selection")
        self.setWindowIcon(QtGui.QIcon('Icon.ico'))

        QBtn = QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        self.buttonBox = QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        self.layout = QVBoxLayout()
        self.message = QLabel("Choose Selection Type For PF Plot ")
        self.layout.addWidget(self.message)
        onept = QRadioButton(self)
        onept.setText("PF For every grain click")
        self.layout.addWidget(onept)
        selectarea = QRadioButton(self)
        selectarea.setText("PF For selected region")
        self.layout.addWidget(selectarea)
        fullarea = QRadioButton(self)
        fullarea.setText("PF For Entire region (Consumes time)")
        self.layout.addWidget(fullarea)
        self.layout.addWidget(self.buttonBox)
        self.setLayout(self.layout)
        onept.toggled.connect(self.onClicked)
        selectarea.toggled.connect(self.onClicked)
        fullarea.toggled.connect(self.onClicked)

    def onClicked(self):
        radioBtn = self.sender()
        if radioBtn.isChecked():
            self.selected = radioBtn.text()


class GrainInfoWin(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Grain Details")
        self.setWindowIcon(QtGui.QIcon('Icon.ico'))
        self.generate_data()

    def generate_data(self):
        try:
            self.centralWidget = QVBoxLayout()

            # self.df = obj.getdf()
            self.df = widgets.w['dfs'][-1]
            self.display()
        except:
            print("error!!!")

    def update_data(self, obj):
        try:
            self.df = obj.getdf()
        except:
            print('error!!')

    def display(self):
        try:
            self.model = QtTable(self.df)
            self.view = QTableView()
            # self.setStyleSheet(load_stylesheet_pyqt5())
            fnt = self.view.font()
            fnt.setPointSize(9)
            self.view .setFont(fnt)
            self.view .setModel(self.model)
            self.view .resize(1080, 400)
            self.savebtn = QPushButton('Export Data')
            self.centralWidget.addWidget(self.view)
            self.centralWidget.addWidget(self.savebtn)
            self.view.show()
        except AttributeError:
            print("error!!")

    def save_data_as_csv(self):
        name = QFileDialog.getSaveFileName(self, 'Save File', filter='*.csv')
        if(name[0] == ''):
            name = 'Segemented_grain_data'
        else:
            self.df.to_csv(name[0], index=False)
