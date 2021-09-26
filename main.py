# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 19:41:19 2021

@author: harini
"""
# ebsd functions and gui resources imports
from ebsd.gui.windows import CustomDialogFilter, KamWindow, PFSelect, SettingsWindow, GrainInfoWin
from ebsd import *
# PyQt5 imports
import sys
from PyQt5 import QtCore, QtGui
from PyQt5.QtCore import QFileInfo, QSettings
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QIcon, QKeySequence
# matplotlib and other functions
from matplotlib.widgets import RectangleSelector
from skimage.segmentation import mark_boundaries
import numpy as np
import pandas as pd
import icons
#global variables
canvas = list()
selpoints = list()
grid = QGridLayout()
# stack for rectangle selections points.
_nav_stack = list()


class CubeWindow(QWidget):
    def __init__(self):
        """Initializer."""
        self.cube = MplCanvas3d()
        w['canvas']['canvas3d'].append(self.cube)


class Window(QMainWindow):
    """Main Window."""
    MaxRecentFiles = 3
    windowList = []

    def __init__(self, parent=None):
        """Initializer."""
        super().__init__(parent)
        self.recentFileActs = []
        self.setWindowTitle("EBSD Analysis")
        self.setWindowIcon(QtGui.QIcon('Icon.ico'))
        self.setMinimumSize(500, 500)
        self._createActions()
        self._createMenuBar()
        self._createToolBars()
        self._connectActions()
        self._createStatusBar()
        global layout
        self.widget = QWidget(self)
        layout = QHBoxLayout(self.widget)
        self.setCentralWidget(self.widget)
        self.win_settings = SettingsWindow()
        self.kseg_win = KamWindow()
        self.windows = list()
        self.tabs = None

    def _createMenuBar(self):
        menuBar = self.menuBar()
        # File menu
        fileMenu = QMenu("&File", self)
        menuBar.addMenu(fileMenu)
        fileMenu.addAction(self.newAction)
        fileMenu.addAction(self.openAction)

        self.openRecentMenu = fileMenu.addMenu("Open Recent")
        fileMenu.addAction(self.saveAction)
        # Adding a separator
        self.separatorAct = fileMenu.addSeparator()
        for i in range(Window.MaxRecentFiles):
            self.openRecentMenu.addAction(self.recentFileActs[i])
        fileMenu.addSeparator()
        fileMenu.addAction(self.exitAction)
        # Edit menu
        editMenu = menuBar.addMenu("&Edit")
        editMenu.addAction(self.copyAction)
        editMenu.addAction(self.pasteAction)
        editMenu.addAction(self.cutAction)
        # Adding a separator
        editMenu.addSeparator()
        # Find and Replace submenu in the Edit menu
        findMenu = editMenu.addMenu("Find and Replace")
        findMenu.addAction("Find...")
        findMenu.addAction("Replace...")
        # Help menu
        helpMenu = menuBar.addMenu(
            QIcon(":help-content.png"), "&Help")
        helpMenu.addAction(self.helpContentAction)
        helpMenu.addAction(self.aboutAction)

    def _createToolBars(self):
        # settings toolbar
        settingsTBar = QToolBar('settings', self)
        self.addToolBar(settingsTBar)
        settingsTBar.addAction(self.settings)
        settingsTBar.addAction(self.info)
        settingsTBar.addAction(self.plotsAction)
        # process tool bar
        processToolBar = QToolBar('process', self)
        self.addToolBar(processToolBar)
        processToolBar.addAction(self.filterAction)
        processToolBar.addAction(self.kamAction)
        processToolBar.addAction(self.ddAction)
        processToolBar.addAction(self.segementAction)
        processToolBar.addAction(self.grodAction)
        # processToolBar.addAction(self.wshedAction)
        # processToolBar.addAction(self.ksegAction)  # segment
        processToolBar.addAction(self.pfAction)  # pf
        processToolBar.addAction(self.ipfAction)
        processToolBar.addAction(self.rectselAction)  # select area
        processToolBar.addAction(self.undoAction)

    def _createActions(self):
        # File actions
        self.newAction = QAction(self)
        self.newAction.setText("&New")
        self.newAction.setIcon(QIcon(":file-new.png"))
        self.openAction = QAction(
            QIcon(":file-open.png"), "&Open...", self)
        self.saveAction = QAction(
            QIcon(":file-save.png"), "&Save", self)
        self.exitAction = QAction("&Exit", self)

        # Using string-based key sequences
        self.newAction.setShortcut("Ctrl+N")
        self.openAction.setShortcut("Ctrl+O")
        self.saveAction.setShortcut("Ctrl+S")
        # adding a tool tip - user friendly
        newTip = "Load a new file"
        self.newAction.setStatusTip(newTip)
        self.newAction.setToolTip(newTip)

        for i in range(Window.MaxRecentFiles):
            self.recentFileActs.append(
                QAction(self, visible=False,
                        triggered=self.openRecentFile))

        # Edit actions
        self.copyAction = QAction(
            QIcon(":edit-copy.png"), "&Copy", self)
        self.pasteAction = QAction(
            QIcon(":edit-paste.png"), "&Paste", self)
        self.cutAction = QAction(QIcon(":edit-cut.png"), "C&ut", self)

        # Using standard keys
        self.copyAction.setShortcut(QKeySequence.Copy)
        self.pasteAction.setShortcut(QKeySequence.Paste)
        self.cutAction.setShortcut(QKeySequence.Cut)

        self.helpContentAction = QAction("&Help Content", self)
        self.aboutAction = QAction("&About", self)

        # filtering
        self.filterAction = QAction(self)
        # QAction(QIcon("resoures/filter.png"),"&Median Filter", self)
        self.filterAction.setText("&Filter")
        self.filterAction.setIcon(QIcon(":filter.png"))
        # kam
        self.kamAction = QAction('KAM', self)
        self.kamAction.setIcon(QIcon(":kam.png"))
        self.kamAction.setStatusTip('Kernel Average Misorientation')
        # dislocation density map
        self.ddAction = QAction('Dislocation Density', self)
        self.ddAction.setIcon(QIcon(":rho.jpg"))
        self.ddAction.setStatusTip('Dislocation Density Map')
        # segmentation
        self.segementAction = QAction(
            QIcon(":segment.png"), 'Grain Segmentation', self)

        # grod action
        self.grodAction = QAction('GROD', self)
        self.grodAction.setStatusTip('Grain Reference Orientation deviation')
        # pf action
        self.pfAction = QAction('Pole figure', self)
        self.pfAction.setIcon(QIcon(":pf.png"))
        self.pfAction.setStatusTip('Pole Figure Map')
        # ipf action
        self.ipfAction = QAction(QIcon(":ipf.png"), 'IPF', self)
        self.ipfAction.setStatusTip('Inverse Pole Figure')

        # settings
        self.settings = QAction(
            QIcon(":pref.png"), 'Settings', self)
        # grain details
        self.info = QAction(
            QIcon(":database-file.png"), 'Grain Details', self)
        self.info.setStatusTip('Grain Details')
        # plots/ histograms
        self.plotsAction = QAction(QIcon(":bar.png"), 'Plots', self)
        self.info.setStatusTip('PLOTS')

        # rect select
        self.rectselAction = QAction(
            QIcon(":select-area.png"), 'Select Area', self)
        self.rectselAction.setStatusTip(
            'Select a part of image for generating PF plot')
        # undo selection
        self.undoAction = QAction(
            QIcon(":undo.png"), 'undo selection', self)

    def contextMenuEvent(self, event):
        # Creating a menu object with the central widget as parent
        menu = QMenu(self.centralWidget)
        # Populating the menu with actions
        menu.addAction(self.newAction)
        menu.addAction(self.openAction)
        menu.addAction(self.saveAction)
        # Creating a separator action
        separator = QAction(self)
        separator.setSeparator(True)
        # Adding the separator to the menu
        menu.addAction(separator)
        menu.addAction(self.copyAction)
        menu.addAction(self.pasteAction)
        menu.addAction(self.cutAction)
        # Launching the menu
        menu.exec(event.globalPos())

    def newFile(self):
        filename = QFileDialog.getOpenFileName(
            self, 'Open File', filter='*.ctf')
        path = filename[0]
        self.setCurrentFile(path)
        self.loadFile(path)
        # try:
        #     canvas.clear()
        #     # widgets.clear_widgets(widgets.w, 'all')
        #     self.clearLayout(layout)
        #     # widgets.w["lineEdit"][-1].setText(path)
        #     global data10, cols, rows, xStep, yStep, Eulers
        #     self.set_status('Reading Data...')
        #     Eulers, data10, cols, rows, xStep, yStep, self.title = ebsd.read_data(
        #         path)
        #     self.title = " ".join(self.title.replace('Prj', "").split())
        #     # widgets.w['dx'].append(xStep)
        #     # widgets.w['dy'].append(yStep)
        #     self.set_status('Ready', 3000)
        #     image = ebsd.rgb_img(Eulers)
        #     canvas.append(MatplotlibWidget('raw data'))
        #     canvas[-1].update_plot(image, xStp=xStep, scb=True)
        #     layout.addWidget(canvas[-1])
        #     # widgets.w['canvas']['canvas1'].append(
        #     # widgets.w['canvas']['canvas1'][-1].update_plot(image, xStp=xStep)
        #     # self.setCentralWidget(widgets.w['canvas']['canvas1'][-1])
        # except FileNotFoundError:
        #     self.set_status('File Not Found!!!', 3000)
        # pass

    def loadFile(self, fname):
        try:
            canvas.clear()
            # widgets.clear_widgets(widgets.w, 'all')
            self.clearLayout(layout)
            # widgets.w["lineEdit"][-1].setText(path)
            global data10, cols, rows, xStep, yStep, Eulers
            self.set_status('Reading Data...')
            Eulers, data10, cols, rows, xStep, yStep, self.title = ebsd.read_data(
                fname)
            self.title = " ".join(self.title.replace('Prj', "").split())
            # widgets.w['dx'].append(xStep)
            # widgets.w['dy'].append(yStep)
            self.set_status('Ready', 3000)
            image = ebsd.rgb_img(Eulers)
            canvas.append(MatplotlibWidget('raw data'))
            canvas[-1].update_plot(image, xStp=xStep, scb=True)
            layout.addWidget(canvas[-1])
            # widgets.w['canvas']['canvas1'].append(
            # widgets.w['canvas']['canvas1'][-1].update_plot(image, xStp=xStep)
            # self.setCentralWidget(widgets.w['canvas']['canvas1'][-1])
        except FileNotFoundError:
            self.set_status('File Not Found!!!', 3000)
        pass

    def openFile(self):
        self.newFile()

    def populateOpenRecent(self):
        self.updateRecentFileActions()

    def saveFile(self):
        try:
            self.grain_info_win.save_data_as_csv()
        except AttributeError:
            pass
        except NameError:
            pass

    def copyContent(self):
        pass

    def pasteContent(self):
        pass

    def cutContent(self):
        pass

    def helpContent(self):
        pass

    def about(self):
        import platform
        import os
        from PyQt5.QtCore import QT_VERSION_STR
        arc = ('x64' if platform.architecture()[0] == '64bit' else 'x86')
        msg1 = f'\nVersion : {ebsd.ebsd.__version__}'
        msg2 = f'\nDeveloped for Scientific Purpose | PyQt5 {QT_VERSION_STR}'
        msg3 = '\nCredits:  Dr. Anish Kumar & Harini T'
        msg4 = f'\nOS: {platform.system()}_{os.name.upper()} {arc} {platform.release()}'
        button = QMessageBox.information(
            self, 'About', msg1+msg2+msg3+msg4)
        if button == QMessageBox.Ok:
            pass

    def openRecentFile(self):
        action = self.sender()
        if action:
            self.loadFile(action.data())

    def stats(self):
        pltname_list = ['raw data', 'medfilter', 'watershed',
                        'kamseg', 'kammap', 'grodmap', 'ddmap', 'pfmap', 'ipfmap']
        try:
            if canvas[-1].pltname in pltname_list:
                print(canvas[-1].pltname)
        except IndexError:
            pass

    def median_filter(self):
        global Eulers
        self._filters = {'3 x 3': 3, '5 x 5': 5,
                         '7 x 7': 7, '9 x 9': 9, '11 x 11': 11}

        try:
            # if not canvas[-1].cid:
            #     widgets.w['canvas']['canvas1'][-1].sc.mpl_disconnect(
            #         widgets.w['canvas']['canvas1'][-1].cid)

            dlg = CustomDialogFilter(self)
            if dlg.exec_():
                if dlg.selected:
                    self.clearLayout(layout)
                    w = self._filters[dlg.selected]
                    Eulers = ebsd.medianFilter(Eulers, data10, w)
                    self.set_status('Filtering...', 1000)
                    image = ebsd.rgb_img(Eulers)
                    canvas.append(MatplotlibWidget('medfilter'))
                    layout.addWidget(canvas[-1])
                    canvas[-1].update_plot(
                        image, xStp=xStep, scb=True)
            else:
                print("Cancel!")
        except NameError:
            self.set_status('Name Error')

    def Kam_map(self):
        global kam_im, Eulers
        global Eulers
        try:

            self.set_status('Calculating KAM Values...')
            kam_im = ebsd.kam(Eulers)
            self.set_status('Ready', 3000)
            self.clearLayout(layout)
            canvas.append(MatplotlibWidget('kammap'))
            canvas[-1].update_plot(
                kam_im, xStp=xStep, cax=True, scb=True)
            layout.addWidget(canvas[-1])

        except NameError:
            self.set_status('Name Error')

    def dd_map(self):
        global dd_im, Eulers, bv
        try:
            bv = float(self.win_settings.getBValue())
            self.set_status('Calculating dislocation density...')
            dd_im = ebsd.dislocation_density_map(kam_im, xStep, bv)
            self.set_status('Ready', 3000)
            self.clearLayout(layout)
            canvas.append(MatplotlibWidget('ddmap'))
            layout.addWidget(canvas[-1])
            canvas[-1].update_plot(
                dd_im, xStp=xStep, cax=True, scb=True)

        except NameError:
            self.set_status('Name Error')

    def ipf_map(self):
        global ipf_im, Eulers
        try:
            if not angles_clr:
                key = 5.
            else:
                key = min(sorted(angles_clr.keys()))
            _, klabels, ngrains = ebsd.kam_segmentation(Eulers, key)
            # im = ebsd.ipfmap(Eulers)
            # ipfdc = MplCanvas.MatplotlibWidget()
            # widgets.w['canvas']['canvas1'].append(ipfdc)
            # self.setCentralWidget(widgets.w['canvas']['canvas1'][-1])
            # widgets.w['canvas']['canvas1'][-1].update_plot(
            #     im, Eulers=Eulers, xStp=xstep, yStp=ystep, w_img=im, click=True, labels=labels)
            ipfch = str(self.win_settings.getIPF())
            self.set_status('Generating IPF Map...')
            ipf_im = ebsd.ipfmap(Eulers, ipfch)

            self.clearLayout(layout)
            ipfdc = MatplotlibWidget('ipfmap')
            canvas.append(ipfdc)
            canvas[-1].update_plot(
                ipf_im, Eulers=Eulers, xStp=xStep, yStp=yStep, w_img=ipf_im, click=True, labels=w['recSegment'][-1], scb=True, ipfleg=True)
            layout.addWidget(canvas[-1])
            self.show_cube()
            # ipfleg = MatplotlibWidget('ipfleg')
            # ipf_legend(ipfleg.sc.axes)
            # self.tabs.addTab(ipfleg, 'ipf legend')
            # layout.addWidget(ipfleg)
            # widgets.w['canvas']['canvas1'][-1].update_plot(ipf_im, xStp=xStep)
            self.set_status('Ready', 3000)
        except IndexError:
            self.set_status('Segment the Grains...')
        except NameError:
            self.set_status('Name Error')

    def show_legend(self):
        pass

    def show_cube(self):
        self.cube_win = CubeWindow()
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)
        self.tabs.addTab(w['canvas']['canvas3d'][-1], 'cube')
        # boxy.addWidget(w['canvas']['canvas3d'][-1])
        # layout.addWidget(w['canvas']['canvas3d'][-1])

    def segmentation(self):
        ch = self.win_settings.getSegmentation()
        if ch == 'KAM Based':
            self.kam_segment()
        elif ch == 'Watershed':
            print('wshed')
            self.watershed_seg()

        elif ch == 'Felzenszwalb Segmentation':
            self.felzenszwalb_seg()

    def watershed_seg(self):
        global wsh_im, Eulers
        try:
            multi, binary = ebsd.find_edges(Eulers[0, :, :])
            wsh_im, labels, n = ebsd.watersheding(multi)
            wcanvas = MatplotlibWidget('watershed')
            self.clearLayout(layout)
            canvas.append(wcanvas)
            layout.addWidget(canvas[-1])
            self.show_cube()
            w['recSegment'].append(labels)
            canvas[-1].update_plot(
                wsh_im, None, True, Eulers, labels, wsh_im, xStep, yStep, gcolor=[1, 1, 1], gbcolor=[0, 0, 0], scb=True)
            # mean0, mean1, mean2 = widgets.w['canvas']['canvas1'][-1].get_mean_angs()
            # print(mean0, mean1, mean2)
            # widgets.w['canvas']['canvas3d'][-1].update_cube(
            #     mean0, mean1, mean2)
            # widgets.w['canvas']['canvas3d'].append(MplCanvas.MplCanvas3d())
            self.wcLabel = QLabel(f"{n} Grains")
            self.set_status(f"{n} Grains")
        except NameError:
            self.set_status('Name Error')

    def clearLayout(self, layout):
        if layout is not None:
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()
                else:
                    self.clearLayout(item.layout())

    def kam_segment(self):
        global kseg_img, Eulers, kamB
        try:
            global angles_clr, min_lbs, min_grns
            angles_clr = {}
            min_lbs = None
            kamB = np.zeros((rows, cols))
            if self.kseg_win.exec_():
                if self.kseg_win.isSet():
                    for items in w['Binstances']:
                        print(items.getAngle(), items.getColor())
                        angles_clr[float(items.getAngle())] = items.getColor()
                cnt = 0
                for key in sorted(angles_clr.keys()):
                    cnt += 1
                    kseg_img, grain_ids, ngrains = ebsd.kam_segmentation(
                        Eulers, key)
                    kamB = mark_boundaries(
                        kamB, grain_ids, color=angles_clr[key])
                    if cnt == 1:
                        min_lbs = grain_ids
                        min_grns = ngrains
                w['recSegment'].append(min_lbs)
                self.clearLayout(layout)
                kseg_canvas = MatplotlibWidget('kamseg')
                canvas.append(kseg_canvas)
                layout.addWidget(canvas[-1])
                self.show_cube()
                canvas[-1].update_plot(
                    kamB, Eulers=Eulers, xStp=xStep, yStp=yStep, w_img=kamB, click=True, labels=min_lbs, gcolor=[1, 1, 1], gbcolor=[0, 0, 0], scb=True)

                # self.grain_info_win.update_data(
                #     widgets.w['canvas']['canvas1'][-1])
                # self.kLabel = QLabel(f"{ngrains} Grains")
                # self.statusbar.removeWidget(self.wcLabel)
                # self.statusbar.addWidget(self.kLabel)
                # self.set_status(f"{min_grns} Grains")
            else:
                pass
        except NameError:
            self.set_status('Name Error')

    def felzenszwalb_seg(self):
        self.set_status('Felzenzwalb segmentation not implemented yet')

    def pf_map(self):
        self.set_status('Preparing pole figure map...')
        global Eulers
        self._selOpt = {'PF For every grain click': 1,
                        'PF For selected region': 2,
                        'PF For Entire region (Consumes time)': 3}
        self.pfch = '{001}'
        try:
            dlg = PFSelect()
            if dlg.exec_():
                if self._selOpt[dlg.selected] == 3:  # and self.checkSegmentationDone()
                    self.pfch = str(self.win_settings.getPF())
                    pfX, pfY = ebsd.pf_map(Eulers, self.pfch)
                    w['pfpoints'].append(np.stack([pfX, pfY]))
                    print(len(pfX))
                    if w['canvas']['pf']:
                        widgets.w['canvas']['pf'][-1].sc.axes.cla()

                    pfplt = MatplotlibWidget('pfmap', pfmap=True)
                    w['canvas']['pf'].append(pfplt)

                    w['canvas']['pf'][-1].sc.axes.scatter(pfX, pfY)
                    w['canvas']['pf'][-1].sc.axes.set_title(
                        f'Stereographic Projection - {self.pfch}')
                    self.pfTab(w['canvas']['pf'][-1])
                    # MatplotlibWidget('pfmap', pfmap=True)
                    # w['canvas']['pf'][-1].sc.axes.scatter(pfX, pfY)
                    # vbox = QVBoxLayout( )
                    # vbox.addWidget(w['canvas']['pf'][-1])
                    # self.tabs.addTab(w['canvas']['pf'][-1], 'Pole Figure')
                    print(self.pfch)
                    # layout.addWidget(self.tabs)
                elif self._selOpt[dlg.selected] == 2:
                    if not selpoints:
                        self.select_area()
                        self.set_status('Select an area')

                    a1, a2, b1, b2 = selpoints[-4], selpoints[-3], selpoints[-2], selpoints[-1]
                    dupEulers = Eulers[:, b1:b2, a1:a2]
                    print(dupEulers)
                    self.pfch = str(self.win_settings.getPF())
                    # widgets.w['pfchoice'].append(pfch)
                    pfX, pfY = ebsd.pf_map(dupEulers, self.pfch)
                    w['pfpoints'].append(np.stack([pfX, pfY]))
                    print(len(pfX))
                    # if w['canvas']['pf']:
                    #     w['canvas']['pf'][-1].sc.axes.cla()
                    #     w['canvas']['pf'][-1].sc.axes.scatter(
                    #         pfX, pfY)
                    #     w['canvas']['pf'][-1].sc.axes.set_title(
                    #         f'Stereographic Projection - {self.pfch}')
                    #     self.pfTab(w['canvas']['pf'][-1])
                    # else:
                    spf = MatplotlibWidget('spfmap', pfmap=True)
                    w['canvas']['pf'].append(spf)
                    self.pfTab(w['canvas']['pf'][-1])
                    self.set_status('Ready')

                elif self._selOpt[dlg.selected] == 1:
                    if np.any(kamB):
                        w['canvas']['pf'].append(
                            MatplotlibWidget('gpf', pfmap=True))
                        self.pfch = str(self.win_settings.getPF())
                        w['pfchoice'].append(self.pfch)
                        w['canvas']['pf'][-1].sc.axes.set_title(
                            f'Stereographic Projection - {self.pfch}')
                        self.pfTab(w['canvas']['pf'][-1])
                        # widgets.w['canvas']['canvas1'][-1].update_plot(
                        #     kamB, Eulers=Eulers, xStp=xStep, yStp=yStep, w_img=kamB, click=True, labels=min_lbs, gcolor=[1, 1, 1], gbcolor=[0, 0, 0], scb=True)
                        w['canvas']['pf'][-1].pfupdate()

                else:
                    self.set_status('Grains not Segmented...')
        except:
            pass

    def pfTab(self, obj):
        self.tab2 = QWidget()
        self.tab2.layout = QVBoxLayout()
        self.tab2.layout.addWidget(obj)
        self.scale = QSlider()
        self.scale.setOrientation(QtCore.Qt.Horizontal)
        self.scale.setMinimum(1)
        self.scale.setMaximum(100)
        self.tab2.layout.addWidget(self.scale)
        self.tab2.setLayout(self.tab2.layout)
        if self.tabs.count() == 1:
            self.tabs.addTab(self.tab2, 'Pole Figure')
        else:
            self.tabs.removeTab(1)
            self.tabs.addTab(self.tab2, 'Pole Figure')

        self.scale.valueChanged.connect(self.update_pointsize)

    def update_pointsize(self, n):
        try:
            widgets.w['canvas']['pf'][-1].sc.axes.clear()
            widgets.w['canvas']['pf'][-1].sc.axes.set_aspect(1)
            draw_circle_frame(widgets.w['canvas']['pf'][-1].sc.axes)
            draw_wulff_net(widgets.w['canvas']['pf'][-1].sc.axes)
            widgets.w['canvas']['pf'][-1].sc.axes.scatter(
                w['pfpoints'][-1][0, :], w['pfpoints'][-1][1, :], s=n)
            w['canvas']['pf'][-1].sc.axes.set_title(
                f'Stereographic Projection - {self.pfch}')
            widgets.w['canvas']['pf'][-1].sc.draw()
        except AttributeError:
            pass

    def checkSegmentationDone(self):
        ionplts = ['kamseg', 'ipfmap', 'watershed', 'felzenswalb']
        if canvas is None:
            return False
        for c in canvas:
            print(c.pltname)
            if c.pltname in ionplts:
                return True
        return False

    def grod_map(self):
        global min_lbs, min_grns, Eulers
        try:
            grod_img = ebsd.grod(
                labels=min_lbs, ngrains=min_grns, Eulers=Eulers)
            self.clearLayout(layout)
            canvas.append(MatplotlibWidget('grodmap'))
            layout.addWidget(canvas[-1])
            canvas[-1].update_plot(
                grod_img, xStp=xStep, cax=True, scb=True)

        except NameError:
            self.set_status('Proceed with kam segmentation.')

    def grain_details(self):
        self.grain_info_win = GrainInfoWin()

    def settings_win(self):
        self.win_settings.show()

    def undoSel(self):
        if len(_nav_stack) > 1:
            canvas[-1].sc.axes._set_view(_nav_stack.pop())
            canvas[-1].sc.draw()
            self.rect_selector.set_active(False)
            for i in canvas:
                name = i.pltname
                if name == 'kamseg' or name == 'ipfmap' or name == 'watershed':
                    i.cid = i.sc.mpl_connect('button_press_event', i.onclick)
        elif len(_nav_stack) == 1:
            canvas[-1].sc.axes._set_view(_nav_stack[0])
            canvas[-1].sc.draw()
        else:
            layout.addWidget(canvas[-1])

    def select_area(self):
        self.view = canvas[-1].sc.axes._get_view()
        _nav_stack.append(self.view)
        for i in canvas:
            name = i.pltname
            if name == 'kamseg' or name == 'ipfmap' or name == 'watershed':
                i.sc.mpl_disconnect(i.cid)
        # , state_modifier_keys=self.state_modifier_keys)
        self.rect_selector = RectangleSelector(
            canvas[-1].sc.axes, self.onselect_function, interactive=False)

    def onselect_function(self, eclick, erelease):

        extent = self.rect_selector.extents
        print("Extents: ", extent)

        canvas[-1].sc.axes.set_xlim(extent[0], extent[1])
        canvas[-1].sc.axes.set_ylim(extent[3], extent[2])

        x1 = int(extent[0])
        x2 = int(extent[1])
        y1 = int(extent[2])
        y2 = int(extent[3])
        selpoints.extend([x1, x2, y1, y2])

    def _connectActions(self):
        # Connect File actions
        self.newAction.triggered.connect(self.newFile)
        self.openAction.triggered.connect(self.openFile)
        self.saveAction.triggered.connect(self.saveFile)
        self.exitAction.triggered.connect(self.close)
        # Connect Edit actions
        self.copyAction.triggered.connect(self.copyContent)
        self.pasteAction.triggered.connect(self.pasteContent)
        self.cutAction.triggered.connect(self.cutContent)
        # Connect Help actions
        self.helpContentAction.triggered.connect(self.helpContent)
        self.aboutAction.triggered.connect(self.about)
        # Connect Open Recent to dynamically populate it
        self.openRecentMenu.aboutToShow.connect(self.populateOpenRecent)
        # median filter
        self.filterAction.triggered.connect(self.median_filter)
        # kam
        self.kamAction.triggered.connect(self.Kam_map)
        # dislocation density
        self.ddAction.triggered.connect(self.dd_map)
        # watershed connect
        self.segementAction.triggered.connect(self.segmentation)
        # kam segment connect
        # self.ksegAction.triggered.connect(self.kam_segment)
        # pf
        self.pfAction.triggered.connect(self.pf_map)
        # ipf
        self.ipfAction.triggered.connect(self.ipf_map)
        self.ipfAction.triggered.connect(self.show_legend)
        # settings
        self.settings.triggered.connect(self.settings_win)
        # grain info
        self.info.triggered.connect(self.grain_details)
        # GROD
        self.grodAction.triggered.connect(self.grod_map)
        # rect selection
        self.rectselAction.triggered.connect(self.select_area)
        # undo rect selection
        self.undoAction.triggered.connect(self.undoSel)
        # plots
        self.plotsAction.triggered.connect(self.stats)

    def _createStatusBar(self, msg=None):
        self.statusbar = self.statusBar()
        # Adding a temporary message
        self.statusbar.showMessage("Ready", 3000)

    def set_status(self, msg, t=None):
        if msg:
            if t:
                self.statusbar.showMessage(msg, t)
            else:
                self.statusbar.showMessage(msg)

    def setCurrentFile(self, fileName):
        self.curFile = fileName
        if self.curFile:
            self.setWindowTitle("%s - Recent Files" %
                                self.strippedName(self.curFile))
        else:
            self.setWindowTitle("Recent Files")

        settings = QSettings('Trolltech', 'Recent Files Example')
        files = settings.value('recentFileList', [])

        try:
            files.remove(fileName)
        except ValueError:
            pass

        files.insert(0, fileName)
        del files[Window.MaxRecentFiles:]

        settings.setValue('recentFileList', files)

        for widget in QApplication.topLevelWidgets():
            if isinstance(widget, Window):
                widget.updateRecentFileActions()

    def updateRecentFileActions(self):
        settings = QSettings('Trolltech', 'Recent Files Example')
        files = settings.value('recentFileList', [])

        numRecentFiles = min(len(files), Window.MaxRecentFiles)

        for i in range(numRecentFiles):
            text = "&%d %s" % (i + 1, self.strippedName(files[i]))
            self.recentFileActs[i].setText(text)
            self.recentFileActs[i].setData(files[i])
            self.recentFileActs[i].setVisible(True)

        for j in range(numRecentFiles, Window.MaxRecentFiles):
            self.recentFileActs[j].setVisible(False)

        self.separatorAct.setVisible((numRecentFiles > 0))

    def strippedName(self, fullFileName):
        return QFileInfo(fullFileName).fileName()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = Window()
    # win.setFixedSize(1000, 600)
    win.showMaximized()
    sys.exit(app.exec_())
