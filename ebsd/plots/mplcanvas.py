import matplotlib.pyplot as plt
from . widgets import w
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
if QtCore.qVersion() >= "5.":
    from matplotlib.backends.backend_qt5agg import (
        FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
else:
    from matplotlib.backends.backend_qt4agg import (
        FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
import matplotlib as mpl
from skimage.measure import regionprops
import os
import numpy as np
import ebsd
from . scalebar import ScaleBar


class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)


class MatplotlibWidget(QtWidgets.QWidget):
    def __init__(self, name, pfmap=None):
        super().__init__()
        self.pltname = name
        layout = QVBoxLayout(self)
        # canvas gets created during init once
        self.sc = MplCanvas(self, width=5, height=4, dpi=100)
        self.toolbar = NavigationToolbar(self.sc, self)
        layout.addWidget(self.sc)
        layout.addWidget(self.toolbar)
        self.sc.draw()
        # widget = QWidget()
        # widget.setLayout(layout)
        # self.setCentralWidget(widget)
        self.mean0, self.mean1, self.mean2 = 0, 0, 0
        self.cid = None
        # set_tight_plt(self.sc.figure, self.sc.axes)
        if pfmap:
            self.sc.axes.set_aspect(1)
            self.sc.axes.axis('off')
            ebsd.draw_circle_frame(self.sc.axes)
            ebsd.draw_wulff_net(self.sc.axes)

    def update_plot(self, image, cmp=None, click=False, Eulers=None, labels=None, w_img=None, xStp=None, yStp=None, cax=False, scb=False, gcolor=None, gbcolor=None, ipfleg=None):
        self.sc.axes.cla()
        self.sc.axes.set_axis_off()
        if cax:
            im = self.sc.axes.imshow(image, cmap=cmp)
            self.show_cbar(image, im)
        else:
            self.sc.axes.imshow(image, cmap=cmp)
        if scb:
            sb = ScaleBar(
                xStp*10**-6, length_fraction=0.2, location=3, box_alpha=0.5)
            self.sc.axes.add_artist(sb)

        if ipfleg:
            ax2 = self.sc.fig.add_axes([0.9, 0.1, 0.1, 0.1])
            ebsd.ipf_legend(ax2)
        # or self.pltname != 'kammap' or self.pltname != 'ddmap':
        if self.pltname != 'ipfmap':
            set_tight_plt(self.sc.fig)

        self.sc.draw()
        if(click):
            self.cid = self.sc.mpl_connect('button_press_event', self.onclick)
        if type(Eulers) == np.ndarray:
            global els, labs, wsh_img, xstep, ystep, prop
            els = Eulers
            labs = labels
            self.ngrains = len(np.unique(labs))-1
            wsh_img = w_img
            xstep = xStp
            ystep = yStp
            prop = regionprops(labs)
            self.df = None
            if gcolor and gbcolor:
                global grainshade, grainboundshade
                grainshade = gcolor
                grainboundshade = gbcolor
            # self.data()
            self.data1()
            w['dfs'].append(self.getdf())

        # if selectarea:
        #     global selpoints
        #     selpoints = list()
        #     from matplotlib.widgets import RectangleSelector

        #     def onselect_function(eclick, erelease):

        #         extent = rect_selector.extents
        #         print("Extents: ", extent)

        #         self.sc.axes.set_xlim(extent[0], extent[1])
        #         self.sc.axes.set_ylim(extent[3], extent[2])

        #         x1 = int(extent[0])
        #         x2 = int(extent[1])
        #         y1 = int(extent[2])
        #         y2 = int(extent[3])
        #         selpoints.extend([x1, x2, y1, y2])
        #     rect_selector = RectangleSelector(
        #         self.sc.axes, onselect_function, button=[1], interactive=True)

    def cbar(self, fig, data):

        mi = np.min(data)
        mx = np.max(data)
        fig.subplots_adjust(right=0.85)
        #cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
        cbar_ax = fig.add_axes([0.96, 0.15, 0.02, 0.7])

        return cbar_ax

    def show_cbar(self, image, im):
        self.cb = self.cbar(self.sc.fig, image)
        self.sc.fig.colorbar(im, cax=self.cb)

    def onclick(self, event):
        x = event.xdata
        y = event.ydata
        global grain_no
        grain_no = labs[int(y)][int(x)]
        self.selected_grain(grain_no-1)
        if grainshade and grainboundshade:
            self.selected_grain(grain_no-1, grainshade=grainshade,
                                grainboundshade=grainboundshade)

        sel_grain_msg = f'Grain No: {grain_no-1} | Mean Eulers: e1 = {round(self.mean0,2)}, e2 = {round(self.mean1,2)}, e3 = {round(self.mean2,2)}'
        w['msgs'].append(sel_grain_msg)
        # self.get_mean_angs()
        try:
            if(isinstance(w['canvas']['canvas3d'][-1], MplCanvas3d)):
                w['canvas']['canvas3d'][-1].update_cube(
                    self.mean0, self.mean1, self.mean2)
                w['canvas']['pf'][-1].pfupdate()
        except IndexError:
            pass

    def selected_grain(self, z, grainshade=[0, 0, 0], grainboundshade=[1, 1, 1]):
        from skimage.segmentation import find_boundaries
        global mean0, mean1, mean2, prop
        c = 0
        if prop[z].area > 9:
            arr1 = prop[z].image
            x1, y1, x2, y2 = (prop[z].bbox)
            # self.sc.axes = self.sc.figure.add_axes([0.0, 0.0, 1.0, 1.0])
            gx, gy = np.where(arr1 == True)
            copy_watershed = np.copy(wsh_img)
            copy_watershed[gx+x1, gy+y1, :] = grainshade

            bound = find_boundaries(prop[z].image)
            bx, by = np.where(bound == True)
            copy_watershed[bx+x1, by+y1, :] = grainboundshade
            self.update_plot(copy_watershed)
            offx, offy, _, _ = (prop[z].bbox)
            gx, gy = np.where(prop[z].image == True)
            self.mean0 = np.mean(els[0, gx+offx, gy+offy])
            self.mean1 = np.mean(els[1, gx+offx, gy+offy])
            self.mean2 = np.mean(els[2, gx+offx, gy+offy])
        else:
            c += 1

    def get_mean_angs(self):
        return self.mean0, self.mean1, self.mean2

    def data1(self):
        my_dict = {'Grain label': [],
                   'Ellipse Area': [],
                   'Area': [],
                   'equivalent_diameter': [],
                   'Xcg': [], 'Ycg': [],
                   'Aspect Ratio': [],
                   'Mean E1': [], 'Mean E2': [], 'Mean E3': [],
                   'Orientation': []
                   }
        pixels2um = xstep*ystep

        for i in range(self.ngrains):
            my_dict['Grain label'].append(i+1)
            a = prop[i].major_axis_length
            b = prop[i].minor_axis_length
            my_dict['Ellipse Area'].append(np.pi*a*b)
            my_dict['Area'].append(prop[i].area)  # area in um
            my_dict['equivalent_diameter'].append(prop[i].equivalent_diameter *
                                                  xstep)  # diameter in um
            my_dict['Xcg'].append(prop[i].centroid[1])  # Xcg
            my_dict['Ycg'].append(prop[i].centroid[0])  # Ycg
            # for aspect ratio
            try:
                aspratio = (a)/(b)
                my_dict['Aspect Ratio'].append(aspratio)
            except ZeroDivisionError:
                pass
                my_dict['Aspect Ratio'].append("inf")
            # for mean Eulers
            arr1 = prop[i].image
            offx, offy, _, _ = (prop[i].bbox)
            gx, gy = np.where(arr1 == True)
            mean0 = np.mean(els[0, gx+offx, gy+offy])
            mean1 = np.mean(els[1, gx+offx, gy+offy])
            mean2 = np.mean(els[2, gx+offx, gy+offy])
            my_dict['Mean E1'].append(mean0)
            my_dict['Mean E2'].append(mean1)
            my_dict['Mean E3'].append(mean2)
            my_dict['Orientation'].append(prop[i].orientation)

        import pandas as pd
        self.df = pd.DataFrame(my_dict)

    def getdf(self):
        try:
            # w['dfs'].append(self.df)
            return self.df
        except NameError:
            pass

    def pfupdate(self):
        pole = {'{100}': ebsd.stereographicProjection_100,
                '{010}': ebsd.stereographicProjection_010,
                '{001}': ebsd.stereographicProjection_001}
        pch = w['pfchoice'][-1]
        print(pch)
        self.sc.axes.cla()
        ebsd.draw_circle_frame(self.sc.axes)
        ebsd.draw_wulff_net(self.sc.axes)
        func = pole[pch]
        pf1, pf2, pf3 = func(qx, qy, qz)
        self.sc.axes.set_title(f'Stereographic Projection - {pch}')
        self.sc.axes.scatter(pf1[0], pf1[1], label='n1', color='red')
        self.sc.axes.scatter(
            pf2[0], pf2[1], label='n2', color='green')
        self.sc.axes.scatter(
            pf3[0], pf3[1], label='n3', color='blue')
        self.sc.draw()
    # def show(self):
    #     self.sc.axes.set_xlim(0, els.shape[1])
    #     self.sc.axes.set_ylim(0, els.shape[2])


class MplCanvas3d(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.fig = Figure()
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.azim = 41  # 45
        self.ax.elev = 19  # 15
        # self.ax.set_axis_off()
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.canvas)
        self.toolbar = NavigationToolbar(self.canvas, self)
        layout.addWidget(self.toolbar)
        x, y, z = ebsd.get_cube()
        # x, y, z = 2*x, 2*y, 2*z
        self.Xref, self.Yref, self.Zref = ebsd.get_quivers()
        self.show_cube(x, y, z, self.Xref, self.Yref, self.Zref, 0, 0, 1)
        self.canvas.draw()

    def show_cube(self, x, y, z, qx, qy, qz, red, green, blue):
        a, b, c = 1.5, 1.5, 1.5
        self.ax.cla()
        self.XYZ_ref()
        self.ax.set_xlim(-3.5, 3.5)
        self.ax.set_ylim(-3.5, 3.5)
        self.ax.set_zlim(-3.5, 3.5)
        set_tight_plt(self.canvas.figure, self.ax)
        self.ax.set_axis_off()
        self.ax.quiver(0, 0, 0, 3*qx[0], 3*qx[1],
                       3*qx[2], color='red', label='x')
        self.ax.quiver(0, 0, 0, 3*qy[0], 3*qy[1],
                       3*qy[2], color='green', label='y')
        self.ax.quiver(0, 0, 0, 3*qz[0], 3*qz[1], 3*qz[2], label='z')
        self.ax.plot_surface(x*a, y*b, z*c, color=(red, green, blue))
        self.canvas.draw()

    def update_cube(self, e1, e2, e3):
        red, green, blue = abs(e1)/360, abs(e2)/90, abs(e3)/90
        x, y, z = ebsd.get_cube()
        global qx, qy, qz
        qx, qy, qz = ebsd.get_quivers()
        m = ebsd.get_m(x, y, z)
        m, qx, qy, qz = ebsd.rotation(m, e1, e2, e3, qx, qy, qz)
        m = m.reshape((3, 5, 5))
        x, y, z = m[0, :, :], m[1, :, :], m[2, :, :]
        self.show_cube(x, y, z, qx, qy, qz, red, green, blue)

    def XYZ_ref(self):
        self.ax.quiver(
            0, 0, 0, 2*self.Xref[0], 2*self.Xref[1], 2*self.Xref[2], color='black', label='x')
        self.ax.quiver(
            0, 0, 0, 2*self.Yref[0], 2*self.Yref[1], 2*self.Yref[2], color='black', label='y')
        self.ax.quiver(
            0, 0, 0, 2*self.Zref[0], 2*self.Zref[1], 2*self.Zref[2], color='black', label='z')


global zoompoints
zoompoints = list()


class NavigationToolbar(NavigationToolbar):
    # toolitems = (
    #     ('Home', 'Reset original view', 'home', 'home'),
    #     ('Back', 'Back to previous view', 'back', 'back'),
    #     ('Forward', 'Forward to next view', 'forward', 'forward'),
    #     (None, None, None, None),
    #     ('Pan',
    #      'Left button pans, Right button zooms\n'
    #      'x/y fixes axis, CTRL fixes aspect',
    #      'move', 'pan'),
    #     ('Zoom', 'Zoom to rectangle\nx/y fixes axis, CTRL fixes aspect',
    #      'zoom_to_rect', 'zoom'),
    #     ('Subplots', 'Configure subplots', 'subplots', 'configure_subplots'),
    #     (None, None, None, None),
    #     ('Save', 'Save the figure', 'filesave', 'save_figure'),
    #   )

    def save_figure(self, *args):
        # plt.imsave('dumm.png',image)
        filetypes = self.canvas.get_supported_filetypes_grouped()
        sorted_filetypes = sorted(filetypes.items())
        default_filetype = self.canvas.get_default_filetype()

        startpath = os.path.expanduser(mpl.rcParams['savefig.directory'])
        start = os.path.join(startpath, self.canvas.get_default_filename())
        filters = []
        selectedFilter = None
        for name, exts in sorted_filetypes:
            exts_list = " ".join(['*.%s' % ext for ext in exts])
            filter = '%s (%s)' % (name, exts_list)
            if default_filetype in exts:
                selectedFilter = filter
            filters.append(filter)
        filters = ';;'.join(filters)

        fname, filter = QtWidgets.QFileDialog.getSaveFileName(
            self.canvas.parent(), "Choose a filename to save to", start,
            filters, selectedFilter)
        if fname:
            # Save dir for next time, unless empty str (i.e., use cwd).
            if startpath != "":
                mpl.rcParams['savefig.directory'] = os.path.dirname(fname)
            try:
                # import matplotlib.pyplot as plt
                # plt.imsave(fname,plt.gcf())
                extent = self.canvas.axes.get_window_extent().transformed(
                    self.canvas.figure.dpi_scale_trans.inverted())
                self.canvas.figure.savefig(
                    fname, bbox_inches=extent, pad_inches=0, dpi=100)
            except Exception as e:
                QtWidgets.QMessageBox.critical(
                    self, "Error saving file", str(e),
                    QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.NoButton)

    # def drag_zoom(self, event):
    #     """Callback for dragging in zoom mode."""
    #     start_xy = self._zoom_info.start_xy
    #     ax = self._zoom_info.axes[0]
    #     (x1, y1), (x2, y2) = np.clip(
    #         [start_xy, [event.x, event.y]], ax.bbox.min, ax.bbox.max)
    #     if event.key == "x":
    #         y1, y2 = ax.bbox.intervaly
    #     elif event.key == "y":
    #         x1, x2 = ax.bbox.intervalx
    #     self.draw_rubberband(event, x1, y1, x2, y2)
    #     x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    #     zoompoints.append(x1)
    #     zoompoints.append(y1)
    #     zoompoints.append(x2)
    #     zoompoints.append(y2)
    #     print(zoompoints)


def set_tight_plt(fig=None, ax=None):
    # if fig == None:
    #     fig = plt.gcf()
    # if ax == None:
    #     ax = plt.gca()
    fig.subplots_adjust(top=1, bottom=0, right=1, left=0)
    # ax.xaxis.set_major_locator(plt.NullLocator())
    # ax.yaxis.set_major_locator(plt.NullLocator())


# class window(QMainWindow):
#     def __init__(self, parent=None):
#         super().__init__(parent)
#         obj = MatplotlibWidget()
#         obj.update_plot(plt.imread('output.png'))
#         self.setCentralWidget(obj)

    # if __name__ == '__main__':
    #     import sys
    #     app = QApplication(sys.argv)
    #     win = window()
    #     # win.setFixedSize(1000, 600)
    #     win.showMaximized()
    #     sys.exit(app.exec_())
