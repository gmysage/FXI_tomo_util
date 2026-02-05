import sys
import h5py
import matplotlib.pyplot as plt
import numpy as np
import json
from PyQt5 import QtGui, QtCore
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QIntValidator, QDoubleValidator, QFont, QColor
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib as mpl
from image_util import plot3D

mpl.use('Qt5Agg')
#mpl.use('qtagg') # automaticall select Qt5Agg or PySide2
from skimage import io
from copy import deepcopy
from mpl_toolkits.axes_grid1 import make_axes_locatable

from pystackreg import StackReg
from scipy.signal import medfilt2d
from recon_process import *
from util import *

try:
    from torch_util import *
    torch_installed = 1
except:
    torch_installed = 0

try:
    import napari
    exist_napari = True
except:
    exist_napari = False

  

global pytomo


class App(QWidget):
    def __init__(self):
        super().__init__()
        self.title = 'Tomo utils'
        screen_resolution = QApplication.desktop().screenGeometry()
        width, height = screen_resolution.width(), screen_resolution.height()
        self.width = 1000
        self.height = 900
        self.left = (width - self.width) // 2
        self.top = (height - self.height) // 2
        self._err_color = QtCore.Qt.red
        self.check_torch_installed()
        self.initUI()
        self.default_layout()

    def check_torch_installed(self):
        global torch_installed
        self.torch_installed = torch_installed

    def closeEvent(self, *args, **kwargs):
        super().closeEvent(*args, **kwargs)
        print('close window')
        plt.close('all')

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.font1 = QtGui.QFont('Arial', 11, QtGui.QFont.Bold)
        self.font2 = QtGui.QFont('Arial', 11, QtGui.QFont.Normal)
        self.gui_fpath = __file__
        try:
            self.gui_fpath = os.readlink(self.gui_fpath)
        except:
            pass
        self.fpath = '/'.join(self.gui_fpath.split('/')[:-1])
        if self.torch_installed:
            try:
                self.gpu_count = torch.cuda.device_count()
            except:
                self.gpu_count = 0
        else:
            self.gpu_count = 0

        grid1 = QGridLayout()
        gpbox_prep = self.layout_GP_prepare()
        gpbox_tomo = self.layout_tomo()

        grid1.addWidget(gpbox_prep, 0, 1)
        #grid1.addLayout(gpbox_msg, 1, 1)
        grid1.addWidget(gpbox_tomo, 1, 1)

        layout1 = QVBoxLayout()
        layout1.addLayout(grid1)
        layout1.addWidget(QLabel())

        tabs = QTabWidget()
        tab1 = QWidget()

        tab1.setLayout(layout1)

        tabs.addTab(tab1, 'Rotation center')
        self.layout = QVBoxLayout(self)
        self.layout.addWidget(tabs)
        self.setLayout(self.layout)

    def default_layout(self):
        s = (1, 100, 100)
        self.file_loaded = []
        #self.tomo_file = {}
        self.img_prj = np.ones(s)
        self.img_dark = np.zeros(s)
        self.img_dark_avg = np.zeros(s)
        self.img_flat = np.ones(s)
        self.img_flat_avg = np.ones(s)
        self.img_rc = np.zeros(s)
        self.img_recon_slice = []
        self.img_recon_slice_crop = []
        self.img_angle = []
        self.img_eng = ''
        self.img_sid = ''
        self.msg = ''
        self.flag_rc = False
        self.rc = ''
        self.fn_rc_tmp = ''
        self.file_loaded = []
        self.fname_rc = {}
        self.fname_rc_batch = {}
        self.current_file = ''
        self.current_file_short = ''
        self.enable_multi_selection()
        self.slider = []
        self.ml_model_path = f'{self.fpath}/pre_traind_model_xanes_denoise.pth'
        self.ml_model_recon_path = f'{self.fpath}/model_tomo_best_psnr.pth'
        self.ml_model_path_default = self.ml_model_path
        self.ml_model_recon_path_default = self.ml_model_recon_path

        if torch_installed:
            self.tx_ml_model.setText(self.ml_model_path_default)
            self.tx_ml_model_path.setText(self.ml_model_recon_path_default)

    def layout_GP_prepare(self):
        lb_empty = QLabel()
        lb_empty.setFixedWidth(10)
        lb_ld = QLabel()
        lb_ld.setFont(self.font2)
        lb_ld.setText('File root folder:')
        lb_ld.setFixedWidth(100)

        self.tx_prj_path = QLineEdit()
        self.tx_prj_path.setFixedWidth(450)
        self.tx_prj_path.setFont(self.font2)
        self.tx_prj_path.setStyleSheet('background: rgb(240, 240, 220);')

        self.pb_open_prj = QPushButton('Open (.h5)')
        self.pb_open_prj.setFixedWidth(110)
        self.pb_open_prj.setFont(self.font2)
        self.pb_open_prj.clicked.connect(self.open_prj_file)

        self.pb_refresh_prj = QPushButton('Reload current folder')
        self.pb_refresh_prj.setFixedWidth(170)
        self.pb_refresh_prj.setFont(self.font2)
        self.pb_refresh_prj.clicked.connect(self.refresh_file_folder)

        lb_prefix = QLabel()
        lb_prefix.setFont(self.font2)
        lb_prefix.setText('file_prefix:')
        lb_prefix.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        lb_prefix.setFixedWidth(100)

        self.tx_prefix = QLineEdit()
        self.tx_prefix.setFixedWidth(60)
        self.tx_prefix.setText('fly')
        self.tx_prefix.setFont(self.font2)

        lb_msg = QLabel()
        lb_msg.setFont(self.font1)
        lb_msg.setStyleSheet('color: rgb(200, 50, 50);')
        lb_msg.setText('Message:')

        self.lb_msg = QLabel()
        self.lb_msg.setFont(self.font2)
        self.lb_msg.setStyleSheet('color: rgb(200, 50, 50);')

        hbox_msg = QHBoxLayout()
        hbox_msg.addWidget(lb_msg)
        hbox_msg.addWidget(self.lb_msg)
        hbox_msg.setAlignment(QtCore.Qt.AlignLeft)

        hbox_load = QHBoxLayout()
        hbox_load.addWidget(lb_ld)
        hbox_load.addWidget(self.tx_prj_path)
        hbox_load.addWidget(lb_prefix)
        hbox_load.addWidget(self.tx_prefix)
        hbox_load.addWidget(self.pb_open_prj)
        hbox_load.addWidget(self.pb_refresh_prj)
        hbox_load.addWidget(lb_empty)
        hbox_load.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)

        vbox = QVBoxLayout()
        vbox.addLayout(hbox_load)
        vbox.addLayout(hbox_msg)
        vbox.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)

        gpbox = QGroupBox('Load projection image  ')
        gpbox.setFont(self.font1)
        gpbox.setLayout(vbox)
        return gpbox

    def layout_rc_list(self):
        lb_empty1 = QLabel()
        lb_empty1.setFixedWidth(10)
        lb_empty2 = QLabel()
        lb_prj_file = QLabel()
        lb_prj_file.setText('Proj. files (FXI .h5)')
        lb_prj_file.setFont(self.font2)
        lb_prj_file.setFixedWidth(300)

        self.lst_prj_file = QListWidget()
        self.lst_prj_file.setFont(self.font2)
        self.lst_prj_file.setSelectionMode(QAbstractItemView.SingleSelection)
        self.lst_prj_file.setFixedWidth(350)
        self.lst_prj_file.setFixedHeight(200)
        '''
        self.pb_clear_list = QPushButton('Clear')
        self.pb_clear_list.setFixedWidth(80)
        self.pb_clear_list.setFont(self.font2)
        self.pb_clear_list.clicked.connect(self.clear_file_list)
        '''
        self.pb_h5_pre_set1 = QPushButton('preset: fly')
        self.pb_h5_pre_set1.setFixedWidth(170)
        self.pb_h5_pre_set1.setFont(self.font2)
        self.pb_h5_pre_set1.clicked.connect(lambda:self.preset_h5('fly'))

        self.pb_h5_pre_set2 = QPushButton('preset: zfly')
        self.pb_h5_pre_set2.setFont(self.font2)
        self.pb_h5_pre_set2.clicked.connect(lambda:self.preset_h5('zfly'))
        self.pb_h5_pre_set2.setFixedWidth(170)


        lb_h5_prj = QLabel()
        lb_h5_prj.setText('Attr: proj:')
        #lb_h5_prj.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        lb_h5_prj.setFixedWidth(85)
        lb_h5_prj.setFont(self.font2)

        self.tx_h5_prj = QLineEdit()
        self.tx_h5_prj.setFixedWidth(80)
        self.tx_h5_prj.setText('img_tomo')
        self.tx_h5_prj.setFont(self.font2)

        lb_h5_dark = QLabel()
        lb_h5_dark.setText('Attr: dark:')
        lb_h5_dark.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        lb_h5_dark.setFixedWidth(85)
        lb_h5_dark.setFont(self.font2)

        self.tx_h5_dark = QLineEdit()
        self.tx_h5_dark.setFixedWidth(80)
        self.tx_h5_dark.setText('img_dark')
        self.tx_h5_dark.setFont(self.font2)

        lb_h5_flat = QLabel()
        lb_h5_flat.setText('Attr: bkg:')
        lb_h5_flat.setFixedWidth(85)
        lb_h5_flat.setFont(self.font2)

        self.tx_h5_flat = QLineEdit()
        self.tx_h5_flat.setFixedWidth(80)
        self.tx_h5_flat.setText('img_bkg')
        self.tx_h5_flat.setFont(self.font2)

        lb_h5_ang = QLabel()
        lb_h5_ang.setText('Attr: angle:')
        lb_h5_ang.setFixedWidth(85)
        lb_h5_ang.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        lb_h5_ang.setFont(self.font2)

        self.tx_h5_ang = QLineEdit()
        self.tx_h5_ang.setFixedWidth(80)
        self.tx_h5_ang.setText('angle')
        self.tx_h5_ang.setFont(self.font2)

        lb_h5_xeng = QLabel()
        lb_h5_xeng.setText('Attr: XEng:')
        lb_h5_xeng.setFixedWidth(85)
        lb_h5_xeng.setFont(self.font2)

        self.tx_h5_xeng = QLineEdit()
        self.tx_h5_xeng.setFixedWidth(80)
        self.tx_h5_xeng.setText('X_eng')
        self.tx_h5_xeng.setFont(self.font2)

        lb_h5_sid = QLabel()
        lb_h5_sid.setText('Attr: Scan id:')
        lb_h5_sid.setFixedWidth(85)
        lb_h5_sid.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        lb_h5_sid.setFont(self.font2)

        self.tx_h5_sid = QLineEdit()
        self.tx_h5_sid.setFixedWidth(80)
        self.tx_h5_sid.setText('scan_id')
        self.tx_h5_sid.setFont(self.font2)

        self.pb_rc_view_prj = QPushButton('View proj')
        self.pb_rc_view_prj.setFont(self.font2)
        self.pb_rc_view_prj.clicked.connect(self.view_projection_images)
        self.pb_rc_view_prj.setFixedWidth(112)

        self.pb_rc_view_prj_1st = QPushButton('1st proj')
        self.pb_rc_view_prj_1st.setFont(self.font2)
        self.pb_rc_view_prj_1st.clicked.connect(self.view_projection_images_1st)
        self.pb_rc_view_prj_1st.setFixedWidth(112)

        self.pb_clear_list = QPushButton('Delete all')
        self.pb_clear_list.setFixedWidth(170)
        self.pb_clear_list.setFont(self.font2)
        self.pb_clear_list.clicked.connect(self.clear_file_list)

        self.pb_del_list_item = QPushButton('Delete')
        self.pb_del_list_item.setFixedWidth(170)
        self.pb_del_list_item.setFont(self.font2)
        self.pb_del_list_item.clicked.connect(self.del_list_item)

        self.pb_rec_view_copy = QPushButton('View recon')
        self.pb_rec_view_copy.setFont(self.font2)
        self.pb_rec_view_copy.clicked.connect(self.view_3D_recon)
        self.pb_rec_view_copy.setFixedWidth(112)

        hbox_h5_preset = QHBoxLayout()
        hbox_h5_preset.addWidget(self.pb_h5_pre_set1)
        hbox_h5_preset.addWidget(self.pb_h5_pre_set2)
        hbox_h5_preset.addStretch()
        hbox_h5_preset.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)

        hbox_rc_h51 = QHBoxLayout()
        hbox_rc_h51.addWidget(lb_h5_prj)
        hbox_rc_h51.addWidget(self.tx_h5_prj)
        hbox_rc_h51.addWidget(lb_h5_dark)
        hbox_rc_h51.addWidget(self.tx_h5_dark)
        hbox_rc_h51.addStretch()
        hbox_rc_h51.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)

        hbox_rc_h52 = QHBoxLayout()
        hbox_rc_h52.addWidget(lb_h5_flat)
        hbox_rc_h52.addWidget(self.tx_h5_flat)
        hbox_rc_h52.addWidget(lb_h5_ang)
        hbox_rc_h52.addWidget(self.tx_h5_ang)
        hbox_rc_h52.addStretch()
        hbox_rc_h52.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)

        hbox_rc_h53 = QHBoxLayout()
        hbox_rc_h53.addWidget(lb_h5_xeng)
        hbox_rc_h53.addWidget(self.tx_h5_xeng)
        hbox_rc_h53.addWidget(lb_h5_sid)
        hbox_rc_h53.addWidget(self.tx_h5_sid)
        hbox_rc_h53.addStretch()
        hbox_rc_h53.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)

        hbox_pb_view = QHBoxLayout()
        hbox_pb_view.addWidget(self.pb_rc_view_prj)
        hbox_pb_view.addWidget(self.pb_rc_view_prj_1st)
        hbox_pb_view.addWidget(self.pb_rec_view_copy)
        hbox_pb_view.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)

        hbox_pb_del = QHBoxLayout()
        hbox_pb_del.addWidget(self.pb_del_list_item)
        hbox_pb_del.addWidget(self.pb_clear_list)
        hbox_pb_del.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)

        vbox_file = QVBoxLayout()
        vbox_file.addWidget(lb_prj_file)
        vbox_file.addWidget(self.lst_prj_file)
        vbox_file.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)

        vbox = QVBoxLayout()
        vbox.addLayout(vbox_file)
        vbox.addLayout(hbox_h5_preset)
        vbox.addLayout(hbox_rc_h51)
        vbox.addLayout(hbox_rc_h52)
        vbox.addLayout(hbox_rc_h53)
        vbox.addLayout(hbox_pb_view)
        vbox.addLayout(hbox_pb_del)
        vbox.addWidget(lb_empty2)
        vbox.addStretch()
        return vbox

    def layout_rc_range(self):
        lb_empty = QLabel()

        lb_sli_id = QLabel()
        lb_sli_id.setText('slice id:')
        lb_sli_id.setFont(self.font2)
        lb_sli_id.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        lb_sli_id.setFixedWidth(85)

        self.tx_sli_id = QLineEdit()
        self.tx_sli_id.setText('0')
        self.tx_sli_id.setFont(self.font2)
        self.tx_sli_id.setFixedWidth(80)
        self.tx_sli_id.setValidator(QIntValidator())

        lb_rc_steps = QLabel()
        lb_rc_steps.setText('steps:')
        lb_rc_steps.setFixedWidth(85)
        lb_rc_steps.setFont(self.font2)

        self.tx_rc_steps = QLineEdit()
        self.tx_rc_steps.setText('60')
        self.tx_rc_steps.setFont(self.font2)
        self.tx_rc_steps.setFixedWidth(80)
        self.tx_rc_steps.setValidator(QIntValidator())

        lb_rc_start = QLabel()
        lb_rc_start.setText('start:')
        lb_rc_start.setFixedWidth(85)
        lb_rc_start.setFont(self.font2)

        self.tx_rc_start = QLineEdit()
        self.tx_rc_start.setText('0')
        self.tx_rc_start.setFont(self.font2)
        self.tx_rc_start.setFixedWidth(80)
        self.tx_rc_start.setValidator(QDoubleValidator())

        lb_rc_stop = QLabel()
        lb_rc_stop.setText('stop:')
        lb_rc_stop.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        lb_rc_stop.setFixedWidth(85)
        lb_rc_stop.setFont(self.font2)

        self.tx_rc_stop = QLineEdit()
        self.tx_rc_stop.setText('0')
        self.tx_rc_stop.setFont(self.font2)
        self.tx_rc_stop.setFixedWidth(80)
        self.tx_rc_stop.setValidator(QDoubleValidator())

        hbox_rc_start_stop = QHBoxLayout()
        hbox_rc_start_stop.addWidget(lb_rc_start)
        hbox_rc_start_stop.addWidget(self.tx_rc_start)
        hbox_rc_start_stop.addWidget(lb_rc_stop)
        hbox_rc_start_stop.addWidget(self.tx_rc_stop)
        hbox_rc_start_stop.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)

        hbox_rc_step_sli = QHBoxLayout()
        hbox_rc_step_sli.addWidget(lb_rc_steps)
        hbox_rc_step_sli.addWidget(self.tx_rc_steps)
        hbox_rc_step_sli.addWidget(lb_sli_id)
        hbox_rc_step_sli.addWidget(self.tx_sli_id)
        hbox_rc_step_sli.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)

        vbox = QVBoxLayout()
        vbox.addLayout(hbox_rc_start_stop)
        vbox.addLayout(hbox_rc_step_sli)
        vbox.addStretch()
        vbox.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)
        return vbox


    def layout_rc_param(self):
        lb_rc_block_list = QLabel()
        lb_rc_block_list.setText('block list:')
        lb_rc_block_list.setFont(self.font2)
        lb_rc_block_list.setFixedWidth(85)

        self.tx_rc_block_list = QLineEdit()
        self.tx_rc_block_list.setText('[]')
        self.tx_rc_block_list.setFont(self.font2)
        self.tx_rc_block_list.setFixedWidth(80)

        lb_rc_denoise = QLabel()
        lb_rc_denoise.setText('denoise flag:')
        lb_rc_denoise.setFont(self.font2)
        lb_rc_denoise.setFixedWidth(85)

        self.tx_rc_denoise = QLineEdit()
        self.tx_rc_denoise.setText('0')
        self.tx_rc_denoise.setFont(self.font2)
        self.tx_rc_denoise.setFixedWidth(80)
        self.tx_rc_denoise.setValidator(QIntValidator())

        lb_rc_dark_scale = QLabel()
        lb_rc_dark_scale.setText('dark scale:')
        lb_rc_dark_scale.setFont(self.font2)
        lb_rc_dark_scale.setFixedWidth(85)

        self.tx_rc_dark_scale = QLineEdit()
        self.tx_rc_dark_scale.setText('1')
        self.tx_rc_dark_scale.setFont(self.font2)
        self.tx_rc_dark_scale.setFixedWidth(80)
        self.tx_rc_dark_scale.setValidator(QDoubleValidator())

        lb_rc_algorithm = QLabel()
        lb_rc_algorithm.setText('Algorithm:')
        lb_rc_algorithm.setFont(self.font2)
        lb_rc_algorithm.setFixedWidth(85)
        lb_rc_algorithm.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)

        self.cb_rc_algorithm = QComboBox()
        self.cb_rc_algorithm.setFont(self.font2)
        self.cb_rc_algorithm.setFixedWidth(80)
        self.cb_rc_algorithm.addItem('gridrec')
        self.cb_rc_algorithm.addItem('astra_fbp')
        self.cb_rc_algorithm.addItem('astra_sirt')
    
        lb_rc_algorithm_iter = QLabel()
        lb_rc_algorithm_iter.setText('Iter:')
        lb_rc_algorithm_iter.setFont(self.font2)
        lb_rc_algorithm_iter.setFixedWidth(85)
        lb_rc_algorithm_iter.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)

        self.tx_rc_algorithm_iter = QLineEdit()
        self.tx_rc_algorithm_iter.setText('100')
        self.tx_rc_algorithm_iter.setFont(self.font2)
        self.tx_rc_algorithm_iter.setFixedWidth(80)
        self.tx_rc_algorithm_iter.setValidator(QIntValidator())

        lb_rc_rm_strip = QLabel()
        lb_rc_rm_strip.setText('ring remove:')
        lb_rc_rm_strip.setFont(self.font2)
        lb_rc_rm_strip.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        lb_rc_rm_strip.setFixedWidth(85)

        self.lb_rm_ring = QLabel()
        self.lb_rm_ring.setText('snr:')
        self.lb_rm_ring.setFixedWidth(85)
        self.lb_rm_ring.setFont(self.font2)
        self.lb_rm_ring.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)

        self.cb_rm_strip = QComboBox()
        self.cb_rm_strip.setFont(self.font2)
        self.cb_rm_strip.setFixedWidth(80)
        self.cb_rm_strip.addItem('wavelet')
        self.cb_rm_strip.addItem('all-stripe')
        self.cb_rm_strip.currentIndexChanged.connect(self.ring_remove)

        lb_rc_snr = QLabel()
        lb_rc_snr.setText('stripe snr:')
        lb_rc_snr.setFont(self.font2)
        lb_rc_snr.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        lb_rc_snr.setFixedWidth(85)

        self.tx_rc_ring_remove = QLineEdit()
        self.tx_rc_ring_remove.setText('4')
        self.tx_rc_ring_remove.setFont(self.font2)
        self.tx_rc_ring_remove.setFixedWidth(80)
        self.tx_rc_ring_remove.setValidator(QDoubleValidator())

        hbox1 = QHBoxLayout()
        hbox1.addWidget(lb_rc_block_list)
        hbox1.addWidget(self.tx_rc_block_list)
        hbox1.addWidget(lb_rc_denoise)
        hbox1.addWidget(self.tx_rc_denoise)
        hbox1.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)

        hbox2 = QHBoxLayout()
        hbox2.addWidget(lb_rc_dark_scale)
        hbox2.addWidget(self.tx_rc_dark_scale)
        hbox2.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)

        hbox3 = QHBoxLayout()
        hbox3.addWidget(lb_rc_rm_strip)
        hbox3.addWidget(self.cb_rm_strip)
        hbox3.addWidget(self.lb_rm_ring)
        hbox3.addWidget(self.tx_rc_ring_remove)
        hbox3.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)

        hbox4 = QHBoxLayout()
        hbox4.addWidget(lb_rc_algorithm)
        hbox4.addWidget(self.cb_rc_algorithm)
        hbox4.addWidget(lb_rc_algorithm_iter)
        hbox4.addWidget(self.tx_rc_algorithm_iter)
        hbox4.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)

        vbox = QVBoxLayout()
        vbox.addLayout(hbox1)
        vbox.addLayout(hbox2)
        vbox.addLayout(hbox3)
        vbox.addLayout(hbox4)
        vbox.setAlignment(QtCore.Qt.AlignTop)
        return vbox

    def layout_rc_exec(self):
        lb_empty1 = QLabel()
        lb_empty1.setFixedHeight(5)
        lb_empty2 = QLabel()
        lb_empty3 = QLabel()
        lb_empty3.setFixedWidth(10)
        self.pb_rc_find = QPushButton('Find center')
        self.pb_rc_find.setFixedWidth(110)
        self.pb_rc_find.setFixedHeight(40)
        self.pb_rc_find.setFont(self.font1)
        self.pb_rc_find.setStyleSheet('color: rgb(50, 50, 250);')
        self.pb_rc_find.clicked.connect(self.find_rotation_center)
        #self.pb_rc_find.clicked.connect(self.click_find_rc)

        self.pb_rc_find_next = QPushButton('Next')
        self.pb_rc_find_next.setFixedWidth(115)
        self.pb_rc_find_next.setFixedHeight(40)
        self.pb_rc_find_next.setFont(self.font1)
        self.pb_rc_find_next.setStyleSheet('color: rgb(50, 50, 250);')
        self.pb_rc_find_next.clicked.connect(self.find_rotation_center_next)

        self.pb_rc_confirm = QPushButton('Confirm')
        self.pb_rc_confirm.setFixedWidth(110)
        self.pb_rc_confirm.setFixedHeight(40)
        self.pb_rc_confirm.setFont(self.font1)
        self.pb_rc_confirm.setStyleSheet('color: rgb(50, 50, 250);')
        self.pb_rc_confirm.clicked.connect(self.confirm_rotation_center)

        self.pb_rc_auto_batch = QPushButton('Auto center')
        self.pb_rc_auto_batch.setFixedWidth(110)
        self.pb_rc_auto_batch.setFixedHeight(40)
        self.pb_rc_auto_batch.setFont(self.font2)
        self.pb_rc_auto_batch.clicked.connect(self.auto_batch_rotation_center)

        self.chkbox_new_file = QCheckBox('new file only')
        self.chkbox_new_file.setFont(self.font2)
        self.chkbox_new_file.setFixedWidth(110)
        self.chkbox_new_file.setChecked(False)

        self.pb_rc_1_sli = QPushButton('recon 1 slice')
        self.pb_rc_1_sli.setFixedWidth(115)
        self.pb_rc_1_sli.setFixedHeight(40)
        self.pb_rc_1_sli.setFont(self.font2)
        self.pb_rc_1_sli.clicked.connect(self.recon_1_slice)

        self.pb_rc_save = QPushButton('Save RC')
        self.pb_rc_save.setFixedWidth(110)
        self.pb_rc_save.setFont(self.font2)
        self.pb_rc_save.clicked.connect(self.save_rotation_center)

        self.pb_rc_load = QPushButton('Load RC')
        self.pb_rc_load.setFixedWidth(110)
        self.pb_rc_load.setFont(self.font2)
        self.pb_rc_load.clicked.connect(self.load_rotation_center)

        lb_rc_mod = QLabel()
        lb_rc_mod.setText('Select & update rotation center  ')
        lb_rc_mod.setFont(self.font2)
        lb_rc_mod.setFixedWidth(250)

        self.tx_rc_mod = QLineEdit()
        self.tx_rc_mod.setFixedWidth(65)
        self.tx_rc_mod.setFont(self.font2)

        self.pb_rc_mod = QPushButton('U')
        self.pb_rc_mod.setFixedWidth(40)
        self.pb_rc_mod.setFont(self.font2)
        self.pb_rc_mod.clicked.connect(self.update_rotation_center)

        hbox_rc_find = QHBoxLayout()
        hbox_rc_find.addWidget(self.pb_rc_find)
        hbox_rc_find.addWidget(self.pb_rc_confirm)
        hbox_rc_find.addWidget(self.pb_rc_find_next)
        hbox_rc_find.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)

        hbox_rc_mod1 = QHBoxLayout()
        hbox_rc_mod1.addWidget(self.tx_rc_mod)
        hbox_rc_mod1.addWidget(self.pb_rc_mod)
        hbox_rc_mod1.addWidget(self.pb_rc_save)
        hbox_rc_mod1.addWidget(self.pb_rc_load)
        hbox_rc_mod1.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)

        vbox_rc_mod = QVBoxLayout()
        vbox_rc_mod.addWidget(lb_rc_mod)
        vbox_rc_mod.addLayout(hbox_rc_mod1)
        vbox_rc_mod.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)

        hbox_rc_auto = QHBoxLayout()
        hbox_rc_auto.addWidget(self.pb_rc_auto_batch)
        #hbox_rc_auto.addWidget(lb_empty3)
        hbox_rc_auto.addWidget(self.chkbox_new_file)
        hbox_rc_auto.addWidget(self.pb_rc_1_sli)
        hbox_rc_auto.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)

        vbox = QVBoxLayout()
        vbox.addLayout(hbox_rc_auto)
        vbox.addLayout(hbox_rc_find)
        vbox.addWidget(lb_empty1)
        vbox.addLayout(vbox_rc_mod)
        vbox.addWidget(lb_empty2)
        vbox.addStretch()
        vbox.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)
        return vbox

    def layout_recon(self):
        lb_empty = QLabel()
        lb_empty1 = QLabel()
        lb_empty1.setFixedHeight(10)
        lb_empty2 = QLabel()
        lb_empty2.setFixedWidth(10)

        lb_link = QLabel()
        lb_link.setText('Function applied to the "proj. files" list:')
        lb_link.setFont(self.font1)

        self.chkbox_multi_selec = QCheckBox('Enable multi-selection')
        self.chkbox_multi_selec.setFont(self.font2)
        self.chkbox_multi_selec.setFixedWidth(300)
        self.chkbox_multi_selec.setChecked(False)
        self.chkbox_multi_selec.stateChanged.connect(self.enable_multi_selection)

        self.pb_filter_all = QPushButton('Select all')
        self.pb_filter_all.setFixedWidth(110)
        self.pb_filter_all.setFont(self.font2)
        self.pb_filter_all.clicked.connect(self.filter_selection_all)

        self.pb_filter_un_select = QPushButton('Un-select all')
        self.pb_filter_un_select.setFixedWidth(110)
        self.pb_filter_un_select.setFont(self.font2)
        self.pb_filter_un_select.clicked.connect(self.filter_un_selection_all)

        self.pb_filter_rotcen = QPushButton('Found rot_cen')
        self.pb_filter_rotcen.setFixedWidth(110)
        self.pb_filter_rotcen.setFont(self.font2)
        self.pb_filter_rotcen.clicked.connect(self.filter_found_rotcen)

        self.pb_filter_recon = QPushButton('Finished recon')
        self.pb_filter_recon.setFixedWidth(110)
        self.pb_filter_recon.setFont(self.font2)
        self.pb_filter_recon.clicked.connect(self.filter_finished_recon)

        self.pb_filter_reverse = QPushButton('Reverse select')
        self.pb_filter_reverse.setFixedWidth(110)
        self.pb_filter_reverse.setFont(self.font2)
        self.pb_filter_reverse.clicked.connect(self.filter_reverse_selection)

        lb_rec_sli = QLabel()
        lb_rec_sli.setText('rec_slices:')
        lb_rec_sli.setFixedWidth(85)
        lb_rec_sli.setFont(self.font2)

        self.tx_rec_sli = QLineEdit()
        self.tx_rec_sli.setFixedWidth(80)
        self.tx_rec_sli.setText('[]')
        self.tx_rec_sli.setFont(self.font2)

        lb_rec_bin = QLabel()
        lb_rec_bin.setText('rec_bin:')
        lb_rec_bin.setFixedWidth(85)
        lb_rec_bin.setFont(self.font2)
        lb_rec_bin.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)

        self.tx_rec_bin = QLineEdit()
        self.tx_rec_bin.setFixedWidth(80)
        self.tx_rec_bin.setText('1')
        self.tx_rec_bin.setFont(self.font2)

        lb_rec_roi_c = QLabel()
        lb_rec_roi_c.setText('roi center:')
        lb_rec_roi_c.setFixedWidth(85)
        lb_rec_roi_c.setFont(self.font2)

        self.tx_rec_roi_c = QLineEdit()
        self.tx_rec_roi_c.setFixedWidth(80)
        self.tx_rec_roi_c.setText('[]')
        self.tx_rec_roi_c.setFont(self.font2)

        lb_rec_roi_s = QLabel()
        lb_rec_roi_s.setText('roi size:')
        lb_rec_roi_s.setFixedWidth(85)
        lb_rec_roi_s.setFont(self.font2)
        lb_rec_roi_s.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)

        self.tx_rec_roi_s = QLineEdit()
        self.tx_rec_roi_s.setFixedWidth(80)
        self.tx_rec_roi_s.setText('[]')
        self.tx_rec_roi_s.setFont(self.font2)
        
        lb_rec_cir_mask_ratio = QLabel()
        lb_rec_cir_mask_ratio.setText('circle mask ratio:')
        lb_rec_cir_mask_ratio.setFixedWidth(125)
        lb_rec_cir_mask_ratio.setFont(self.font2)
        lb_rec_cir_mask_ratio.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)

        self.tx_rec_cir_mask_ratio = QLineEdit()
        self.tx_rec_cir_mask_ratio.setFixedWidth(80)
        self.tx_rec_cir_mask_ratio.setValidator(QDoubleValidator())
        self.tx_rec_cir_mask_ratio.setText('0.95')
        self.tx_rec_cir_mask_ratio.setFont(self.font2)
        
        
        self.chkbox_auto_bl = QCheckBox('Auto block_list:')
        self.chkbox_auto_bl.setFont(self.font2)
        self.chkbox_auto_bl.setFixedWidth(120)
        self.chkbox_auto_bl.setChecked(False)

        lb_auto_bl_ratio = QLabel()
        lb_auto_bl_ratio.setText('ratio = ')
        lb_auto_bl_ratio.setFixedWidth(50)
        lb_auto_bl_ratio.setFont(self.font2)
        lb_auto_bl_ratio.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)

        self.tx_auto_bl_ratio = QLineEdit()
        self.tx_auto_bl_ratio.setFixedWidth(40)
        self.tx_auto_bl_ratio.setText('0.4')
        self.tx_auto_bl_ratio.setFont(self.font2)

        self.pb_rec_s = QPushButton('Recon single')
        self.pb_rec_s.setFixedWidth(170)
        self.pb_rec_s.setFont(self.font2)
        self.pb_rec_s.clicked.connect(lambda:self.recon_single_file(True))

        self.pb_rec_b = QPushButton('Recon batch')
        self.pb_rec_b.setFixedWidth(170)
        self.pb_rec_b.setFont(self.font2)
        self.pb_rec_b.clicked.connect(self.recon_batch_file)

        self.chkbox_napari = QCheckBox(' View in napari')
        self.chkbox_napari.setFont(self.font2)
        self.chkbox_napari.setFixedWidth(200)
        self.chkbox_napari.setChecked(False)

        self.terminal = QPlainTextEdit()
        self.terminal.setFixedWidth(600)
        self.terminal.setFixedHeight(200)
        self.terminal.setFont(self.font2)

        self.pb_rec_view = QPushButton('View 3D recon')
        self.pb_rec_view.setFixedWidth(170)
        self.pb_rec_view.setFont(self.font2)
        self.pb_rec_view.clicked.connect(self.view_3D_recon)

        lb_v_sep = QLabel()
        lb_v_sep.setFixedHeight(10)
        if torch_installed:
            vbox_ml = self.layout_ml()
            vbox_ml_rec = self.layout_ml_recon()


        vbox_link = QVBoxLayout()
        vbox_link.addWidget(lb_link)
        vbox_link.addWidget(self.chkbox_multi_selec)
        vbox_link.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)

        hbox_filt = QHBoxLayout()
        hbox_filt.addWidget(self.pb_filter_all)
        hbox_filt.addWidget(self.pb_filter_un_select)
        hbox_filt.addWidget(self.pb_filter_reverse)
        hbox_filt.addWidget(self.pb_filter_rotcen)
        hbox_filt.addWidget(self.pb_filter_recon)
        hbox_filt.addStretch()
        hbox_filt.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)

        hbox_rec1 = QHBoxLayout()
        hbox_rec1.addWidget(lb_rec_sli)
        hbox_rec1.addWidget(self.tx_rec_sli)
        hbox_rec1.addWidget(lb_rec_bin)
        hbox_rec1.addWidget(self.tx_rec_bin)
        hbox_rec1.addWidget(lb_rec_cir_mask_ratio)
        hbox_rec1.addWidget(self.tx_rec_cir_mask_ratio)
        hbox_rec1.addStretch()
        hbox_rec1.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)

        hbox_rec2 = QHBoxLayout()
        hbox_rec2.addWidget(lb_rec_roi_c)
        hbox_rec2.addWidget(self.tx_rec_roi_c)
        hbox_rec2.addWidget(lb_rec_roi_s)
        hbox_rec2.addWidget(self.tx_rec_roi_s)
        hbox_rec2.addWidget(lb_empty2)
        hbox_rec2.addWidget(self.chkbox_auto_bl)
        hbox_rec2.addWidget(lb_auto_bl_ratio)
        hbox_rec2.addWidget(self.tx_auto_bl_ratio)
        hbox_rec2.addStretch()
        hbox_rec2.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)

        hbox_rec3 = QHBoxLayout()
        hbox_rec3.addWidget(self.pb_rec_s)
        hbox_rec3.addWidget(self.pb_rec_b)
        hbox_rec3.addWidget(self.chkbox_napari)
        hbox_rec3.addStretch()
        hbox_rec3.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)

        vbox_rec_view = QVBoxLayout()
        vbox_rec_view.addLayout(hbox_rec3)
        vbox_rec_view.addWidget(self.pb_rec_view)
        vbox_rec_view.addStretch()
        vbox_rec_view.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)

        vbox_rec = QVBoxLayout()
        #vbox_rec.addWidget(lb_recon)
        vbox_rec.addLayout(vbox_link)
        vbox_rec.addLayout(hbox_filt)
        vbox_rec.addWidget(lb_empty1)
        vbox_rec.addLayout(hbox_rec1)
        vbox_rec.addLayout(hbox_rec2)
        if torch_installed:
            vbox_rec.addLayout(vbox_ml)
        vbox_rec.addLayout(vbox_rec_view)
        if torch_installed:
            vbox_rec.addWidget(lb_v_sep)
            vbox_rec.addLayout(vbox_ml_rec)
        #vbox_rec.addWidget(self.terminal)
        vbox_rec.addWidget(lb_empty)
        vbox_rec.addStretch()
        vbox_rec.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        return vbox_rec



    def layout_ml(self):
        lb_empty = QLabel()
        lb_empty1 = QLabel()
        self.chkbox_ml = QCheckBox(' Apply ML on projection image')
        self.chkbox_ml.setFont(self.font2)
        self.chkbox_ml.setFixedWidth(220)
        self.chkbox_ml.setChecked(False)
        self.chkbox_ml.stateChanged.connect(self.ml_check_model)

        lb_ml_model = QLabel()
        lb_ml_model.setText('ML model path:')
        lb_ml_model.setFont(self.font2)
        lb_ml_model.setFixedWidth(105)

        self.pb_ml_model = QPushButton('load model')
        self.pb_ml_model.setFixedWidth(105)
        self.pb_ml_model.setFont(self.font2)
        self.pb_ml_model.clicked.connect(self.ml_load_model)

        self.tx_ml_model = QLineEdit()
        self.tx_ml_model.setFixedWidth(400)
        self.tx_ml_model.setText('')
        self.tx_ml_model.setFont(self.font2)

        self.pb_ml_model_test = QPushButton('Test on selected projection file')
        self.pb_ml_model_test.setFixedWidth(345)
        self.pb_ml_model_test.setFont(self.font2)
        self.pb_ml_model_test.clicked.connect(self.ml_test_proj)

        lb_ml_num_iter = QLabel()
        lb_ml_num_iter.setText('Num. Iter:')
        lb_ml_num_iter.setFont(self.font2)
        lb_ml_num_iter.setFixedWidth(80)

        self.tx_ml_num_iter = QLineEdit()
        self.tx_ml_num_iter.setFixedWidth(35)
        self.tx_ml_num_iter.setText('1')
        self.tx_ml_num_iter.setFont(self.font2)

        lb_ml_fz = QLabel()
        lb_ml_fz.setText('filter sz:')
        lb_ml_fz.setFont(self.font2)
        lb_ml_fz.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        lb_ml_fz.setFixedWidth(80)

        self.tx_ml_fz = QLineEdit()
        self.tx_ml_fz.setFixedWidth(35)
        self.tx_ml_fz.setText('1')
        self.tx_ml_fz.setFont(self.font2)

        lb_ml_device = QLabel()
        lb_ml_device.setText('Device:')
        lb_ml_device.setFont(self.font2)
        lb_ml_device.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        lb_ml_device.setFixedWidth(80)

        self.cb_ml_device = QComboBox()
        self.cb_ml_device.setFont(self.font2)
        self.cb_ml_device.setFixedWidth(85)
        if self.gpu_count == 0:
            self.cb_ml_device.addItem('cpu')
        elif self.gpu_count == 1:
            self.cb_ml_device.addItem('cuda')
        else:
            for i in range(self.gpu_count):
                self.cb_ml_device.addItem(f'cuda:{i:d}')

        hbox = QHBoxLayout()
        hbox.addWidget(lb_ml_model)
        hbox.addWidget(self.tx_ml_model)
        hbox.addWidget(self.pb_ml_model)
        hbox.setAlignment(QtCore.Qt.AlignLeft)

        hbox_ml = QHBoxLayout()
        hbox_ml.addWidget(lb_ml_num_iter)
        hbox_ml.addWidget(self.tx_ml_num_iter)
        hbox_ml.addWidget(lb_ml_fz)
        hbox_ml.addWidget(self.tx_ml_fz)
        hbox_ml.addWidget(lb_ml_device)
        hbox_ml.addWidget(self.cb_ml_device)
        hbox_ml.addWidget(lb_empty1)
        hbox_ml.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)

        hbox_ml2 = QHBoxLayout()
        hbox_ml2.addWidget(self.pb_ml_model_test)
        hbox_ml2.addWidget(self.chkbox_ml)
        hbox_ml2.addWidget(lb_empty1)
        hbox_ml2.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)


        vbox = QVBoxLayout()
        vbox.addWidget(lb_empty)
        #vbox.addWidget(self.chkbox_ml)
        vbox.addLayout(hbox)
        vbox.addLayout(hbox_ml)
        vbox.addLayout(hbox_ml2)
        vbox.addWidget(lb_empty1)
        #vbox.addWidget(self.pb_ml_model_test)
        vbox.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        return vbox


    def layout_ml_recon(self):

        lb_sep1 = QLabel()
        lb_sep1.setFixedWidth(60)

        lb_sep2 = QLabel()
        lb_sep2.setFixedWidth(5)

        lb_empty1 = QLabel()
        lb_v_sep = QLabel()
        lb_v_sep.setFixedHeight(5)

        lb_title = QLabel()
        lb_title.setText('ML denoise on recon3D')
        lb_title.setFont(self.font1)
        lb_title.setFixedWidth(200)

        lb_model_sel = QLabel()
        lb_model_sel.setText('ML model:')
        lb_model_sel.setFont(self.font2)
        lb_model_sel.setFixedWidth(105)

        # can add more models in the future
        self.cb_ml_model_sel = QComboBox()
        self.cb_ml_model_sel.setFont(self.font2)
        self.cb_ml_model_sel.setFixedWidth(105)
        self.cb_ml_model_sel.addItem('RRDB 4')

        lb_ml_model_path = QLabel()
        lb_ml_model_path.setText('ML model path:')
        lb_ml_model_path.setFont(self.font2)
        lb_ml_model_path.setFixedWidth(105)

        self.pb_ml_model_path = QPushButton('load model')
        self.pb_ml_model_path.setFixedWidth(105)
        self.pb_ml_model_path.setFont(self.font2)
        self.pb_ml_model_path.clicked.connect(self.ml_load_model_rec)

        self.tx_ml_model_path = QLineEdit()
        self.tx_ml_model_path.setFixedWidth(400)
        self.tx_ml_model_path.setText('')
        self.tx_ml_model_path.setFont(self.font2)

        ## OSTU auto_threshold
        lb_ml_rm_bkg = QLabel()
        lb_ml_rm_bkg.setFont(self.font2)
        lb_ml_rm_bkg.setText('Auto Bkg. Rmv:')
        lb_ml_rm_bkg.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        lb_ml_rm_bkg.setFixedWidth(105)

        lb_ml_filt_sz = QLabel()
        lb_ml_filt_sz.setFont(self.font2)
        lb_ml_filt_sz.setText('Filt_sz:')
        lb_ml_filt_sz.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        lb_ml_filt_sz.setFixedWidth(50)

        self.tx_ml_filt_sz = QLineEdit()
        self.tx_ml_filt_sz.setFixedWidth(40)
        self.tx_ml_filt_sz.setText('3')
        self.tx_ml_filt_sz.setFont(self.font2)
        self.tx_ml_filt_sz.setValidator(QIntValidator())

        lb_ml_filt_iter = QLabel()
        lb_ml_filt_iter.setFont(self.font2)
        lb_ml_filt_iter.setText('Iter:')
        lb_ml_filt_iter.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        lb_ml_filt_iter.setFixedWidth(50)

        self.tx_ml_filt_iter = QLineEdit()
        self.tx_ml_filt_iter.setFixedWidth(40)
        self.tx_ml_filt_iter.setText('2')
        self.tx_ml_filt_iter.setFont(self.font2)
        self.tx_ml_filt_iter.setValidator(QIntValidator())

        lb_ml_filt_bins = QLabel()
        lb_ml_filt_bins.setFont(self.font2)
        lb_ml_filt_bins.setText('Bins:')
        lb_ml_filt_bins.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        lb_ml_filt_bins.setFixedWidth(50)

        self.tx_ml_filt_bins = QLineEdit()
        self.tx_ml_filt_bins.setFixedWidth(40)
        self.tx_ml_filt_bins.setText('256')
        self.tx_ml_filt_bins.setFont(self.font2)
        self.tx_ml_filt_bins.setValidator(QIntValidator())

        """
        self.pb_ml_filt_test = QPushButton('Test')
        self.pb_ml_filt_test.setFixedWidth(105)
        self.pb_ml_filt_test.setFont(self.font2)
        self.pb_ml_filt_test.clicked.connect(self.ml_auto_thresh_bkg)
        """
        ## median filter
        lb_sep3 = QLabel()
        lb_sep3.setFixedWidth(264)

        lb_ml_medfilt = QLabel()
        lb_ml_medfilt.setFont(self.font2)
        lb_ml_medfilt.setText('Median filter:')
        lb_ml_medfilt.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        lb_ml_medfilt.setFixedWidth(105)

        lb_ml_medfilt_sz = QLabel()
        lb_ml_medfilt_sz.setFont(self.font2)
        lb_ml_medfilt_sz.setText('Filt_sz:')
        lb_ml_medfilt_sz.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        lb_ml_medfilt_sz.setFixedWidth(50)

        self.tx_ml_medfilt_sz = QLineEdit()
        self.tx_ml_medfilt_sz.setFixedWidth(40)
        self.tx_ml_medfilt_sz.setText('3')
        self.tx_ml_medfilt_sz.setFont(self.font2)
        self.tx_ml_medfilt_sz.setValidator(QIntValidator())

        """
        self.pb_ml_medfilt_test = QPushButton('Test')
        self.pb_ml_medfilt_test.setFixedWidth(105)
        self.pb_ml_medfilt_test.setFont(self.font2)
        self.pb_ml_medfilt_test.clicked.connect(self.ml_mdedfilt)
        """

        ### droplist
        lb_droplist_empty = QLabel()
        lb_droplist_tit = QLabel()
        lb_droplist_v_sep = QLabel()
        lb_droplist_v_sep.setFixedHeight(10)

        lb_droplist_tit.setText(' - Compile denoise methods in sequence')
        lb_droplist_tit.setFont(self.font1)
        #lb_droplist_tit.setStyleSheet('color: rgb(50, 50, 250);')

        self.cb_ml_choice = QComboBox()
        self.cb_ml_choice.setFont(self.font2)
        self.cb_ml_choice.setFixedWidth(120)
        self.cb_ml_choice.addItem('Median filter')
        self.cb_ml_choice.addItem('Auto Bkg. Rmv')
        self.cb_ml_choice.addItem('ML denoise')

        self.pb_ml_choice = QPushButton('Insert method')
        self.pb_ml_choice.setFixedWidth(120)
        self.pb_ml_choice.setFont(self.font2)
        self.pb_ml_choice.setStyleSheet('color: rgb(50, 50, 250);')
        self.pb_ml_choice.clicked.connect(self.ml_select_denoise_methode)

        self.pb_ml_rm_choice = QPushButton('Reset')
        self.pb_ml_rm_choice.setFixedWidth(120)
        self.pb_ml_rm_choice.setFont(self.font2)
        self.pb_ml_rm_choice.clicked.connect(self.ml_reset_denoise_methode)

        self.lst_ml_choice = QListWidget()
        self.lst_ml_choice.setFont(self.font2)
        self.lst_ml_choice.setSelectionMode(QAbstractItemView.MultiSelection)
        self.lst_ml_choice.setFixedWidth(120)
        self.lst_ml_choice.setFixedHeight(90)

        # execusion 1
        self.pb_rec_ml_enoise = QPushButton('ML denoise on recon')
        self.pb_rec_ml_enoise.setFixedWidth(170)
        self.pb_rec_ml_enoise.setFont(self.font2)
        self.pb_rec_ml_enoise.clicked.connect(self.ml_3D_denosie_recon)

        lb_ml_device_rec = QLabel()
        lb_ml_device_rec.setText('Device:')
        lb_ml_device_rec.setFont(self.font2)
        lb_ml_device_rec.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        lb_ml_device_rec.setFixedWidth(105)

        self.cb_ml_device_rec = QComboBox()
        self.cb_ml_device_rec.setFont(self.font2)
        self.cb_ml_device_rec.setFixedWidth(85)
        if self.gpu_count == 0:
            self.cb_ml_device_rec.addItem('cpu')
        elif self.gpu_count == 1:
            self.cb_ml_device_rec.addItem('cuda')
        else:
            for i in range(self.gpu_count):
                self.cb_ml_device_rec.addItem(f'cuda:{i:d}')

        # test slice
        self.pb_rec_ml_test = QPushButton('Test slice')
        self.pb_rec_ml_test.setFixedWidth(170)
        self.pb_rec_ml_test.setFont(self.font2)
        self.pb_rec_ml_test.clicked.connect(self.ml_3D_denoise_slice) # need to modify

        lb_ml_test_sli = QLabel()
        lb_ml_test_sli.setText('Slice:')
        lb_ml_test_sli.setFont(self.font2)
        lb_ml_test_sli.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        lb_ml_test_sli.setFixedWidth(105)

        self.tx_ml_test_sli = QLineEdit()
        self.tx_ml_test_sli.setFixedWidth(85)
        self.tx_ml_test_sli.setFont(self.font2)
        self.tx_ml_test_sli.setValidator(QIntValidator())

        # msg box
        self.lb_ml_model_rec_msg = QLabel()
        self.lb_ml_model_rec_msg.setStyleSheet('color: rgb(250, 50, 50);')

        hbox_ml_model_path = QHBoxLayout()
        hbox_ml_model_path.addWidget(lb_ml_model_path)
        hbox_ml_model_path.addWidget(self.tx_ml_model_path)
        hbox_ml_model_path.addWidget(self.pb_ml_model_path)
        hbox_ml_model_path.setAlignment(QtCore.Qt.AlignLeft)

        hbox_ml_model_sel = QHBoxLayout()
        hbox_ml_model_sel.addWidget(lb_model_sel)
        hbox_ml_model_sel.addWidget(self.cb_ml_model_sel)
        hbox_ml_model_sel.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)

        hbox_ml_test = QHBoxLayout()
        hbox_ml_test.addWidget(self.pb_rec_ml_test)
        hbox_ml_test.addWidget(lb_ml_test_sli)
        hbox_ml_test.addWidget(self.tx_ml_test_sli)
        hbox_ml_test.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)

        hbox_ml_denoise = QHBoxLayout()
        hbox_ml_denoise.addWidget(self.pb_rec_ml_enoise)
        hbox_ml_denoise.addWidget(lb_ml_device_rec)
        hbox_ml_denoise.addWidget(self.cb_ml_device_rec)
        hbox_ml_denoise.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)

        vbox_ml_exe = QVBoxLayout()
        vbox_ml_exe.addLayout(hbox_ml_test)
        vbox_ml_exe.addLayout(hbox_ml_denoise)
        vbox_ml_exe.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)

        hbox_ml_medfilt = QHBoxLayout()
        hbox_ml_medfilt.addWidget(lb_ml_medfilt)
        hbox_ml_medfilt.addWidget(lb_ml_medfilt_sz)
        hbox_ml_medfilt.addWidget(self.tx_ml_medfilt_sz)
        hbox_ml_medfilt.addWidget(lb_sep3)
        #hbox_ml_medfilt.addWidget(self.pb_ml_medfilt_test)
        hbox_ml_medfilt.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)

        hbox_ml_rm_bkg = QHBoxLayout()
        hbox_ml_rm_bkg.addWidget(lb_ml_rm_bkg)
        hbox_ml_rm_bkg.addWidget(lb_ml_filt_sz)
        hbox_ml_rm_bkg.addWidget(self.tx_ml_filt_sz)
        hbox_ml_rm_bkg.addWidget(lb_ml_filt_iter)
        hbox_ml_rm_bkg.addWidget(self.tx_ml_filt_iter)
        hbox_ml_rm_bkg.addWidget(lb_ml_filt_bins)
        hbox_ml_rm_bkg.addWidget(self.tx_ml_filt_bins)
        hbox_ml_rm_bkg.addWidget(lb_sep1)
        #hbox_ml_rm_bkg.addWidget(self.pb_ml_filt_test )
        hbox_ml_rm_bkg.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)

        vbox_ml_select = QVBoxLayout()
        vbox_ml_select.addWidget(self.cb_ml_choice)
        vbox_ml_select.addWidget(self.pb_ml_choice)
        vbox_ml_select.addWidget(self.pb_ml_rm_choice)
        vbox_ml_select.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)

        hbox_ml_test = QHBoxLayout()
        hbox_ml_test.addLayout(vbox_ml_select)
        hbox_ml_test.addWidget(self.lst_ml_choice)
        hbox_ml_test.addLayout(vbox_ml_exe)
        hbox_ml_test.addWidget(lb_droplist_empty)
        hbox_ml_test.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)

        vbox = QVBoxLayout()
        vbox.addWidget(lb_title)
        vbox.addLayout(hbox_ml_medfilt)
        vbox.addLayout(hbox_ml_rm_bkg)
        vbox.addWidget(lb_v_sep)
        vbox.addLayout(hbox_ml_model_sel)
        vbox.addLayout(hbox_ml_model_path)
        #vbox.addLayout(hbox_ml)
        vbox.addWidget(lb_droplist_v_sep)
        vbox.addWidget(lb_droplist_tit)
        vbox.addLayout(hbox_ml_test)
        vbox.addWidget(self.lb_ml_model_rec_msg)
        vbox.addWidget(lb_empty1)
        vbox.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        return vbox

    def layout_recon_and_canvas(self):
        vbox_canvas = self.layout_canvas()
        vbox_rec = self.layout_recon()

        gp_box_canvas = QGroupBox()
        gp_box_rec = QGroupBox()

        gp_box_canvas.setLayout(vbox_canvas)
        gp_box_rec.setLayout(vbox_rec)

        tabs = QTabWidget()
        tab_canvas = QWidget()
        tab_rec = QWidget()

        tab_canvas.setLayout(vbox_canvas)
        tab_rec.setLayout(vbox_rec)

        tabs.addTab(tab_canvas, 'Image plot')
        tabs.addTab(tab_rec, 'Other utils')
        return tabs

    def layout_tomo(self):
        lb_empty1 = QLabel()
        lb_empty2 = QLabel()
        hbox_rc_list = self.layout_rc_list()
        vbox_rc_param = self.layout_rc_param()
        vbox_rc_exec = self.layout_rc_exec()
        vbox_rc_range = self.layout_rc_range()
        #vbox_canvas = self.layout_canvas()
        #vbox_rec = self.layout_recon()
        tab_rec_canvas = self.layout_recon_and_canvas()

        vbox_comb = QVBoxLayout()
        vbox_comb.addLayout(hbox_rc_list)
        vbox_comb.addLayout(vbox_rc_range)
        vbox_comb.addLayout(vbox_rc_param)
        vbox_comb.addLayout(vbox_rc_exec)
        #vbox_comb.addLayout(vbox_rec)

        hbox_comb = QHBoxLayout()
        hbox_comb.addLayout(vbox_comb)
        hbox_comb.addWidget(lb_empty1)
        #hbox_comb.addLayout(vbox_canvas)
        hbox_comb.addWidget(tab_rec_canvas)
        hbox_comb.addWidget(lb_empty2)
        hbox_comb.addStretch()
        hbox_comb.setAlignment(QtCore.Qt.AlignLeft)

        gpbox = QGroupBox('Rotation center and tomography  ')
        gpbox.setFont(self.font1)
        gpbox.setLayout(hbox_comb)
        return gpbox

    '''
    def retrieve_file_type(self, file_path, file_prefix='fly', file_type='.h5'):
        import os
        path = os.path.abspath(file_path)
        files = sorted(os.listdir(file_path))
        files_filted = []
        n_type = len(file_type)
        n_start = len(file_prefix)
        for f in files:
            if f[-n_type:] == file_type and f[:n_start] == file_prefix:
                f = f'{path}/{f}'
                files_filted.append(f)
        return files_filted
    '''

    def ml_get_param(self):
        ml_param = {}
        if torch_installed:
            apply_ml_flag = self.ml_check_model()
            if apply_ml_flag:
                n_iter = int(self.tx_ml_num_iter.text())
                filt_sz = int(self.tx_ml_fz.text())
                device = self.cb_ml_device.currentText()
                model_path = self.tx_ml_model.text()
                ml_param['n_iter'] = n_iter
                ml_param['filt_sz'] = filt_sz
                ml_param['device'] = device
                ml_param['model_path'] = model_path
        return ml_param

    def ml_load_model(self):
        options = QFileDialog.Option()
        options |= QFileDialog.DontUseNativeDialog
        try:
            file_type = 'pth files (*.pth)'
            fn, _ = QFileDialog.getOpenFileName(pytomo, "QFileDialog.getOpenFileName()", "", file_type, options=options)
            if fn:
                self.ml_model_path = fn
                self.tx_ml_model.setText(fn)
            else:
                self_ml_model_path = ''
            QApplication.processEvents()
        except Exception as err:
            self.msg = str(err)
            self.update_msg()

    def ml_load_model_rec(self):
        options = QFileDialog.Option()
        options |= QFileDialog.DontUseNativeDialog
        try:
            file_type = 'pth files (*.pth)'
            fn, _ = QFileDialog.getOpenFileName(pytomo, "QFileDialog.getOpenFileName()", "", file_type, options=options)
            if fn:
                self.ml_model_recon_path = fn
                self.tx_ml_model_path.setText(fn)
            else:
                self_ml_model_path = ''
            QApplication.processEvents()
        except Exception as err:
            self.msg = str(err)
            self.update_msg()


    def ml_test_proj(self):
        try:
            self.pb_ml_model_test.setText('Wait ...')
            self.pb_ml_model_test.setEnabled(False)
            QApplication.processEvents()
            if self.chkbox_multi_selec.isChecked():
                self.set_single_selection()
                return 0
            item = self.lst_prj_file.selectedItems()
            if len(item) == 0:
                self.msg = 'No file selected'
                self.update_msg()
                return 0
            fn_short = item[0].text()
            fn_short = fn_short.split(':')[0]
            fn = self.fname_rc[fn_short]['full_path']

            self.load_proj_file(fn)  # get self.img_prj_norm
            proj_norm = self.img_prj_norm

            n_iter = int(self.tx_ml_num_iter.text())
            filt_sz = int(self.tx_ml_fz.text())
            device = self.cb_ml_device.currentText()
            model_path = self.tx_ml_model.text()

            self.proj_ml = apply_ML_prj(proj_norm, n_iter, filt_sz, model_path, device)
            if exist_napari and self.chkbox_napari.isChecked():
                napari.view_image(proj_norm, title='Raw')
                napari.view_image(self.proj_ml, title='ML')

            sup_title = fn_short
            self.canvas1.sup_title = sup_title
            if self.cb1.findText('proj vs. proj_ml (raw)') < 0:
                self.cb1.addItem('proj vs. proj_ml (raw)')
                self.cb1.addItem('proj vs. proj_ml (ml)')
            self.update_canvas_img()
            self.msg = f'{fn_short}: ML applied to projection image'
        except Exception as err:
            self.msg = str(err)
        finally:
            self.update_msg()
            self.pb_ml_model_test.setText('Test on selected projection file')
            self.pb_ml_model_test.setEnabled(True)
            QApplication.processEvents()



    def ml_check_model(self):
        if self.chkbox_ml.isChecked():
            return True
        else:
            return False


    def get_proj_from_file(self, fn, attr_proj, attr_flat, attr_dark):
        with h5py.File(fn, 'r') as hf:
            img_proj = np.array(hf[attr_proj])
            img_flat = np.array(hf[attr_flat])
            img_dark = np.array(hf[attr_dark])
        img_dark = np.median(img_dark, axis=0, keepdims=True)
        img_flat = np.median(img_flat, axis=0, keepdims=True)
        proj_norm = (img_proj-img_dark) / (img_flat - img_dark)
        return proj_norm

    def open_prj_file(self):
        file_prefix = self.tx_prefix.text()
        file_dict = {}
        options = QFileDialog.Option()
        options |= QFileDialog.DontUseNativeDialog
        file_type = 'h5 file (*.h5)'
        fn, _ = QFileDialog.getOpenFileName(pytomo, "QFileDialog.getOpenFileName()", "", file_type, options=options)
        if fn:
            fn_tmp = fn.split('/')
            file_root_path = '/'.join(t for t in fn_tmp[:-1])

            tmp_prefix = fn_tmp[-1][0]
            if not (tmp_prefix in file_prefix):
                file_prefix = fn_tmp[-1][0]
                self.tx_prefix.setText(file_prefix)
                QApplication.processEvents()
            file_type = '.' + fn_tmp[-1].split('.')[-1]
            file_loaded = retrieve_file_type(file_root_path, file_prefix, file_type)
            num = len(file_loaded)
            self.tx_prj_path.setText(file_root_path)
            for i in range(num):
                full_path = file_loaded[i]
                fn_short = full_path.split('/')[-1]
                file_dict[fn_short] = {'rc': 0, 'recon_flag':None, 'full_path':full_path}
            self.fname_rc.update(file_dict)
            self.file_loaded.append(file_loaded)
            self.update_list()
            self.fn_rc_tmp = ''

    def refresh_file_folder(self):
        file_dict = {}
        file_prefix = self.tx_prefix.text()
        file_type = '.h5'
        file_root_path = self.tx_prj_path.text().replace(' ', '')
        file_loaded = retrieve_file_type(file_root_path, file_prefix, file_type)
        num = len(file_loaded)
        self.tx_prj_path.setText(file_root_path)
        exist_files = self.fname_rc.keys()
        for i in range(num):
            full_path = file_loaded[i]
            fn_short = full_path.split('/')[-1]
            if not fn_short in exist_files: 
                file_dict[fn_short] = {'rc': 0, 'recon_flag': None, 'full_path': full_path}
        self.fname_rc.update(file_dict)
        self.fname_rc = dict(sorted(self.fname_rc.items()))
        self.file_loaded.append(file_loaded)
        self.update_list()

    def clear_file_list(self):
        self.fname_rc = {}
        self.file_loaded = []
        self.update_list()

    def del_list_item(self):
        try:
            item = self.lst_prj_file.selectedItems()
            n = len(item)
            for i in range(n):
                fn_short = item[i].text()
                fn_short = fn_short.split(':')[0]
                if fn_short in self.fname_rc.keys():
                    self.remove_fname_rc(fn_short)
            self.update_list()
            self.msg = f'deleted {fn_short}'
        except Exception as err:
            self.msg = str(err)
        finally:
            self.update_msg()

    def filter_selection_all(self):
        self.lst_prj_file.selectAll()


    def filter_un_selection_all(self):
        self.lst_prj_file.clearSelection()


    def filter_found_rotcen(self):
        n_list = self.lst_prj_file.count()
        for i in range(n_list):
            item = self.lst_prj_file.item(i)
            fn_short = item.text().split(':')[0]
            if self.fname_rc[fn_short]['rc'] > 0:
                #self.lst_prj_file.setItemSelected(item, True)
                item.setSelected(True)
            else:
                #self.lst_prj_file.setItemSelected(item, False)
                item.setSelected(False)


    def filter_finished_recon(self):
        n_list = self.lst_prj_file.count()
        for i in range(n_list):
            item = self.lst_prj_file.item(i)
            fn_short = item.text().split(':')[0]
            recon_flag = self.fname_rc[fn_short]['recon_flag']
            if not recon_flag is None:
                #self.lst_prj_file.setItemSelected(item, True)
                item.setSelected(True)
            else:
                #self.lst_prj_file.setItemSelected(item, False)
                item.setSelected(False)

    def filter_reverse_selection(self):
        n_list = self.lst_prj_file.count()

        items = self.lst_prj_file.selectedItems()
        n_sel = len(items)
        item_sel_idx = []
        for i in range(n_sel):
            item = items[i]
            item_idx = self.lst_prj_file.indexFromItem(item).row()
            item_sel_idx.append(item_idx)

        for i in range(n_list):
            item = self.lst_prj_file.item(i)
            if i in item_sel_idx:
                #self.lst_prj_file.setItemSelected(item, False)
                item.setSelected(False)
            else:
                #self.lst_prj_file.setItemSelected(item, True)
                item.setSelected(True)

    def update_list(self):
        n_file = len(self.fname_rc)
        keys = list(self.fname_rc.keys())
        n_list = self.lst_prj_file.count()
        if n_file != n_list:
            self.lst_prj_file.clear()
            for i in range(n_file):
                self.lst_prj_file.addItem(keys[i])
        QApplication.processEvents()

        n_list = self.lst_prj_file.count() # updated list
        for i in range(n_list):
            item = self.lst_prj_file.item(i)
            tx_current = item.text() # e.g. 'fly_scan_123:  640' or 'fly_scan_123:  640    (recon)
            fn_short = tx_current.split(':')[0]
            current_rc, recon_flag = self.check_fname_rc_states(fn_short)
            if current_rc != 0:
                if not recon_flag is None and recon_flag[0] == 'Y':
                    txt = f'{fn_short}:    {current_rc:3.2f}    (recon)'
                else:
                    txt = f'{fn_short}:    {current_rc:3.2f}'
            else:
                txt = fn_short
            item.setText(txt)


    def find_rotation_center_core(self, fn):
        self.flag_rc = False
        try:
            attr_proj = self.tx_h5_prj.text()
            attr_flat = self.tx_h5_flat.text()
            attr_dark = self.tx_h5_dark.text()
            attr_angle = self.tx_h5_ang.text()
            attr_xeng = self.tx_h5_xeng.text()

            start = self.tx_rc_start.text()
            start = None if start == 'None' else float(start)

            stop = self.tx_rc_stop.text()
            stop = None if stop == 'None' else float(stop)

            steps = int(self.tx_rc_steps.text())
            sli = int(self.tx_sli_id.text())
            block_list = self._get_block_list()
            auto_block_list = self._get_block_list_auto()
            denoise_flag = int(self.tx_rc_denoise.text())
            dark_scale = int(self.tx_rc_dark_scale.text())
            
            mask_r = float(self.tx_rec_cir_mask_ratio.text())
            
            
            p = int(self.tx_rc_ring_remove.text())
            if self.cb_rm_strip.currentText() == 'wavelet':
                fw_level = p
                snr = 0
            else:
                fw_level = 0
                snr = p

            options = {}
            algorithm = self.cb_rc_algorithm.currentText()
            print(algorithm)
            if algorithm != 'gridrec':               
 
                if algorithm == 'astra_fbp':
                    method = 'FBP_CUDA'
                elif algorithm == 'astra_sirt':
                    method = 'SIRT_CUDA'
                num_iter = int(self.tx_rc_algorithm_iter.text())
                extra_options = {'MinConstraint': 0}                     
                options = {
                            'proj_type': 'cuda',
                            'method': method,
                            'num_iter': num_iter,
                            'extra_options': extra_options
                          }
                    


            self.img_rc, self.rc, start, stop, steps, sli = rotcen_test(fn, attr_proj,
                                   attr_flat, attr_dark, attr_angle, start, stop, steps,
                                   sli, block_list, denoise_flag, dark_scale=dark_scale,
                                   circ_mask_ratio=mask_r,
                                   algorithm=algorithm, options=options,
                                   snr=snr, fw_level=fw_level, ml_param={},
                                   auto_block_list=auto_block_list)
            self.tx_rc_start.setText(str(start))
            self.tx_rc_stop.setText(str(stop))
            self.tx_rc_steps.setText(str(steps))
            self.tx_sli_id.setText(str(sli))
            with h5py.File(fn, 'r') as hf:
                self.img_eng = np.array(hf[attr_xeng])
            self.flag_rc = True
        except Exception as err:
            print(err)
            self.img_rc = np.zeros((1, 100, 100))
            self.flag_rc = False



    def find_rotation_center(self):
        if self.chkbox_multi_selec.isChecked():
            self.set_single_selection()
            return 0

        item = self.lst_prj_file.selectedItems()
        if len(item) == 0:
            self.msg = 'No file selected'
            self.update_msg()
            return 0

        fn_short = item[0].text()
        fn_short = fn_short.split(':')[0]
        fn = self.fname_rc[fn_short]['full_path']
        self.current_file = fn
        self.current_file_short = fn_short
        self.pb_rc_find.setEnabled(False)
        self.pb_rc_find.setText('wait ...')
        self.pb_rc_find.setStyleSheet('color: rgb(200, 200, 200);')
        QApplication.processEvents()
        try:
            self.find_rotation_center_core(fn)
            sup_title = f'{fn_short}:     {self.img_eng:2.4f} keV'
            self.canvas1.sup_title = sup_title
            if self.flag_rc and self.cb1.findText('Rotation center') < 0:
                self.cb1.addItem('Rotation center')
            if self.flag_rc and self.cb1.findText('Rotation center (crop)') < 0:
                self.cb1.addItem('Rotation center (crop)')
            self.cb1.setCurrentText('Rotation center (crop)')
            self.update_canvas_img()
            self.msg = 'Rotation center test finished'
        except Exception as err:
            print(err)
            self.msg = str(err)
        finally:
            self.pb_rc_find.setEnabled(True)
            self.pb_rc_find.setText('Find center')
            self.pb_rc_find.setStyleSheet('color: rgb(50, 50, 250);')
            self.update_msg()
            QApplication.processEvents()

    def _get_block_list(self):
        tx = self.tx_rc_block_list.text()
        tx = tx.replace('[', '')
        tx = tx.replace(']', '')
        try:
            tmp = tx.split(',')
            bl1 = int(tmp[0])
            bl2 = int(tmp[-1])
            block_list = np.arange(bl1, bl2)
        except:
            block_list = []
        return block_list

    def _get_block_list_auto(self):
        r = float(self.tx_auto_bl_ratio.text())
        auto_block_list = {}
        if self.chkbox_auto_bl.isChecked():
            auto_block_list['flag'] = True
            auto_block_list['ratio'] = r
        else:
            auto_block_list['flag'] = False
            auto_block_list['ratio'] = 0
        return auto_block_list

    def find_rotation_center_next(self):
        try:
            self.pb_rc_find_next.setText('wait ...')
            self.pb_rc_find_next.setEnabled(False)
            self.pb_rc_find_next.setStyleSheet('color: rgb(200, 200, 200);')
            n_list = self.lst_prj_file.count()
            item = self.lst_prj_file.selectedItems()[0]
            item_idx = self.lst_prj_file.indexFromItem(item).row()
            item_next_idx = item_idx + 1
            if item_next_idx <= n_list - 1:
                self.lst_prj_file.setCurrentRow(item_next_idx)
                QApplication.processEvents()
                self.find_rotation_center()
            else:
                self.msg = 'reach the end of file list'
        except Exception as err:
            self.msg = str(err)
        finally:
            self.pb_rc_find_next.setText('Next')
            self.pb_rc_find_next.setEnabled(True)
            self.pb_rc_find_next.setStyleSheet('color: rgb(50, 50, 250);')
            self.update_msg()


    def confirm_rotation_center(self):
        try:
            idx = self.sl1.value()
            rc = self.rc[idx]
            fn = self.current_file
            fn_short = fn.split('/')[-1]
            if len(fn_short) > 0:
                self.update_fname_rc(fn_short, rc, None)
            self.update_list()
        except Exception as err:
            self.msg = str(err)
            self.update_msg()


    def update_fname_rc(self, fn_short, rc, recon_flag=None):
        '''
        recon_flag:
            if None: reset to un-reconstructed states.
            if -1: keep the current status of reconstruction: either None or "Y"
        '''

        full_path = self.fname_rc[fn_short].get('full_path')
        if recon_flag == -1:
            recon_flag = self.fname_rc[fn_short].get('recon_flag')  # None if not exist
            self.fname_rc[fn_short] = {'rc': rc, 'recon_flag': recon_flag, 'full_path':full_path}
        elif recon_flag is None:
            self.fname_rc[fn_short] = {'rc': rc, 'recon_flag': None, 'full_path':full_path}
        else:
            self.fname_rc[fn_short] = {'rc': rc, 'recon_flag': recon_flag, 'full_path': full_path}

    def remove_fname_rc(self, fn_short):
        self.fname_rc.pop(fn_short, f'{fn_short} not exist')

    def update_rotation_center(self):
        try:
            rc = float(self.tx_rc_mod.text())
            item = self.lst_prj_file.selectedItems()
            n = len(item)
            for i in range(n):
                fn_short = item[i].text()
                fn_short = fn_short.split(':')[0]
                if fn_short in self.fname_rc.keys():
                    self.update_fname_rc(fn_short, rc, None)
            self.update_list()
            self.msg = f'rotation center updated for {fn_short}: {rc}'
        except Exception as err:
            print(err)
            self.msg = 'rotation center should be a float number'
        finally:
            self.update_msg()


    def check_fname_rc_states(self, fn_short):
        current_states = self.fname_rc[fn_short]
        if type(current_states) is dict:
            current_rc = current_states.get('rc')
            recon_flag = current_states.get('recon_flag')
        else:
            current_rc = current_states
            recon_flag = None
            self.fname_rc[fn_short] = {'rc': current_rc, 'recon_flag': recon_flag}

        return current_rc, recon_flag

    def auto_batch_rotation_center(self):
        self.pb_rc_auto_batch.setEnabled(False)
        self.pb_rc_auto_batch.setText('wait ...')
        QApplication.processEvents()
        try:
            n_list = self.lst_prj_file.count()
            self.img_recon_slice = []
            self.img_recon_slice_crop = []
            self.fname_rc_batch = {}

            for i in range(n_list):
                self.pb_rc_auto_batch.setText(f'{i+1}/{n_list}')
                QApplication.processEvents()
                item = self.lst_prj_file.item(i)
                tx_current = item.text()  # e.g. 'fly_scan_123:  640'
                fn_short = tx_current.split(':')[0]
                fn = self.fname_rc[fn_short]['full_path']
                if self.chkbox_new_file.isChecked():
                    rc, recon_flag = self.check_fname_rc_states(fn_short)
                    if rc: # exist
                        continue
                #rc, sino_norm, theta, sli = self.auto_rotation_center(fn)
                rc = self.auto_rotation_center(fn)
                if fn_short in self.fname_rc.keys():
                    self.update_fname_rc(fn_short, rc, None)
                self.fname_rc_batch[fn_short] = {'rc': rc}
                self.msg = f'{fn_short}:  {rc:4.2f}'
                self.update_msg()
                self.update_fname_rc(fn_short, rc, None)
                self.update_list()
                QApplication.processEvents()
                '''
                rec = tomopy.recon(sino_norm, theta, rc, algorithm='gridrec')
                rec = tomopy.circ_mask(rec, axis=0, ratio=0.9)[0]
                self.tx_sli_id.setText(str(sli))
                QApplication.processEvents()
                self.img_recon_slice.append(rec)
                
                s = rec.shape
                c = s[-1] // 2
                r_s = max(c - 200, 0)
                r_e = min(c + 200, s[-1])
                self.img_recon_slice_crop.append(rec[r_s:r_e, r_s:r_e])

            #self.img_recon_slice = np.array(self.img_recon_slice)
            #self.update_list()
            if len(self.img_recon_slice):
                #self.img_recon_slice = tomopy.circ_mask(self.img_recon_slice, axis=0, ratio=0.9)
                if self.cb1.findText('Batch recon at single slice') < 0:
                    self.cb1.addItem('Batch recon at single slice')
                if self.cb1.findText('Batch recon at single slice (crop)') < 0:
                    self.cb1.addItem('Batch recon at single slice (crop)')

                    self.cb1.setCurrentText('Batch recon at single slice (crop)')
                self.update_canvas_img()
                self.msg = 'auto center search finished '
            else:
                self.msg = 'No file processed'
            '''
        except Exception as err:
            self.msg = str(err)
        finally:
            self.update_msg()
            print(self.msg)
            self.pb_rc_auto_batch.setEnabled(True)
            self.pb_rc_auto_batch.setText('Auto center')
            QApplication.processEvents()


    def auto_rotation_center(self, fn):
        attr_proj = self.tx_h5_prj.text()
        attr_flat = self.tx_h5_flat.text()
        attr_dark = self.tx_h5_dark.text()
        attr_angle = self.tx_h5_ang.text()
        attr_xeng = self.tx_h5_xeng.text()
        attr_sid = self.tx_h5_sid.text()
        sli = int(self.tx_sli_id.text())
        with h5py.File(fn, 'r') as hf:
            try:
                ang = np.array(hf[attr_angle])  # in unit of degrees
                img0 = np.array(list(hf[attr_proj][0]))
                s = img0.shape

                img_flat = np.array(hf[attr_flat])
                if len(img_flat.shape) == 2:
                    img_flat = np.expand_dims(img_flat, axis=0)
                img_flat_avg = np.median(img_flat, axis=0)

                img_dark = np.array(hf[attr_dark])
                if len(img_dark.shape) == 2:
                    img_dark = np.expand_dims(img_dark, axis=0)
                img_dark_avg = np.median(img_dark, axis=0)

                if np.abs(ang[0]) < np.abs(ang[0] - 90):  # e.g, rotate from 0 - 180 deg
                    tmp = np.abs(ang - ang[0] - 180).argmin()
                else:  # e.g.,rotate from -90 - 90 deg
                    tmp = np.abs(ang - np.abs(ang[0])).argmin()
                
                img180_raw = np.array(list(hf[attr_proj][tmp]))
                img0 = (img0 - img_dark_avg)/(img_flat_avg - img_dark_avg)
                img180_raw = (img180_raw - img_dark_avg) / (img_flat_avg - img_dark_avg)
                im0 = -np.log(img0)
                im0[np.isnan(im0)] = 0
                im1 = -np.log(img180_raw[:, ::-1])
                im1[np.isnan(im1)] = 0
                im0 = img_smooth(im0, 3).squeeze()
                im1 = img_smooth(im1, 3).squeeze()
                sr = StackReg(StackReg.TRANSLATION)
                tmat = sr.register(im0, im1)
                rshft = -tmat[1, 2]
                cshft = -tmat[0, 2]
                rot_cen = s[1] / 2 + cshft / 2 - 1

                #rot_cen = find_cen(fn)
                if sli == 0:
                    sli = s[1]//2

                '''
                sino_prj = np.array(list(hf[attr_proj][:, sli:sli+1]))
                sino_flat = np.array(hf[attr_flat][:, sli:sli+1])
                sino_flat = np.median(sino_flat, axis=0, keepdims=True)
                sino_dark = np.array(hf[attr_dark][:, sli:sli + 1])
                sino_dark = np.median(sino_dark, axis=0, keepdims=True)
                sino_norm = (sino_prj - sino_dark) / (sino_flat - sino_dark)
                sino_norm = -np.log(sino_norm)
                sino_norm[np.isnan(sino_norm)] = 0
            
                p = int(self.tx_rc_ring_remove.text())
                if self.cb_rm_strip.currentText() == 'wavelet':
                    fw_level = p
                    sino_norm = tomopy.prep.stripe.remove_stripe_fw(sino_norm, level=fw_level)
                else:
                    snr = p
                    sino_norm = tomopy.prep.stripe.remove_all_stripe(sino_norm, snr=snr)
                
                theta = ang / 180. * np.pi
                '''
            except Exception as err:
                rot_cen = 0
                sino = None
                ang = None
                self.msg = str(err)
        
        #return rot_cen, sino_norm, theta, sli
        return rot_cen


    def save_rotation_center_tmp(self):
        if len(self.fn_rc_tmp):
            fn = self.fn_rc_tmp
        else:
            fpath_current = self.tx_prj_path.text().strip(' ')
            fn = fpath_current + '/rc_tmp.json'
        try:
            if fn[-5:] != '.json':
                fn += '.json'
        except:
            fn += '.json'
        dict_to_save = self.fname_rc.copy()
        with open(fn, 'w') as json_file:
            json_file.write(json.dumps(dict_to_save, indent=4))
                

    def save_rotation_center(self):
        options = QFileDialog.Option()
        options |= QFileDialog.DontUseNativeDialog
        file_type = 'json files (*.json)'
        try:
            fn, _ = QFileDialog.getSaveFileName(self, 'Save File', "", file_type, options=options)
            if fn:                
                try:
                    if fn[-5:] != '.json':
                        fn += '.json'
                except:
                    fn += '.json'
                self.fn_rc_tmp = fn
                dict_to_save = self.fname_rc.copy()
                with open(fn, 'w') as json_file:
                    json_file.write(json.dumps(dict_to_save, indent=4))
                self.msg = f'saved to {fn}'
        except Exception as err:
            self.msg = str(err)
        finally:
            self.update_msg()
            QApplication.processEvents()


    def load_rotation_center(self):
        options = QFileDialog.Option()
        options |= QFileDialog.DontUseNativeDialog
        file_type = 'json files (*.json)'
        try:
            fn, _ = QFileDialog.getOpenFileName(pytomo, "QFileDialog.getOpenFileName()", "", file_type, options=options)
            if fn:
                with open(fn, 'r') as jf:
                    dict_loaded = json.load(jf)
                keys = list(dict_loaded.keys())
                tx = dict_loaded[keys[0]]['full_path'].split('/')
                root_path = '/'.join(t for t in tx[:-1])
                self.tx_prj_path.setText(root_path)
                file_dict = {}
                for key in keys:
                    file_dict[key] = dict_loaded[key]
                self.msg = f'load rotation center from "{fn}"'
                self.fname_rc.update(file_dict)
                self.fname_rc = dict(sorted(self.fname_rc.items()))
                self.update_list()

                # update self.file_loaded
                for key in keys:
                    fn = file_dict[key]['full_path']
                    if not fn in self.file_loaded:
                        self.file_loaded.append(fn)
        except Exception as err:
            self.msg = str(err)
        finally:
            self.update_msg()
            QApplication.processEvents()


    def preset_h5(self, mode):
        if mode == 'fly':
            tx_proj = 'img_tomo'
            tx_dark = 'img_dark'
            tx_bkg = 'img_bkg'
            tx_ang = 'angle'
            tx_eng = 'X_eng'
            tx_sid = 'scan_id'
        if mode == 'zfly':
            tx_proj = 'Exchange/data'
            tx_dark = 'Exchange/dark'
            tx_bkg = 'Exchange/flat'
            tx_ang = 'Exchange/angle'
            tx_eng = 'Experiment/X_eng (keV)'
            tx_sid = 'Experiment/scan_id'
        self.tx_h5_prj.setText(tx_proj)
        self.tx_h5_dark.setText(tx_dark)
        self.tx_h5_flat.setText(tx_bkg)
        self.tx_h5_ang.setText(tx_ang)
        self.tx_h5_xeng.setText(tx_eng)
        self.tx_h5_sid.setText(tx_sid)
        QApplication.processEvents()


    def view_projection_images(self):
        try:
            item = self.lst_prj_file.selectedItems()
            #file_path = self.tomo_file['file_path']
            self.pb_rc_view_prj.setEnabled(False)
            self.pb_rc_view_prj.setText('wait ...')
            QApplication.processEvents()
            fn_short = item[0].text()
            fn_short = fn_short.split(':')[0]
            fn = self.fname_rc[fn_short]['full_path']
            self.load_proj_file(fn)

            sup_title = f'{fn_short}:     {self.img_eng:2.4f} keV'
            self.canvas1.sup_title = sup_title

            if self.exist_prj_norm and self.cb1.findText('Projection (norm)') < 0:
                self.cb1.addItem('Projection (norm)')
                self.cb1.setCurrentText('Projection (norm)')
            if self.exist_prj and self.cb1.findText('Projection') < 0:
                self.cb1.addItem('Projection')
            if self.exist_dark and self.cb1.findText('Dark field') < 0:
                self.cb1.addItem('Dark field')
            if self.exist_flat and self.cb1.findText('Flat field') < 0:
                self.cb1.addItem('Flat field')
            self.update_canvas_img()

            # set default start and stop for finding rotation center
            s = self.img_prj.shape
            self.tx_rc_start.setText(f'{s[-1]//2 - 30}')
            self.tx_rc_stop.setText(f'{s[-1] // 2 + 30}')
            self.tx_rc_steps.setText('30')
            self.tx_sli_id.setText(f'{s[1]//2}')
            QApplication.processEvents()
            self.msg = ''
        except Exception as err:
            self.msg = str(err)
        finally:
            self.pb_rc_view_prj.setEnabled(True)
            self.pb_rc_view_prj.setText('View proj.')
            self.update_msg()

    def view_projection_images_1st(self):
        try:
            self.pb_rc_view_prj_1st.setEnabled(False)
            QApplication.processEvents()
            item = self.lst_prj_file.selectedItems()
            fn_short = item[0].text()
            fn_short = fn_short.split(':')[0]
            fn = self.fname_rc[fn_short]['full_path']
            attr_proj = self.tx_h5_prj.text()
            attr_flat = self.tx_h5_flat.text()
            attr_dark = self.tx_h5_dark.text()
            with h5py.File(fn, 'r') as hf:
                img_prj = np.array(hf[attr_proj][0])
                img_flat = np.median(np.array(hf[attr_flat]), axis=0)
                img_dark = np.median(np.array(hf[attr_dark]), axis=0)
            img_norm = (img_prj - img_dark) / (img_flat - img_dark)
            plt.figure(figsize=(18, 6))
            plt.subplot(121)
            plt.imshow(img_prj)
            plt.colorbar()

            plt.subplot(122)
            plt.imshow(img_norm)
            plt.colorbar()
            plt.suptitle(f'{fn_short}\n{fn}')
            plt.show()
        except Exception as err:
            print(err)
            self.msg = str(err)
            self.update_msg()
        finally:
            self.pb_rc_view_prj_1st.setEnabled(True)

    def load_proj_file(self, fn):
        attr_proj = self.tx_h5_prj.text()
        attr_flat = self.tx_h5_flat.text()
        attr_dark = self.tx_h5_dark.text()
        attr_angle = self.tx_h5_ang.text()
        attr_xeng = self.tx_h5_xeng.text()
        attr_sid = self.tx_h5_sid.text()

        with h5py.File(fn, 'r') as hf:
            try:
                self.img_prj = np.array(hf[attr_proj])
                self.exist_prj = True
            except:
                self.exist_prj = False
                self.img_prj = np.zeros((1, 100, 100))
            try:
                self.img_flat = np.array(hf[attr_flat])
                if len(self.img_flat.shape) == 2:
                    self.img_flat = np.expand_dims(self.img_flat, axis=0)
                self.img_flat_avg = np.median(self.img_flat, axis=0, keepdims=True)
                self.exist_flat = True
            except:
                self.exist_flat = False
                self.img_flat = 1
                self.img_flat_avg = 1
            try:
                self.img_dark = np.array(hf[attr_dark])
                if len(self.img_dark.shape) == 2:
                    self.img_dark = np.expand_dims(self.img_dark, axis=0)
                self.img_dark_avg = np.median(self.img_dark, axis=0, keepdims=True)
                self.exist_dark = True
            except:
                self.img_dark = 0
                self.img_dark_avg = 0
                self.exist_dark = False
            try:
                self.img_angle = np.array(hf[attr_angle]) # in unit of degrees
            except:
                self.img_angle = []
            if self.exist_prj:
                self.img_prj_norm = (self.img_prj - self.img_dark_avg) / (self.img_flat_avg - self.img_dark_avg)
                self.exist_prj_norm = True
            else:
                self.img_prj_norm = np.zeros((1, 100, 100))
                self.exist_prj_norm = False
            try:
                self.img_eng = np.array(hf[attr_xeng])
            except:
                self.img_eng = ''
            try:
                self.img_sid = np.array(hf[attr_sid])
            except:
                self.img_sid = ''



    def recon_single_file_slices(self, fn, sli, fsave_flag, fsave_root, fsave_prefix, return_flag):
        attr_proj = self.tx_h5_prj.text()
        attr_flat = self.tx_h5_flat.text()
        attr_dark = self.tx_h5_dark.text()
        attr_angle = self.tx_h5_ang.text()
        attr_xeng = self.tx_h5_xeng.text()
        attr_sid = self.tx_h5_sid.text()
        block_list = self._get_block_list()
        auto_block_list = self._get_block_list_auto()
        denoise_flag = int(self.tx_rc_denoise.text())
        dark_scale = int(self.tx_rc_dark_scale.text())
        binning = int(self.tx_rec_bin.text())
        algorithm = self.cb_rc_algorithm.currentText()
        options = {}
        if algorithm != 'gridrec':               
            if algorithm == 'astra_fbp':
                method = 'FBP_CUDA'
            elif algorithm == 'astra_sirt':
                method = 'SIRT_CUDA'
            num_iter = int(self.tx_rc_algorithm_iter.text())
            extra_options = {'MinConstraint': 0}                     
            options = {
                        'proj_type': 'cuda',
                        'method': method,
                        'num_iter': num_iter,
                        'extra_options': extra_options
                      }

        p = int(self.tx_rc_ring_remove.text())
        if self.cb_rm_strip.currentText() == 'wavelet':
            fw_level = p
            snr = 0
        else:
            fw_level = 0
            snr = p

        roi_c = self.tx_rec_roi_c.text()
        roi_c = extract_range(roi_c, 'int')
        roi_s = self.tx_rec_roi_s.text()
        roi_s = extract_range(roi_s, 'int')
        mask_r = float(self.tx_rec_cir_mask_ratio.text())
               
        
        fn_short = fn.split('/')[-1]
        rc, recon_flag = self.check_fname_rc_states(fn_short)
        if rc == 0:
            rc = self.auto_rotation_center(fn)
            self.msg = f'No center assigned. Automatically find center = {rc:4.2f}'
            self.update_msg(); QApplication.processEvents()
        ml_param = self.ml_get_param()
        rec, fsave = recon_and_save(fn, rc,
                       attr_proj, attr_flat, attr_dark, attr_angle, attr_xeng, attr_sid,
                       sli, binning, block_list, dark_scale,
                       denoise_flag, snr, fw_level,
                       algorithm=algorithm,
                       circ_mask_ratio=mask_r,
                       options = options,
                       fsave_flag=fsave_flag,
                       fsave_root=fsave_root,
                       fsave_prefix=fsave_prefix,
                       roi_cen=roi_c,
                       roi_size=roi_s,
                       return_flag=return_flag,
                       ml_param=ml_param,
                       auto_block_list=auto_block_list
                       )
        if return_flag:
            return rec, fsave, rc
        else:
            return 0, 0, 0


    def recon_1_slice(self):
        self.pb_rc_1_sli.setEnabled(False)
        QApplication.processEvents()
        sli = self.tx_sli_id.text()
        if len(sli) == 0:
            sli = []
        else:
            sli = [int(sli)]

        try:
            n_list = self.lst_prj_file.count()
            if n_list == 0:
                self.msg = 'no files selected'
            self.img_recon_slice = []
            self.img_recon_slice_crop = []
            self.fname_rc_batch = {}
            for i in range(n_list):
                self.pb_rc_1_sli.setText(f'recon {i + 1}/{n_list}')
                QApplication.processEvents()
                item = self.lst_prj_file.item(i)
                tx_current = item.text()  # e.g. 'fly_scan_123:  640'
                fn_short = tx_current.split(':')[0]
                fn = self.fname_rc[fn_short]['full_path']
                rc, recon_flag = self.check_fname_rc_states(fn_short)
                if rc == 0:
                    continue
                
                rec, _, _ = self.recon_single_file_slices(fn, sli, fsave_flag=False, fsave_root='.',
                                                     fsave_prefix='rec', return_flag=1)
                rec = tomopy.circ_mask(rec, axis=0, ratio=0.9)
                rec = rec[0]
                if sli == [] or sli == [0]:
                    sli = [rec.shape[0]//2]
                    self.tx_sli_id.setText(str(rec.shape[0]//2))
                    QApplication.processEvents()
                self.fname_rc_batch[fn_short] = {'rc': rc}
                self.img_recon_slice.append(rec)  # this is list
                if i == 0:
                    s = rec.shape
                    c = s[-1] // 2
                    r_s = max(c - 200, 0)
                    r_e = min(c + 200, s[-1])
                self.img_recon_slice_crop.append(rec[r_s:r_e, r_s:r_e])

            if len(self.img_recon_slice):
                if self.cb1.findText('Batch recon at single slice') < 0:
                    self.cb1.addItem('Batch recon at single slice')
                if  self.cb1.findText('Batch recon at single slice (crop)') < 0:
                    self.cb1.addItem('Batch recon at single slice (crop)')
                    self.cb1.setCurrentText('Batch recon at single slice (crop)')
                self.update_canvas_img()
        except Exception as err:
            self.msg = str(err)
            exc_type, exc_value, exc_traceback = sys.exc_info()
            print(exc_traceback.tb_lineno)
        finally:
            self.update_msg()
            print(self.msg)
            self.pb_rc_1_sli.setEnabled(True)
            self.pb_rc_1_sli.setText('recon 1 silice')
            QApplication.processEvents()



    def recon_single_file_core(self, fn):
        sli = self.tx_rec_sli.text()
        sli = extract_range(sli, 'int')
        tx = fn.split('/')
        fn_short = fn.split('/')[-1]
        fsave_root = '/'.join(t for t in tx[:-1])
        fsave_prefix = (fn_short.split('.')[0]).split('_')[-1]
        fsave_flag = True

        rec, fsave, rc = self.recon_single_file_slices(fn, sli, fsave_flag, fsave_root,
                                                                     fsave_prefix, return_flag=True)
        return rec, fsave, rc

    def set_single_selection(self):
        self.msg = 'Uncheck "Enable multi-selection", and then select the file to reconstruct'
        self.update_msg()
        self.chkbox_multi_selec.setChecked(False)
        self.lst_prj_file.clearSelection()
        QApplication.processEvents()

    def recon_single_file(self, plot_flag):
        try:
            if self.chkbox_multi_selec.isChecked():
                self.set_single_selection()
                return 0

            self.pb_rec_s.setEnabled(False)
            self.pb_rec_s.setText('Wait ...')
            QApplication.processEvents()
            '''
            sli = self.tx_rec_sli.text()
            sli = extract_range(sli, 'int')
            '''
            item = self.lst_prj_file.selectedItems()
            fn_short = item[0].text()
            fn_short = fn_short.split(':')[0]
            fn = self.fname_rc[fn_short]['full_path']
            self.current_file = fn
            self.current_file_short = fn_short
            '''
            tx = fn.split('/')
            fsave_root = '/'.join(t for t in tx[:-1])
            fsave_prefix = (fn_short.split('.')[0]).split('_')[-1]
            fsave_flag = True

            self.img_rec_tomo, fsave, rc = self.recon_single_file_slices(fn, sli, fsave_flag, fsave_root,
                                                                   fsave_prefix, return_flag=True)
            '''
            self.img_rec_tomo, fsave, rc = self.recon_single_file_core(fn)
            recon_flag = f'Y: {fsave}'
            self.update_fname_rc(fn_short, rc, recon_flag)
            self.update_list()
            QApplication.processEvents()
            if plot_flag:
                if exist_napari and self.chkbox_napari.isChecked():
                    napari.view_image(self.img_rec_tomo)
                fsave_short = fsave.split('/')[-1]
                sup_title = fsave_short
                self.canvas1.sup_title = sup_title
                if self.cb1.findText('3D tomo') < 0:
                    self.cb1.addItem('3D tomo')
                    self.cb1.setCurrentText('3D tomo')
                self.update_canvas_img()
            self.msg = f'{fn_short}: reconstruction finished'
        except Exception as err:
            self.msg = str(err)
        finally:
            self.update_msg()
            self.pb_rec_s.setEnabled(True)
            self.pb_rec_s.setText('Recon single')


    def recon_batch_file(self):
        self.pb_rec_b.setEnabled(False)
        self.pb_rec_b.setText('Wait ...')
        QApplication.processEvents()
        try:
            items = self.lst_prj_file.selectedItems()
            n_list = len(items)
            for i in range(n_list):
                try:
                    tx = items[i].text()
                    fn_short = tx.split(':')[0]
                    fn = self.fname_rc[fn_short]['full_path']
                    self.pb_rec_b.setText(f'Processing {i + 1}/{n_list}')
                    self.msg = f'reconstructing {i + 1}/{n_list}: {fn_short}'
                    self.update_msg()
                    items[i].setText(tx + '  <--')
                    QApplication.processEvents()
                    rec, fsave, rc = self.recon_single_file_core(fn)
                    items[i].setText(tx)
                    recon_flag = f'Y: {fsave}'
                    self.update_fname_rc(fn_short, rc, recon_flag)
                    self.update_list()
                    self.save_rotation_center_tmp()
                    QApplication.processEvents()
                    self.msg = 'reconstruction finished'
                except Exception as err:
                    self.msg = f'fails on {fn_short}: {err}'
                    self.update_msg()

        except Exception as err:
            print(err)
            self.msg = str(err)
        finally:
            self.pb_rec_b.setEnabled(True)
            self.pb_rec_b.setText('Recon batch')
            self.update_msg()
            QApplication.processEvents()

    def view_3D_recon(self):
        try:
            if self.chkbox_multi_selec.isChecked():
                self.set_single_selection()
                return 0
            self.pb_rec_view.setEnabled(False)
            self.pb_rec_view_copy.setEnabled(False)
            QApplication.processEvents()
            item = self.lst_prj_file.selectedItems()
            fn_short = item[0].text()
            fn_short = fn_short.split(':')[0]
            recon_flag = self.fname_rc[fn_short]['recon_flag']
            if not recon_flag is None:  # load reconstructed file
                fn_recon = recon_flag.split(':')[-1].strip(' ')                
            else:  # load any 3D file
                file_root_path = self.tx_prj_path.text().strip(' ')
                fn_recon = file_root_path + '/' + fn_short

            file_type = fn_recon.split('.')[-1]
            print(file_type)
            print(fn_recon)
            if file_type == 'tiff' or file_type == 'tif':
                self.img_rec_tomo = io.imread(fn_recon)
            elif file_type == 'h5':
                with h5py.File(fn_recon, 'r') as hf:
                    self.img_rec_tomo = np.array(hf['img'])
            else:
                self.msg = 'fail in loading image file'
                self.update_msg()
                return 0
            if exist_napari and self.chkbox_napari.isChecked():
                napari.view_image(self.img_rec_tomo)
            fn_recon_short = fn_recon.split('/')[-1]
            sup_title = fn_recon_short
            self.canvas1.sup_title = sup_title
            if self.cb1.findText('3D tomo') < 0:
                self.cb1.addItem('3D tomo')
                self.cb1.setCurrentText('3D tomo')
            self.update_canvas_img()
            self.msg = f'view {fn_recon_short}'                     

        except Exception as err:
            self.msg = str(err)
        finally:
            self.update_msg()
            self.pb_rec_view.setEnabled(True)
            self.pb_rec_view_copy.setEnabled(True)
            QApplication.processEvents()





    def ml_collect_auto_threshold_param(self):
        filt_sz = int(self.tx_ml_filt_sz.text())
        filt_iter = int(self.tx_ml_filt_iter.text())
        filt_bins = int(self.tx_ml_filt_bins.text())
        medfilt_sz = int(self.tx_ml_medfilt_sz.text()) # for median filter
        filt_param = {}
        filt_param['sz'] = filt_sz
        filt_param['iter'] = filt_iter
        filt_param['bins'] = filt_bins
        filt_param['medfilt_sz'] = medfilt_sz
        return filt_param


    def ml_mdedfilt(self):
        try:
            self.lb_ml_model_rec_msg.setText('')
            filt_param = self.ml_collect_auto_threshold_param()
            medfilt_sz = filt_param['medfilt_sz']
            img3D, fn_short = self.get_tomo_image_from_file()
            s = img3D.shape
            sli = s[0] // 2
            cur_img = img3D[sli]
            img_d = medifilt_3D(cur_img, medfilt_sz).squeeze()
            img_c = np.concatenate((cur_img, img_d), axis=1)
            plt.figure(figsize=(12, 6))
            plt.imshow(img_c)
            plt.title(f'Median filter \n{fn_short}')
            plt.colorbar()
            plt.show()
        except Exception as err:
            print(err)
            self.lb_ml_model_rec_msg.setText(str(err))
            QApplication.processEvents()

    def ml_auto_thresh_bkg(self):
        try:
            self.lb_ml_model_rec_msg.setText('')
            filt_param = self.ml_collect_auto_threshold_param()
            filt_sz = filt_param['sz']
            filt_iter = filt_param['iter']
            filt_bins = filt_param['bins']
            img3D, fn_short = self.get_tomo_image_from_file()
            s = img3D.shape
            sli = s[0] // 2
            cur_img = img3D[sli]
            img_d = ostu_mask_3D(cur_img, filt_sz, filt_iter, filt_bins).squeeze()
            img_c = np.concatenate((cur_img, img_d), axis=1)
            plt.figure(figsize=(12, 6))
            plt.imshow(img_c)
            plt.title(f'Bkg. removal\n{fn_short}')
            plt.colorbar()
            plt.show()
        except Exception as err:
            print(err)
            self.lb_ml_model_rec_msg.setText(str(err))
            QApplication.processEvents()


    def ml_select_denoise_methode(self):
        item_name = self.cb_ml_choice.currentText()
        item = QListWidgetItem(item_name)
        self.lst_ml_choice.addItem(item)

    def ml_reset_denoise_methode(self):
        self.lst_ml_choice.clear()

    def get_tomo_image_from_file(self):
        item = self.lst_prj_file.selectedItems()
        if len(item) > 0:
            fn_short = item[0].text()
            fn_short = fn_short.split(':')[0]
            fn_recon = self.fname_rc[fn_short]['full_path']
            file_type = fn_recon.split('.')[-1]
            if file_type == 'tiff' or file_type == 'tif':
                img_rec_tomo = io.imread(fn_recon)
            elif file_type == 'h5':
                with h5py.File(fn_recon, 'r') as hf:
                    img_rec_tomo = np.array(hf['img'])
            return img_rec_tomo, fn_short
        else:
            return None, None

    def ml_assemble_denoise_seq(self):
        ml_denoise_seq = []
        n = self.lst_ml_choice.count()
        for i in range(n):
            item = self.lst_ml_choice.item(i)
            txt = item.text()
            ml_denoise_seq.append(txt)
        return ml_denoise_seq


    def ml_3D_denoise_slice(self):
        self.lb_ml_model_rec_msg.setText('')
        err_msg = ''
        try:
            ml_denoise_seq = self.ml_assemble_denoise_seq()
            if len(ml_denoise_seq) == 0:
                self.lb_ml_model_rec_msg.setText('No denoising methods added')
            img3D, fn_short = self.get_tomo_image_from_file()
            if img3D is None:
                err_msg = 'Fail to extract image, check if recon_file is selected'
            s = img3D.shape
            try:
                sli = int(self.tx_ml_test_sli.text())
                if sli < 0 or sli > s[0] - 1:
                    sli = s[0] // 2
            except:
                sli = s[0] // 2
            img = img3D[sli:sli+1]
            img_d, img_comb = self.ml_apply_denoise_seq(img, ml_denoise_seq)
            plt.figure(figsize=(14, 6))
            plt.imshow(img_comb.squeeze())
            plt.colorbar()
            tit = ' + '.join(f'"{k}"' for k in ml_denoise_seq)
            plt.title(f'Denoise on slice {sli}\n{tit}')
            plt.show()
        except Exception as err:
            print(err)
            self.lb_ml_model_rec_msg.setText(err_msg + '\n' + str(err))
        finally:
            QApplication.processEvents()


    def ml_apply_denoise_seq(self, img3D, ml_denoise_seq):
        self.lb_ml_model_rec_msg.setText('')
        filt_param = self.ml_collect_auto_threshold_param()
        filt_sz = filt_param['sz']
        filt_iter = filt_param['iter']
        filt_bins = filt_param['bins']
        medfilt_sz = filt_param['medfilt_sz']
        if len(img3D.shape) == 2:
            img3D = img3D[np.newaxis]
        img_d = img3D.copy()
        img_comb = img_d.copy()
        n = len(ml_denoise_seq)
        for i, method in enumerate(ml_denoise_seq):
            if 'Median' in method:
                print(f'\n{i + 1}/{n}: Apply median filter ...')
                img_d, img_comb = medifilt_3D(img_d, medfilt_sz)
            elif 'Bkg' in method:
                print(f'\n{i + 1}/{n}: Apply Bkg. removal ...')
                img_d, img_comb = ostu_mask_3D(img_d, filt_sz, filt_iter, filt_bins)
            elif 'ML' in method:
                print(f'\n{i + 1}/{n}: Apply ML denoising ...')
                model_name, model_path, device = self.ml_select_3D_denosie_model()
                img_d, img_comb = apply_ML_tomo(img3D, model_name, model_path, device=device)
            self.lb_ml_model_rec_msg.setText(f'Applying {method} ...')
            QApplication.processEvents()
        return img_d, img_comb


    def ml_select_3D_denosie_model(self):
        device = self.cb_ml_device_rec.currentText()
        model_path = self.tx_ml_model_path.text()
        model_name = self.cb_ml_model_sel.currentText()
        return model_name, model_path, device


    def ml_3D_denosie_recon(self):
        try:
            self.pb_rec_ml_enoise.setEnabled(False)
            self.pb_rec_ml_enoise.setText('wait ...')
            self.lb_ml_model_rec_msg.setText('')
            QApplication.processEvents()

            img3D, fn_short = self.get_tomo_image_from_file()
            ml_denoise_seq = self.ml_assemble_denoise_seq()
            img_d, img_comb = self.ml_apply_denoise_seq(img3D, ml_denoise_seq)

            self.img_rec_tomo = img3D
            self.img_rec_tomo_denoise = img_d.copy()
            self.img_rec_tomo_denoise_compare = img_comb.copy()
            sup_title = fn_short
            self.canvas1.sup_title = sup_title
            if self.cb1.findText('3D tomo denoised') < 0:
                self.cb1.addItem('3D tomo denoised')
                self.cb1.setCurrentText('3D tomo denoised')
            if self.cb1.findText('3D tomo denoise compare') < 0:
                self.cb1.addItem('3D tomo denoise compare')
            self.update_canvas_img()
            self.pb_rec_ml_enoise.setEnabled(True)
            self.lb_ml_model_rec_msg.setText('"3D tomo denoised" has been added for plot ')
            QApplication.processEvents()

        except Exception as err:
            print(err)
            self.lb_ml_model_rec_msg.setText(str(err))
        finally:
            self.pb_rec_ml_enoise.setText('ML denoise on recon')
            self.pb_rec_ml_enoise.setEnabled(True)
            QApplication.processEvents()

    def enable_multi_selection(self):
        if  self.chkbox_multi_selec.isChecked():
            self.lst_prj_file.setSelectionMode(QAbstractItemView.MultiSelection)
            self.pb_filter_recon.setEnabled(True)
            self.pb_filter_rotcen.setEnabled(True)
            self.pb_filter_un_select.setEnabled(True)
            self.pb_filter_all.setEnabled(True)
            self.pb_filter_reverse.setEnabled(True)
        else:
            self.lst_prj_file.setSelectionMode(QAbstractItemView.SingleSelection)
            self.lst_prj_file.clearSelection()
            self.pb_filter_recon.setEnabled(False)
            self.pb_filter_rotcen.setEnabled(False)
            self.pb_filter_un_select.setEnabled(False)
            self.pb_filter_all.setEnabled(False)
            self.pb_filter_reverse.setEnabled(False)


    def ring_remove(self):
        tx = self.cb_rm_strip.currentText()
        if tx == 'wavelet':
            self.lb_rm_ring.setText('fw level:')
            self.tx_rc_ring_remove.setText('4')
        else:
            self.lb_rm_ring.setText('snr:')
            self.tx_rc_ring_remove.setText('3')


    def layout_canvas(self):
        from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
        lb_empty = QLabel()
        lb_empty2 = QLabel()
        lb_empty2.setFixedWidth(10)
        self.canvas1 = MyCanvas(obj=self)
        self.toolbar = NavigationToolbar(self.canvas1, self)
        self.sl1 = QScrollBar(QtCore.Qt.Horizontal)
        self.sl1.setMaximum(0)
        self.sl1.setMinimum(0)
        self.sl1.valueChanged.connect(lambda:self.sliderval(self.canvas1))

        self.cb1 = QComboBox()
        self.cb1.setFont(self.font2)
        self.cb1.setFixedWidth(620)
        self.cb1.currentIndexChanged.connect(self.update_canvas_img)

        self.pb_save_img_stack = QPushButton('Save image stack')
        self.pb_save_img_stack.setFont(self.font2)
        self.pb_save_img_stack.clicked.connect(lambda:self.save_img_stack(self.canvas1))
        self.pb_save_img_stack.setFixedWidth(150)

        self.pb_save_img_single = QPushButton('Save current image')
        self.pb_save_img_single.setFont(self.font2)
        self.pb_save_img_single.clicked.connect(self.save_img_single)
        self.pb_save_img_single.setFixedWidth(150)

        hbox_can_l = QHBoxLayout()
        hbox_can_l.addWidget(self.cb1)
        hbox_can_l.setAlignment(QtCore.Qt.AlignLeft)

        hbox_can_save = QHBoxLayout()
        hbox_can_save.addWidget(self.cb1)
        hbox_can_save.addWidget(self.pb_save_img_single)
        hbox_can_save.addWidget(self.pb_save_img_stack)
        hbox_can_save.setAlignment(QtCore.Qt.AlignLeft)

        self.lb_x_l = QLabel()
        self.lb_x_l.setFont(self.font2)
        self.lb_x_l.setText('x: ')
        self.lb_x_l.setFixedWidth(80)

        self.lb_y_l = QLabel()
        self.lb_y_l.setFont(self.font2)
        self.lb_y_l.setText('y: ')
        self.lb_y_l.setFixedWidth(80)

        self.lb_z_l = QLabel()
        self.lb_z_l.setFont(self.font2)
        self.lb_z_l.setText('intensity: ')
        self.lb_z_l.setFixedWidth(120)

        lb_cmap = QLabel()
        lb_cmap.setFont(self.font2)
        lb_cmap.setText('colormap: ')
        lb_cmap.setFixedWidth(80)

        cmap = ['gray', 'bone', 'viridis', 'terrain', 'gnuplot', 'bwr', 'plasma', 'PuBu', 'summer', 'rainbow', 'jet']
        self.cb_cmap = QComboBox()
        self.cb_cmap.setFont(self.font2)
        for i in cmap:
            self.cb_cmap.addItem(i)
        self.cb_cmap.setCurrentText('viridis')
        self.cb_cmap.currentIndexChanged.connect(lambda:self.change_colormap(self.canvas1))
        self.cb_cmap.setFixedWidth(80)

        self.pb_adj_cmap = QPushButton('Auto Contrast')
        self.pb_adj_cmap.setFont(self.font2)
        self.pb_adj_cmap.clicked.connect(lambda:self.auto_contrast(self.canvas1))
        self.pb_adj_cmap.setEnabled(True)
        self.pb_adj_cmap.setFixedWidth(120)

        lb_cmax = QLabel()
        lb_cmax.setFont(self.font2)
        lb_cmax.setText('cmax: ')
        lb_cmax.setFixedWidth(40)
        lb_cmin = QLabel()
        lb_cmin.setFont(self.font2)
        lb_cmin.setText('cmin: ')
        lb_cmin.setFixedWidth(40)

        self.tx_cmax = QLineEdit(self)
        self.tx_cmax.setFont(self.font2)
        self.tx_cmax.setFixedWidth(80)
        self.tx_cmax.setText('1.')
        self.tx_cmax.setValidator(QDoubleValidator())
        self.tx_cmax.setEnabled(True)

        self.tx_cmin = QLineEdit(self)
        self.tx_cmin.setFont(self.font2)
        self.tx_cmin.setFixedWidth(80)
        self.tx_cmin.setText('0.')
        self.tx_cmin.setValidator(QDoubleValidator())
        self.tx_cmin.setEnabled(True)

        self.pb_set_cmap = QPushButton('Set')
        self.pb_set_cmap.setFont(self.font2)
        self.pb_set_cmap.clicked.connect(lambda:self.set_contrast(self.canvas1))
        self.pb_set_cmap.setEnabled(True)
        self.pb_set_cmap.setFixedWidth(55)

        hbox_chbx_l = QHBoxLayout()
        hbox_chbx_l.addWidget(self.lb_x_l)
        hbox_chbx_l.addWidget(self.lb_y_l)
        hbox_chbx_l.addWidget(self.lb_z_l)
        hbox_chbx_l.addWidget(lb_empty)
        hbox_chbx_l.addWidget(self.pb_save_img_single)
        hbox_chbx_l.addWidget(self.pb_save_img_stack)
        hbox_chbx_l.setAlignment(QtCore.Qt.AlignLeft)

        hbox_cmap = QHBoxLayout()
        hbox_cmap.addWidget(lb_cmap)
        hbox_cmap.addWidget(self.cb_cmap)
        hbox_cmap.addWidget(self.pb_adj_cmap)
        hbox_cmap.addWidget(lb_cmin)
        hbox_cmap.addWidget(self.tx_cmin)
        hbox_cmap.addWidget(lb_cmax)
        hbox_cmap.addWidget(self.tx_cmax)
        hbox_cmap.addWidget(self.pb_set_cmap)
        hbox_chbx_l.addWidget(lb_empty)
        hbox_cmap.setAlignment(QtCore.Qt.AlignLeft)

        vbox_can1 = QVBoxLayout()
        vbox_can1.addWidget(self.toolbar)
        vbox_can1.addWidget(self.canvas1)
        vbox_can1.addWidget(self.sl1)
        vbox_can1.addLayout(hbox_can_l)
        vbox_can1.addLayout(hbox_chbx_l)
        vbox_can1.addLayout(hbox_cmap)
        vbox_can1.setAlignment(QtCore.Qt.AlignLeft)
        return vbox_can1


    def sliderval(self, current_canvas):
        canvas = current_canvas
        img_index = self.sl1.value()
        canvas.current_img_index = img_index
        canvas.current_img = canvas.img_stack[img_index]
        img = canvas.img_stack[img_index]
        canvas.update_img_one(img, img_index=img_index)


    def change_colormap(self, current_canvas):
        #canvas = self.canvas1
        canvas = current_canvas
        if canvas == self.canvas1:
            cmap = self.cb_cmap.currentText()
        elif canvas == self.canvas_ml:
            cmap = self.cb_ml_cmap.currentText()
        canvas.colormap = cmap
        canvas.colorbar_on_flag = True
        canvas.update_img_one(canvas.current_img, canvas.current_img_index)


    def auto_contrast(self, current_canvas):
        canvas = current_canvas
        cmin, cmax = canvas.auto_contrast()
        self.display_cmin_cmax(cmin, cmax, current_canvas)

    def display_cmin_cmax(self, cmin, cmax, current_canvas):
        if np.abs(cmax) > 1e3 or np.abs(cmax)<1e-3:
            txt_max = f'{cmax:1.3e}'
        else:
            txt_max = f'{cmax:6.3f}'
        if np.abs(cmin) > 1e3 or np.abs(cmin) < 1e-3:
            txt_min = f'{cmin:1.3e}'
        else:
            txt_min = f'{cmin:6.3f}'
        if current_canvas == self.canvas1:
            self.tx_cmax.setText(txt_max)
            self.tx_cmin.setText(txt_min)
        elif current_canvas == self.canvas_ml:
            self.tx_ml_cmax.setText(txt_max)
            self.tx_ml_cmin.setText(txt_min)
        QApplication.processEvents()


    def set_contrast(self, current_canvas):
        try:
            canvas = current_canvas
            if canvas == self.canvas1:
                cmax = np.float32(self.tx_cmax.text())
                cmin = np.float32(self.tx_cmin.text())
            else:
                cmax = np.float32(self.tx_ml_cmax.text())
                cmin = np.float32(self.tx_ml_cmin.text())
            assert cmax >= cmin, "cmax should > cmin"
            canvas.set_contrast(cmin, cmax)
            if canvas == self.canvas1 and canvas.rgb_flag:
                self.xanes_colormix()
        except Exception as err:
            print(err)
            self.msg = str(err)
            self.update_msg()


    def save_img_stack(self, current_canvas):
        try:
            QApplication.processEvents()
            canvas = current_canvas
            save_stack = True
            if canvas == self.canvas1:
                if self.cb1.currentText() == 'Color mix':
                    save_stack = False
                    self.save_img_single()
                else:
                    img_stack = (canvas.img_stack).astype(np.float32)
            elif canvas == self.canvas_ml:
                img_stack = (canvas.img_stack).astype(np.float32)
            if save_stack:
                options = QFileDialog.Option()
                options |= QFileDialog.DontUseNativeDialog
                file_type = 'tif files (*.tiff)'
                fn, _ = QFileDialog.getSaveFileName(self, 'Save File', "", file_type, options=options)
                if fn[-5:] != '.tiff' and fn[-4:]!='.tif':
                    fn += '.tiff'
                io.imsave(fn, img_stack)
                print(f'current image stack has been saved to file: {fn}')
                self.msg = f'image stack saved to: {fn}'
        except Exception as err:
            self.msg = f'file saving fails.  Error: {str(err)}'
        finally:
            self.update_msg()
            if canvas == self.canvas1:
                self.pb_save_img_stack.setEnabled(True)
            elif canvas == self.canvas_ml:
                self.pb_ml_save_img_stack.setEnabled(True)
            QApplication.processEvents()


    def save_img_single(self):
        try:
            self.pb_save_img_single.setEnabled(False)
            QApplication.processEvents()
            canvas = self.canvas1
            cmax = np.float32(self.tx_cmax.text())
            cmin = np.float32(self.tx_cmin.text())
            if self.cb1.currentText() == 'Color mix':
                img = self.img_colormix
                for i in range(img.shape[2]):
                    img[:, :, i] = img[:, :, i]
                plt.figure()
                img = (img - cmin) / (cmax - cmin)
                plt.imshow(img, clim=[cmin, cmax])
                plt.show()
            else:
                img_stack = canvas.img_stack[canvas.current_img_index]
                img_stack = np.array(img_stack, dtype = np.float32)
                options = QFileDialog.Option()
                options |= QFileDialog.DontUseNativeDialog
                file_type = 'tif files (*.tiff)'
                fn, _ = QFileDialog.getSaveFileName(self, 'Save File', "", file_type, options=options)
                if not(fn[-5:] == '.tiff' or fn[-4:] =='.tif'):
                    fn += '.tiff'
                if not fn == '.tiff':
                    io.imsave(fn, img_stack)
                    print(f'current image has been saved to file: {fn}')
                    self.msg = f'current image saved to: {fn}'
                plt.figure()
                plt.imshow(img_stack, clim=[cmin, cmax], cmap=canvas.colormap)
                plt.axis('off')
                plt.show()
        except Exception as err:
            self.msg = f'file saving fails.  Error: {str(err)}'
        finally:
            self.update_msg()
            self.pb_save_img_single.setEnabled(True)
            QApplication.processEvents()


    def check_contrast_range(self):
        pass

    def update_msg(self):
        self.lb_msg.setFont(self.font2)
        self.lb_msg.setText(self.msg)
        self.lb_msg.setStyleSheet('color: rgb(200, 50, 50);')

    def update_canvas_img(self):
        canvas = self.canvas1
        slide = self.sl1
        type_index = self.cb1.currentText()
        QApplication.processEvents()
        canvas.draw_line = False
        self.pb_adj_cmap.setEnabled(True)
        self.pb_set_cmap.setEnabled(True)
        try:
            if type_index == 'Projection':
                self.img_colormix_raw = np.array([])
                canvas.rgb_flag = 0
                canvas.x, canvas.y = [], []
                canvas.axes.clear()  # this is important, to clear the current image before another imshow()
                sh = self.img_prj.shape
                canvas.img_stack = self.img_prj
                canvas.special_info = None
                canvas.current_img_index = self.sl1.value()
                canvas.title = [f'#{i:3d},   {self.img_angle[i]:3.3f} deg' for i in range(len(self.img_angle))]
                canvas.update_img_stack()
                slide.setMaximum(max(sh[0] - 1, 0))
                self.current_image = self.img_prj[0]
                self.auto_contrast(canvas)
            if type_index == 'Projection (norm)':
                self.img_colormix_raw = np.array([])
                canvas.rgb_flag = 0
                canvas.x, canvas.y = [], []
                canvas.axes.clear()  # this is important, to clear the current image before another imshow()
                sh = self.img_prj_norm.shape
                canvas.img_stack = self.img_prj_norm
                canvas.special_info = None
                canvas.current_img_index = self.sl1.value()
                canvas.title = [f'#{i:3d},   {self.img_angle[i]:3.3f} deg' for i in range(len(self.img_angle))]
                canvas.update_img_stack()
                slide.setMaximum(max(sh[0] - 1, 0))
                self.auto_contrast(canvas)
                self.current_image = self.img_prj_norm[0]
            if type_index == 'Dark field':
                self.img_colormix_raw = np.array([])
                canvas.rgb_flag = 0
                canvas.x, canvas.y = [], []
                canvas.axes.clear()  # this is important, to clear the current image before another imshow()
                sh = self.img_dark.shape
                canvas.img_stack = self.img_dark
                canvas.special_info = None
                canvas.current_img_index = self.sl1.value()
                canvas.title = [f'#{i:3d}' for i in range(len(self.img_dark))]
                canvas.update_img_stack()
                slide.setMaximum(max(sh[0] - 1, 0))
                self.current_image = self.img_dark[0]
                self.auto_contrast(canvas)
            if type_index == 'Flat field':
                self.img_colormix_raw = np.array([])
                canvas.rgb_flag = 0
                canvas.x, canvas.y = [], []
                canvas.axes.clear()  # this is important, to clear the current image before another imshow()
                sh = self.img_flat.shape
                canvas.img_stack = self.img_flat
                canvas.special_info = None
                canvas.current_img_index = self.sl1.value()
                canvas.title = [f'#{i:3d}' for i in range(len(self.img_flat))]
                canvas.update_img_stack()
                slide.setMaximum(max(sh[0] - 1, 0))
                self.current_image = self.img_flat[0]
                self.auto_contrast(canvas)
            if type_index == 'Rotation center':
                self.img_colormix_raw = np.array([])
                canvas.rgb_flag = 0
                canvas.x, canvas.y = [], []
                canvas.axes.clear()  # this is important, to clear the current image before another imshow()
                sh = self.img_rc.shape
                canvas.img_stack = self.img_rc
                canvas.special_info = None
                canvas.current_img_index = self.sl1.value()
                canvas.title = [f'{i:3d}:   cen={self.rc[i]}' for i in range(len(self.img_rc))]
                canvas.update_img_stack()
                slide.setMaximum(max(sh[0] - 1, 0))
                self.current_image = self.img_rc[0]
                self.auto_contrast(canvas)
            if type_index == 'Rotation center (crop)':
                self.img_colormix_raw = np.array([])
                canvas.rgb_flag = 0
                canvas.x, canvas.y = [], []
                canvas.axes.clear()  # this is important, to clear the current image before another imshow()
                sh = self.img_rc.shape
                c = sh[-1] // 2
                d = sh[-1] // 4
                r_s = max(c - d, 0)
                r_e = min(c + d, sh[-1])
                img_crop = self.img_rc[:, r_s:r_e, r_s:r_e]
                canvas.img_stack = img_crop
                canvas.special_info = None
                canvas.current_img_index = self.sl1.value()
                canvas.title = [f'{i:3d}:   cen={self.rc[i]}' for i in range(sh[0])]
                canvas.update_img_stack()
                slide.setMaximum(max(sh[0] - 1, 0))
                self.current_image = img_crop[0]
                self.auto_contrast(canvas)
            if type_index == 'Batch recon at single slice':
                self.img_colormix_raw = np.array([])
                canvas.rgb_flag = 0
                canvas.x, canvas.y = [], []
                canvas.axes.clear()  # this is important, to clear the current image before another imshow()
                sh = len(self.img_recon_slice)
                canvas.img_stack = self.img_recon_slice
                canvas.special_info = None
                canvas.current_img_index = self.sl1.value()
                keys = list(self.fname_rc_batch.keys())
                canvas.title = [f'{keys[i]}:   cen={self.fname_rc_batch[keys[i]]["rc"]:4.2f}' for i in range(sh)]
                canvas.update_img_stack()
                slide.setMaximum(max(sh - 1, 0))
                self.current_image = self.img_recon_slice[canvas.current_img_index]
                self.auto_contrast(canvas)
            if type_index == 'Batch recon at single slice (crop)':
                self.img_colormix_raw = np.array([])
                canvas.rgb_flag = 0
                canvas.x, canvas.y = [], []
                canvas.axes.clear()  # this is important, to clear the current image before another imshow()
                sh = len(self.img_recon_slice_crop)
                img_crop = self.img_recon_slice_crop
                canvas.img_stack = img_crop
                canvas.special_info = None
                canvas.current_img_index = self.sl1.value()
                keys = list(self.fname_rc_batch.keys())
                canvas.title = [f'{keys[i]}:   cen={self.fname_rc_batch[keys[i]]["rc"]:4.2f}' for i in range(sh)]
                canvas.update_img_stack()
                slide.setMaximum(max(sh - 1, 0))
                self.current_image = img_crop
                self.auto_contrast(canvas)
            if type_index == '3D tomo':
                self.img_colormix_raw = np.array([])
                canvas.rgb_flag = 0
                canvas.x, canvas.y = [], []
                canvas.axes.clear()  # this is important, to clear the current image before another imshow()
                sh = self.img_rec_tomo.shape
                canvas.img_stack = self.img_rec_tomo
                canvas.special_info = None
                canvas.current_img_index = sh[0]//2
                canvas.title = [f'{i}:' for i in range(sh[0])]
                canvas.update_img_stack()
                slide.setMaximum(max(sh[0] - 1, 0))
                self.current_image = self.img_rec_tomo[canvas.current_img_index]
                self.auto_contrast(canvas)
            if type_index == '3D tomo denoised':
                self.img_colormix_raw = np.array([])
                canvas.rgb_flag = 0
                canvas.x, canvas.y = [], []
                canvas.axes.clear()  # this is important, to clear the current image before another imshow()
                sh = self.img_rec_tomo_denoise.shape
                canvas.img_stack = self.img_rec_tomo_denoise
                canvas.special_info = None
                canvas.current_img_index = sh[0]//2
                canvas.title = [f'{i}:' for i in range(sh[0])]
                canvas.update_img_stack()
                slide.setMaximum(max(sh[0] - 1, 0))
                self.current_image = self.img_rec_tomo_denoise[canvas.current_img_index]
                self.auto_contrast(canvas)
            if type_index == '3D tomo denoise compare':
                self.img_colormix_raw = np.array([])
                canvas.rgb_flag = 0
                canvas.x, canvas.y = [], []
                canvas.axes.clear()  # this is important, to clear the current image before another imshow()
                sh = self.img_rec_tomo_denoise_compare.shape
                canvas.img_stack = self.img_rec_tomo_denoise_compare
                canvas.special_info = None
                canvas.current_img_index = sh[0]//2
                canvas.title = [f'{i}:' for i in range(sh[0])]
                canvas.update_img_stack()
                slide.setMaximum(max(sh[0] - 1, 0))
                self.current_image = self.img_rec_tomo_denoise_compare[canvas.current_img_index]
                self.auto_contrast(canvas)
                try:
                    self.plot3D_slider = plot3D(self.img_rec_tomo_denoise_compare)
                except Exception as err:
                    print(err)
                    self.lb_msg.setText(str(err))


            if type_index == 'proj vs. proj_ml (raw)':
                self.img_colormix_raw = np.array([])
                canvas.rgb_flag = 0
                canvas.x, canvas.y = [], []
                canvas.axes.clear()  # this is important, to clear the current image before another imshow()
                sh = self.img_prj_norm.shape
                canvas.img_stack = self.img_prj_norm
                canvas.special_info = None
                canvas.current_img_index = self.sl1.value()
                canvas.title = [f'#{i:3d}' for i in range(len(self.img_prj_norm))]
                canvas.update_img_stack()
                slide.setMaximum(max(sh[0] - 1, 0))
                self.auto_contrast(canvas)
                self.current_image = self.img_prj_norm[0]
            if type_index == 'proj vs. proj_ml (ml)':
                self.img_colormix_raw = np.array([])
                canvas.rgb_flag = 0
                canvas.x, canvas.y = [], []
                canvas.axes.clear()  # this is important, to clear the current image before another imshow()
                sh = self.proj_ml.shape
                canvas.img_stack = self.proj_ml
                canvas.special_info = None
                canvas.current_img_index = self.sl1.value()
                canvas.title = [f'#{i:3d}' for i in range(len(self.proj_ml))]
                canvas.update_img_stack()
                slide.setMaximum(max(sh[0] - 1, 0))
                self.auto_contrast(canvas)
                self.current_image = self.proj_ml[0]
        except Exception as err:
            print(err)

class MyCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=5, dpi=110, obj=[]):
        self.obj = obj
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        self.axes.axis('off')
        self.cmax = 1
        self.cmin = 0
        self.rgb_flag = 0
        self.img_stack = np.zeros([1, 100, 100])
        self.current_img = self.img_stack[0]
        self.current_img_index = 0
        self.mask = np.array([1])
        self.rgb_mask = np.array([1])
        self.colorbar_on_flag = True
        self.colormap = 'viridis'
        self.title = []
        self.sup_title = ''
        self.draw_line = False
        self.overlay_flag = True
        self.x, self.y, = [], []
        self.plot_label = ''
        self.legend_flag = False
        self.roi_list = {}
        self.roi_color = {}
        self.roi_count = 0
        self.show_roi_flag = False
        self.current_roi = [0, 0, 0, 0, '0'] # x1, y1, x2, y1, roi_name
        self.color_list = ['red', 'green', 'blue', 'cyan', 'pink', 'yellow', 'orange', 'olive', 'purple', 'gray']
        self.current_color = 'red'
        self.special_info = None
        FigureCanvas.__init__(self, self.fig)
        #FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        self.setParent(parent)
        self.mpl_connect('motion_notify_event', self.mouse_moved)

    def mouse_moved(self, mouse_event):
        if mouse_event.inaxes:
            x, y = mouse_event.xdata, mouse_event.ydata
            self.obj.lb_x_l.setText('x: {:3.2f}'.format(x))
            self.obj.lb_y_l.setText('y: {:3.2f}'.format(y))
            row = int(np.max([np.min([self.current_img.shape[0], y]), 0]))
            col = int(np.max([np.min([self.current_img.shape[1], x]), 0]))
            try:
                z = self.current_img[row][col]
                self.obj.lb_z_l.setText('intensity: {:3.4f}'.format(z))
            except Exception as err:
                print(err)
                self.obj.lb_z_l.setText('')

    def update_img_stack(self):
        #self.axes = self.fig.add_subplot(111)
        n = len(self.img_stack)
        if self.rgb_flag:  # RGB image
            return self.update_img_one(self.img_stack)
        elif n == 0:
            img_blank = np.zeros([100, 100])
            return self.update_img_one(img_blank, img_index=0)
        else:
            if self.current_img_index >= len(self.img_stack):
                self.current_img_index = 0

            if type(self.img_stack) is np.ndarray:
                s = self.img_stack.shape
                if len(s) == 2:
                    self.img_stack = self.img_stack.reshape(1, s[0], s[1])

            return self.update_img_one(self.img_stack[self.current_img_index], img_index=self.current_img_index)

    def update_img_one(self, img=np.array([]), img_index=0):
        self.axes.clear()
        try:
            if self.rgb_flag:
                self.rgb_mask = self.mask
                if len(self.mask.shape) == 3:
                    self.rgb_mask = self.mask[0]
                for i in range(img.shape[2]):
                    img[:,:,i] *= self.rgb_mask
                self.im = self.axes.imshow(img)
                self.draw()
            else:
                if len(img) == []:
                    img = self.current_img
                self.current_img = img
                self.current_img_index = img_index
                self.im = self.axes.imshow(img*self.mask, cmap=self.colormap, vmin=self.cmin, vmax=self.cmax)
                self.axes.axis('on')
                self.axes.set_aspect('equal', 'box')
                if len(self.title) == len(self.img_stack):
                    self.axes.set_title(f'{self.sup_title}\n\ncurrent image: ' + self.title[img_index])
                else:
                    self.axes.set_title(f'{self.sup_title}\n\ncurrent image: ' + str(img_index))
                self.axes.title.set_fontsize(10)

                plt.tight_layout()
                if self.colorbar_on_flag:
                    self.add_colorbar()
                    self.colorbar_on_flag = False
                self.draw()
            if self.show_roi_flag:
                for i in range(len(self.roi_list)):
                    self.current_color = self.color_list[i % 10]
                    try:
                        self.roi_display(self.roi_list[f'roi_{i}'])
                    except:
                        pass
        except Exception as err:
            print(f'Error in updating image. Error: {str(err)}')

    def add_line(self):
        if self.draw_line:
            if self.overlay_flag:
                self.axes.plot(self.x, self.y, '-', color=self.current_color, linewidth=1.0, label=self.plot_label)
            else:
                self.rm_colorbar()
                line, = self.axes.plot(self.x, self.y, '.-', color=self.current_color, linewidth=1.0, label=self.plot_label)
                if self.legend_flag:
                    self.axes.legend(handles=[line])
                self.axes.axis('on')
                self.axes.set_aspect('auto')
                self.draw()

    def draw_roi(self):
        self.cidpress = self.mpl_connect('button_press_event', self.on_press)
        self.cidrelease = self.mpl_connect('button_release_event', self.on_release)
        self.show_roi_flag = True

    def on_press(self, event):
        x1, y1 = event.xdata, event.ydata
        self.current_roi[0] = x1
        self.current_roi[1] = y1

    def on_release(self, event):
        x2, y2 = event.xdata, event.ydata
        self.current_roi[2] = x2
        self.current_roi[3] = y2
        self.current_roi[4] = str(self.roi_count)
        self.roi_add_to_list()
        self.roi_display(self.current_roi)
        self.roi_disconnect()

    def roi_disconnect(self):
        self.mpl_disconnect(self.cidpress)
        self.mpl_disconnect(self.cidrelease)

    def roi_display(self, selected_roi):
        x1, y1 = selected_roi[0], selected_roi[1]
        x2, y2 = selected_roi[2], selected_roi[3]
        roi_index = selected_roi[4]
        self.x = [x1, x2, x2, x1, x1]
        self.y = [y1, y1, y2, y2, y1]
        self.draw_line = True
        self.add_line()
        self.draw_line = False
        roi_name = f'#{roi_index}'
        s = self.current_img.shape
        self.axes.annotate(roi_name, xy=(x1, y1 - s[0] // 40),
                           bbox={'facecolor': self.current_color, 'alpha': 0.5, 'pad': 2},
                           fontsize=10)
        self.draw()
        self.obj.pb_roi_draw.setEnabled(True)
        QApplication.processEvents()

    def roi_add_to_list(self, roi_name=''):
        if not len(roi_name):
            roi_name = 'roi_' + str(self.roi_count)
        self.roi_list[roi_name] = deepcopy(self.current_roi)
        self.current_color = self.color_list[self.roi_count % 10]
        self.roi_color[roi_name] = self.current_color
        self.roi_count += 1
        self.obj.update_roi_list(mode='add', item_name=roi_name)

    def set_contrast(self, cmin, cmax):
        self.cmax = cmax
        self.cmin = cmin
        self.colorbar_on_flag = True
        if self.rgb_flag:
            img = (self.img_stack - cmin) / (cmax - cmin)
            mask = deepcopy(self.mask)
            if len(mask.shape) == 2:
                mask = np.expand_dims(mask,axis=2)
                mask = np.repeat(mask, repeats=3, axis=2)
            self.update_img_one(img[:,:,:3] * mask)
        else:
            self.update_img_one(self.current_img*self.mask, self.current_img_index)

    def auto_contrast(self):
        try:
            img = self.current_img*self.mask
        except:
            print('image and mask has different shape, will not apply mask')
            img = self.current_img.copy()
        self.cmax = np.max(img)
        self.cmin = np.min(img)
        self.colorbar_on_flag = True
        self.update_img_one(self.current_img*self.mask, self.current_img_index)
        return self.cmin, self.cmax

    def rm_colorbar(self):
        try:
            self.cb.remove()
            self.draw()
        except Exception as err:
            print(err)

    def add_colorbar(self):
        if self.colorbar_on_flag:
            try:
                self.cb.remove()
                self.draw()
            except Exception as err:
                print(err)
            self.divider = make_axes_locatable(self.axes)
            self.cax = self.divider.append_axes('right', size='3%', pad=0.06)
            self.cb = self.fig.colorbar(self.im, cax=self.cax, orientation='vertical')
            self.cb.ax.tick_params(labelsize=10)
            self.draw()


def find_nearest(data, value):
    data = np.array(data)
    return np.abs(data - value).argmin()

def denoise(prj, denoise_flag):
    if denoise_flag == 2:  # Wiener denoise
        import skimage.restoration as skr
        ss = prj.shape
        psf = np.ones([2, 2]) / (2 ** 2)
        reg = None
        balance = 0.3
        is_real = True
        clip = True
        for j in range(ss[0]):
            prj[j] = skr.wiener(
                prj[j], psf=psf, reg=reg, balance=balance, is_real=is_real, clip=clip
            )
    elif denoise_flag == 1:  # Gaussian denoise
        from skimage.filters import gaussian as gf
        prj = gf(prj, [0, 1, 1])
    return prj

if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")
    global pytomo

    app = QApplication(sys.argv)
    pytomo = App()

    pytomo.show()
    sys.exit(app.exec_())
