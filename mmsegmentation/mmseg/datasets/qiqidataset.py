import os.path as osp
import tempfile

import mmcv
import numpy as np
from mmcv.utils import print_log
from PIL import Image
from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset

classes = ('background', 'drivable_areas', 'alternatively_drivable_areas')
palette = [[0, 0, 0], [255, 0, 0], [0, 0, 255]]
@DATASETS.register_module()
class QiqiDataset(CustomDataset):
  CLASSES = classes
  PALETTE = palette
  def __init__(self, split, **kwargs):
    super().__init__(img_suffix='.jpg', seg_map_suffix='.png', split=split, **kwargs)
    assert osp.exists(self.img_dir) and self.split is not None