class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/ssd-nvme1/gs/HCAT'    # Base directory for saving network checkpoints.
        # self.tensorboard_dir = self.workspace_dir + '/tensorboard/rgb3d-cat-sa-train-val'    # Directory for tensorboard files.
        # self.tensorboard_dir = self.workspace_dir + '/tensorboard/early-fusion-noM-bi'
        # self.tensorboard_dir = self.workspace_dir + '/tensorboard/one-layer-noM-bi-moredata'
        # self.tensorboard_dir = self.workspace_dir + '/tensorboard/four-layer-noM-bi-moredata'
        self.tensorboard_dir = self.workspace_dir + '/tensorboard/resnet50-noM-bi-moredata'
        # self.tensorboard_dir = self.workspace_dir + '/test-zhan'
        self.lasot_dir = '/ssd2/gaoshang/LaSOT'
        self.got10k_dir = ''
        self.trackingnet_dir = ''
        self.coco_dir = ''
        self.lvis_dir = ''
        self.sbd_dir = ''
        self.imagenet_dir = ''
        self.imagenetdet_dir = ''
        self.ecssd_dir = ''
        self.hkuis_dir = ''
        self.msra10k_dir = ''
        self.davis_dir = ''
        self.youtubevos_dir = ''
        self.depthtrack_dir = '/ssd-sata1/gs/det-train-test/'
        self.cocodepth_dir = '/ssd-sata1/gs/COCO_densedepth/'
        self.got10kdepth_dir = '/ssd-sata1/gs/GOT-10k-depth/full_data/train_data'
        self.lasotdepth_dir = '/ssd-sata1/gs/LaSOT/dataset/'
        self.cdtb_dir = '/ssd-sata1/gs/CDTB'