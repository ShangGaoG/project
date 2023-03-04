# -*-coding:utf-8-*-
import torch
from ltr.data import processing, sampler, LTRLoader
from ltr.dataset import Lasot, MSCOCOSeq, Got10k, TrackingNet, DepthTrack, MSCOCOSeq_depth, Lasot_depth, CDTB, Got10k_depth


import ltr.models.tracking.early_fusion_noM as hcat_early_noM
import ltr.models.tracking.early_fusion_noM_bi as hcat_early_noM_bi
from ltr import actors
from ltr.trainers import LTRTrainer
import ltr.data.transforms as tfm
from ltr import MultiGPU
from ltr.trainers.hcat_trainer import HCATLTRTrainer
import os
os.environ["CUDA_VISIBLE_DEVICES"]= "0"

def run(settings):
    # Most common settings are assigned in the settings struct
    settings.device = 'cuda'
    settings.description = 'emat noM bi with more data. parameter finetune'
    settings.batch_size = 128#128
    settings.num_workers = 4
    settings.multi_gpu = False
    settings.print_interval = 1
    settings.normalize_mean = [0.485, 0.456, 0.406]
    settings.normalize_std = [0.229, 0.224, 0.225]
    settings.search_area_factor = 4.0
    settings.template_area_factor = 2.0
    settings.search_feature_sz = 32
    settings.template_feature_sz = 16
    settings.search_sz = settings.search_feature_sz * 8
    settings.temp_sz = settings.template_feature_sz * 8
    settings.center_jitter_factor = {'search': 3, 'template': 0}
    settings.scale_jitter_factor = {'search': 0.25, 'template': 0}

    settings.backbone = 'resnet18'

    # Transformer
    settings.position_embedding = 'sine'
    settings.hidden_dim = 256
    settings.dropout = 0.1
    settings.nheads = 8
    settings.dim_feedforward = 2048
    settings.featurefusion_layers = 2
    
    #Save 
    # settings.exname = 'rgb3d-add'
    settings.M = 0.1
    settings.epoch = -1
    # settings.exname = 'colormap-early-fusion-noM'
    # settings.exname = 'colormap-early-fusion-noM-bi'
    settings.exname = 'NoM-bi-param-moredata'
    input_dtype = 'rgbcolormap'
    
    # Train datasets
    #lasot_train = Lasot(settings.env.lasot_dir, split='train')
    got10k_train = Got10k_depth(root=settings.env.got10kdepth_dir, split='train', dtype=input_dtype)
    coco_train = MSCOCOSeq_depth(settings.env.cocodepth_dir, dtype=input_dtype)
    lasot_depth_train = Lasot_depth(root=settings.env.lasotdepth_dir, dtype=input_dtype)
    depthtrack_train = DepthTrack(root=settings.env.depthtrack_dir, split='train', dtype=input_dtype)
    depthtrack_val = DepthTrack(root=settings.env.depthtrack_dir, split='val', dtype=input_dtype)
    cdtb_val = CDTB(settings.env.cdtb_dir, split='val', dtype='rgbcolormap')
    # got10k_train = Got10k(settings.env.got10k_dir, split='all')
    #got10k_train = Got10k(settings.env.got10k_dir, split='vottrain')
    # trackingnet_train = TrackingNet(settings.env.trackingnet_dir, set_ids=list(range(12)))
    # coco_train = MSCOCOSeq(settings.env.coco_dir)

    # The joint augmentation transform, that is applied to the pairs jointly
    transform_joint = tfm.Transform(tfm.ToGrayscale(probability=0.05))

    # The augmentation transform applied to the training set (individually to each image in the pair)
    transform_train = tfm.Transform(tfm.ToTensorAndJitter(0.2),
                                    tfm.Normalize(mean=settings.normalize_mean, std=settings.normalize_std))
    transform_val = tfm.Transform(tfm.ToTensor(),
                                    tfm.Normalize(mean=settings.normalize_mean, std=settings.normalize_std))
    # Data processing to do on the training pairs
    data_processing_train = processing.TransTProcessing(search_area_factor=settings.search_area_factor,
                                                      template_area_factor = settings.template_area_factor,
                                                      search_sz=settings.search_sz,
                                                      temp_sz=settings.temp_sz,
                                                      center_jitter_factor=settings.center_jitter_factor,
                                                      scale_jitter_factor=settings.scale_jitter_factor,
                                                      mode='sequence',
                                                      transform=transform_train,
                                                      joint_transform=transform_joint)
    
    data_processing_val = processing.TransTProcessing(search_area_factor=settings.search_area_factor,
                                                      template_area_factor = settings.template_area_factor,
                                                      search_sz=settings.search_sz,
                                                      temp_sz=settings.temp_sz,
                                                      center_jitter_factor=settings.center_jitter_factor,
                                                      scale_jitter_factor=settings.scale_jitter_factor,
                                                      mode='sequence',
                                                      transform=transform_val,
                                                      joint_transform=transform_joint)

    # The sampler for training
    dataset_train = sampler.HCATSampler([got10k_train, lasot_depth_train, coco_train, depthtrack_train,], [1,1,1,1],
                                samples_per_epoch=90000, max_gap=30, processing=data_processing_train)
    
    dataset_val = sampler.HCATSampler([depthtrack_val,cdtb_val], [1,1],
                                samples_per_epoch=10000, max_gap=30, processing=data_processing_val)


    # dataset_train = sampler.HCATSampler([lasot_train], [1],
    #                             samples_per_epoch=1000*settings.batch_size, max_gap=100, processing=data_processing_train)

    # The loader for training
    loader_train = LTRLoader('train', dataset_train, training=True, batch_size=settings.batch_size, num_workers=settings.num_workers,
                             shuffle=True, drop_last=True, stack_dim=0)
    
    loader_val = LTRLoader('val', dataset_val, training=False, batch_size=settings.batch_size, num_workers=settings.num_workers,
                             shuffle=False, drop_last=True, epoch_interval=5,stack_dim=0)

    # Create network and actor
    #model = hcat_rgbd_add_models.hcat(settings)
    #model = hcat_models.hcat(settings)
    model = hcat_early_noM_bi.hcat(settings)

    # Wrap the network for multi GPU training
    if settings.multi_gpu:
        model = MultiGPU(model, dim=0)

    objective = hcat_early_noM_bi.hcat_loss(settings)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    actor = actors.HCATActor(net=model, objective=objective)

    # Optimizer
    param_dicts = [
        # {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": 0.000005, #0.00001
        },
        {
            "params": [p for n, p in model.named_parameters() if "fusionblock.rgbblock" in n and p.requires_grad],
            "lr": 1e-4,
        },

        {
            "params": [p for n, p in model.named_parameters() if "fusionblock.depthblock" in n and p.requires_grad],
            "lr": 1e-4,
        },
        {
            "params": [p for n, p in model.named_parameters() if "fusionblock.se" in n and p.requires_grad],
            "lr": 1e-4,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=1e-4,
                                  weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 25)

    # Create trainer
    trainer = LTRTrainer(actor, [loader_train, loader_val], optimizer, settings, lr_scheduler)

    # Run training (set fail_safe=False if you are debugging)
    trainer.train(300, load_latest=True, fail_safe=True)
