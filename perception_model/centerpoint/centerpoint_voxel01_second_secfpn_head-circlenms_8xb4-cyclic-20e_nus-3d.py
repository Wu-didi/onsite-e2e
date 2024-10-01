auto_scale_lr = dict(base_batch_size=32, enable=False)
backend_args = None
class_names = [
    'car',
    'truck',
    'construction_vehicle',
    'bus',
    'trailer',
    'barrier',
    'motorcycle',
    'bicycle',
    'pedestrian',
    'traffic_cone',
]
data_prefix = dict(img='', pts='samples/LIDAR_TOP', sweeps='sweeps/LIDAR_TOP')
data_root = 'data/nuscenes/'
dataset_type = 'NuScenesDataset'
db_sampler = dict(
    backend_args=None,
    classes=[
        'car',
        'truck',
        'construction_vehicle',
        'bus',
        'trailer',
        'barrier',
        'motorcycle',
        'bicycle',
        'pedestrian',
        'traffic_cone',
    ],
    data_root='data/nuscenes/',
    info_path='data/nuscenes/nuscenes_dbinfos_train.pkl',
    points_loader=dict(
        backend_args=None,
        coord_type='LIDAR',
        load_dim=5,
        type='LoadPointsFromFile',
        use_dim=[
            0,
            1,
            2,
            3,
            4,
        ]),
    prepare=dict(
        filter_by_difficulty=[
            -1,
        ],
        filter_by_min_points=dict(
            barrier=5,
            bicycle=5,
            bus=5,
            car=5,
            construction_vehicle=5,
            motorcycle=5,
            pedestrian=5,
            traffic_cone=5,
            trailer=5,
            truck=5)),
    rate=1.0,
    sample_groups=dict(
        barrier=2,
        bicycle=6,
        bus=4,
        car=2,
        construction_vehicle=7,
        motorcycle=6,
        pedestrian=2,
        traffic_cone=2,
        trailer=6,
        truck=3))
default_hooks = dict(
    checkpoint=dict(interval=-1, type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='Det3DVisualizationHook'))
default_scope = 'mmdet3d'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
eval_pipeline = [
    dict(
        backend_args=None,
        coord_type='LIDAR',
        load_dim=5,
        type='LoadPointsFromFile',
        use_dim=5),
    dict(
        backend_args=None,
        sweeps_num=10,
        test_mode=True,
        type='LoadPointsFromMultiSweeps'),
    dict(keys=[
        'points',
    ], type='Pack3DDetInputs'),
]
input_modality = dict(use_camera=False, use_lidar=True)
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
lr = 0.0001
metainfo = dict(classes=[
    'car',
    'truck',
    'trailer',
    'bus',
    'construction_vehicle',
    'bicycle',
    'motorcycle',
    'pedestrian',
    'traffic_cone',
    'barrier',
])
model = dict(
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_layer=dict(
            max_num_points=10,
            max_voxels=(
                90000,
                120000,
            ),
            point_cloud_range=[
                -51.2,
                -51.2,
                -5.0,
                51.2,
                51.2,
                3.0,
            ],
            voxel_size=[
                0.1,
                0.1,
                0.2,
            ])),
    pts_backbone=dict(
        conv_cfg=dict(bias=False, type='Conv2d'),
        in_channels=256,
        layer_nums=[
            5,
            5,
        ],
        layer_strides=[
            1,
            2,
        ],
        norm_cfg=dict(eps=0.001, momentum=0.01, type='BN'),
        out_channels=[
            128,
            256,
        ],
        type='SECOND'),
    pts_bbox_head=dict(
        bbox_coder=dict(
            code_size=9,
            max_num=500,
            out_size_factor=8,
            pc_range=[
                -51.2,
                -51.2,
            ],
            post_center_range=[
                -61.2,
                -61.2,
                -10.0,
                61.2,
                61.2,
                10.0,
            ],
            score_threshold=0.1,
            type='CenterPointBBoxCoder',
            voxel_size=[
                0.1,
                0.1,
            ]),
        common_heads=dict(
            dim=(
                3,
                2,
            ),
            height=(
                1,
                2,
            ),
            reg=(
                2,
                2,
            ),
            rot=(
                2,
                2,
            ),
            vel=(
                2,
                2,
            )),
        in_channels=512,
        loss_bbox=dict(
            loss_weight=0.25, reduction='mean', type='mmdet.L1Loss'),
        loss_cls=dict(reduction='mean', type='mmdet.GaussianFocalLoss'),
        norm_bbox=True,
        separate_head=dict(
            final_kernel=3, init_bias=-2.19, type='SeparateHead'),
        share_conv_channel=64,
        tasks=[
            dict(class_names=[
                'car',
            ], num_class=1),
            dict(class_names=[
                'truck',
                'construction_vehicle',
            ], num_class=2),
            dict(class_names=[
                'bus',
                'trailer',
            ], num_class=2),
            dict(class_names=[
                'barrier',
            ], num_class=1),
            dict(class_names=[
                'motorcycle',
                'bicycle',
            ], num_class=2),
            dict(class_names=[
                'pedestrian',
                'traffic_cone',
            ], num_class=2),
        ],
        type='CenterHead'),
    pts_middle_encoder=dict(
        block_type='basicblock',
        encoder_channels=(
            (
                16,
                16,
                32,
            ),
            (
                32,
                32,
                64,
            ),
            (
                64,
                64,
                128,
            ),
            (
                128,
                128,
            ),
        ),
        encoder_paddings=(
            (
                0,
                0,
                1,
            ),
            (
                0,
                0,
                1,
            ),
            (
                0,
                0,
                [
                    0,
                    1,
                    1,
                ],
            ),
            (
                0,
                0,
            ),
        ),
        in_channels=5,
        order=(
            'conv',
            'norm',
            'act',
        ),
        output_channels=128,
        sparse_shape=[
            41,
            1024,
            1024,
        ],
        type='SparseEncoder'),
    pts_neck=dict(
        in_channels=[
            128,
            256,
        ],
        norm_cfg=dict(eps=0.001, momentum=0.01, type='BN'),
        out_channels=[
            256,
            256,
        ],
        type='SECONDFPN',
        upsample_cfg=dict(bias=False, type='deconv'),
        upsample_strides=[
            1,
            2,
        ],
        use_conv_for_no_stride=True),
    pts_voxel_encoder=dict(num_features=5, type='HardSimpleVFE'),
    test_cfg=dict(
        pts=dict(
            max_per_img=500,
            max_pool_nms=False,
            min_radius=[
                4,
                12,
                10,
                1,
                0.85,
                0.175,
            ],
            nms_thr=0.2,
            nms_type='circle',
            out_size_factor=8,
            pc_range=[
                -51.2,
                -51.2,
            ],
            post_center_limit_range=[
                -61.2,
                -61.2,
                -10.0,
                61.2,
                61.2,
                10.0,
            ],
            post_max_size=83,
            pre_max_size=1000,
            score_threshold=0.1,
            voxel_size=[
                0.1,
                0.1,
            ])),
    train_cfg=dict(
        pts=dict(
            code_weights=[
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                0.2,
                0.2,
            ],
            dense_reg=1,
            gaussian_overlap=0.1,
            grid_size=[
                1024,
                1024,
                40,
            ],
            max_objs=500,
            min_radius=2,
            out_size_factor=8,
            point_cloud_range=[
                -51.2,
                -51.2,
                -5.0,
                51.2,
                51.2,
                3.0,
            ],
            voxel_size=[
                0.1,
                0.1,
                0.2,
            ])),
    type='CenterPoint')
optim_wrapper = dict(
    clip_grad=dict(max_norm=35, norm_type=2),
    optimizer=dict(lr=0.0001, type='AdamW', weight_decay=0.01),
    type='OptimWrapper')
param_scheduler = [
    dict(
        T_max=8,
        begin=0,
        by_epoch=True,
        convert_to_iter_based=True,
        end=8,
        eta_min=0.001,
        type='CosineAnnealingLR'),
    dict(
        T_max=12,
        begin=8,
        by_epoch=True,
        convert_to_iter_based=True,
        end=20,
        eta_min=1e-08,
        type='CosineAnnealingLR'),
    dict(
        T_max=8,
        begin=0,
        by_epoch=True,
        convert_to_iter_based=True,
        end=8,
        eta_min=0.8947368421052632,
        type='CosineAnnealingMomentum'),
    dict(
        T_max=12,
        begin=8,
        by_epoch=True,
        convert_to_iter_based=True,
        end=20,
        eta_min=1,
        type='CosineAnnealingMomentum'),
]
point_cloud_range = [
    -51.2,
    -51.2,
    -5.0,
    51.2,
    51.2,
    3.0,
]
resume = False
test_cfg = dict()
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='nuscenes_infos_val.pkl',
        backend_args=None,
        box_type_3d='LiDAR',
        data_prefix=dict(
            img='', pts='samples/LIDAR_TOP', sweeps='sweeps/LIDAR_TOP'),
        data_root='data/nuscenes/',
        metainfo=dict(classes=[
            'car',
            'truck',
            'construction_vehicle',
            'bus',
            'trailer',
            'barrier',
            'motorcycle',
            'bicycle',
            'pedestrian',
            'traffic_cone',
        ]),
        modality=dict(use_camera=False, use_lidar=True),
        pipeline=[
            dict(
                backend_args=None,
                coord_type='LIDAR',
                load_dim=5,
                type='LoadPointsFromFile',
                use_dim=5),
            dict(
                backend_args=None,
                pad_empty_sweeps=True,
                remove_close=True,
                sweeps_num=9,
                type='LoadPointsFromMultiSweeps',
                use_dim=[
                    0,
                    1,
                    2,
                    3,
                    4,
                ]),
            dict(
                flip=False,
                img_scale=(
                    1333,
                    800,
                ),
                pts_scale_ratio=1,
                transforms=[
                    dict(
                        rot_range=[
                            0,
                            0,
                        ],
                        scale_ratio_range=[
                            1.0,
                            1.0,
                        ],
                        translation_std=[
                            0,
                            0,
                            0,
                        ],
                        type='GlobalRotScaleTrans'),
                    dict(type='RandomFlip3D'),
                    dict(
                        point_cloud_range=[
                            -51.2,
                            -51.2,
                            -5.0,
                            51.2,
                            51.2,
                            3.0,
                        ],
                        type='PointsRangeFilter'),
                ],
                type='MultiScaleFlipAug3D'),
            dict(keys=[
                'points',
            ], type='Pack3DDetInputs'),
        ],
        test_mode=True,
        type='NuScenesDataset'),
    drop_last=False,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file='data/nuscenes/nuscenes_infos_val.pkl',
    backend_args=None,
    data_root='data/nuscenes/',
    metric='bbox',
    type='NuScenesMetric')
test_pipeline = [
    dict(
        backend_args=None,
        coord_type='LIDAR',
        load_dim=5,
        type='LoadPointsFromFile',
        use_dim=5),
    dict(
        backend_args=None,
        pad_empty_sweeps=True,
        remove_close=True,
        sweeps_num=9,
        type='LoadPointsFromMultiSweeps',
        use_dim=[
            0,
            1,
            2,
            3,
            4,
        ]),
    dict(
        flip=False,
        img_scale=(
            1333,
            800,
        ),
        pts_scale_ratio=1,
        transforms=[
            dict(
                rot_range=[
                    0,
                    0,
                ],
                scale_ratio_range=[
                    1.0,
                    1.0,
                ],
                translation_std=[
                    0,
                    0,
                    0,
                ],
                type='GlobalRotScaleTrans'),
            dict(type='RandomFlip3D'),
            dict(
                point_cloud_range=[
                    -51.2,
                    -51.2,
                    -5.0,
                    51.2,
                    51.2,
                    3.0,
                ],
                type='PointsRangeFilter'),
        ],
        type='MultiScaleFlipAug3D'),
    dict(keys=[
        'points',
    ], type='Pack3DDetInputs'),
]
train_cfg = dict(by_epoch=True, max_epochs=20, val_interval=20)
train_dataloader = dict(
    batch_size=4,
    dataset=dict(
        dataset=dict(
            ann_file='nuscenes_infos_train.pkl',
            backend_args=None,
            box_type_3d='LiDAR',
            data_prefix=dict(
                img='', pts='samples/LIDAR_TOP', sweeps='sweeps/LIDAR_TOP'),
            data_root='data/nuscenes/',
            metainfo=dict(classes=[
                'car',
                'truck',
                'construction_vehicle',
                'bus',
                'trailer',
                'barrier',
                'motorcycle',
                'bicycle',
                'pedestrian',
                'traffic_cone',
            ]),
            pipeline=[
                dict(
                    backend_args=None,
                    coord_type='LIDAR',
                    load_dim=5,
                    type='LoadPointsFromFile',
                    use_dim=5),
                dict(
                    backend_args=None,
                    pad_empty_sweeps=True,
                    remove_close=True,
                    sweeps_num=9,
                    type='LoadPointsFromMultiSweeps',
                    use_dim=[
                        0,
                        1,
                        2,
                        3,
                        4,
                    ]),
                dict(
                    type='LoadAnnotations3D',
                    with_bbox_3d=True,
                    with_label_3d=True),
                dict(
                    db_sampler=dict(
                        backend_args=None,
                        classes=[
                            'car',
                            'truck',
                            'construction_vehicle',
                            'bus',
                            'trailer',
                            'barrier',
                            'motorcycle',
                            'bicycle',
                            'pedestrian',
                            'traffic_cone',
                        ],
                        data_root='data/nuscenes/',
                        info_path='data/nuscenes/nuscenes_dbinfos_train.pkl',
                        points_loader=dict(
                            backend_args=None,
                            coord_type='LIDAR',
                            load_dim=5,
                            type='LoadPointsFromFile',
                            use_dim=[
                                0,
                                1,
                                2,
                                3,
                                4,
                            ]),
                        prepare=dict(
                            filter_by_difficulty=[
                                -1,
                            ],
                            filter_by_min_points=dict(
                                barrier=5,
                                bicycle=5,
                                bus=5,
                                car=5,
                                construction_vehicle=5,
                                motorcycle=5,
                                pedestrian=5,
                                traffic_cone=5,
                                trailer=5,
                                truck=5)),
                        rate=1.0,
                        sample_groups=dict(
                            barrier=2,
                            bicycle=6,
                            bus=4,
                            car=2,
                            construction_vehicle=7,
                            motorcycle=6,
                            pedestrian=2,
                            traffic_cone=2,
                            trailer=6,
                            truck=3)),
                    type='ObjectSample'),
                dict(
                    rot_range=[
                        -0.3925,
                        0.3925,
                    ],
                    scale_ratio_range=[
                        0.95,
                        1.05,
                    ],
                    translation_std=[
                        0,
                        0,
                        0,
                    ],
                    type='GlobalRotScaleTrans'),
                dict(
                    flip_ratio_bev_horizontal=0.5,
                    flip_ratio_bev_vertical=0.5,
                    sync_2d=False,
                    type='RandomFlip3D'),
                dict(
                    point_cloud_range=[
                        -51.2,
                        -51.2,
                        -5.0,
                        51.2,
                        51.2,
                        3.0,
                    ],
                    type='PointsRangeFilter'),
                dict(
                    point_cloud_range=[
                        -51.2,
                        -51.2,
                        -5.0,
                        51.2,
                        51.2,
                        3.0,
                    ],
                    type='ObjectRangeFilter'),
                dict(
                    classes=[
                        'car',
                        'truck',
                        'construction_vehicle',
                        'bus',
                        'trailer',
                        'barrier',
                        'motorcycle',
                        'bicycle',
                        'pedestrian',
                        'traffic_cone',
                    ],
                    type='ObjectNameFilter'),
                dict(type='PointShuffle'),
                dict(
                    keys=[
                        'points',
                        'gt_bboxes_3d',
                        'gt_labels_3d',
                    ],
                    type='Pack3DDetInputs'),
            ],
            test_mode=False,
            type='NuScenesDataset',
            use_valid_flag=True),
        type='CBGSDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(
        backend_args=None,
        coord_type='LIDAR',
        load_dim=5,
        type='LoadPointsFromFile',
        use_dim=5),
    dict(
        backend_args=None,
        pad_empty_sweeps=True,
        remove_close=True,
        sweeps_num=9,
        type='LoadPointsFromMultiSweeps',
        use_dim=[
            0,
            1,
            2,
            3,
            4,
        ]),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(
        db_sampler=dict(
            backend_args=None,
            classes=[
                'car',
                'truck',
                'construction_vehicle',
                'bus',
                'trailer',
                'barrier',
                'motorcycle',
                'bicycle',
                'pedestrian',
                'traffic_cone',
            ],
            data_root='data/nuscenes/',
            info_path='data/nuscenes/nuscenes_dbinfos_train.pkl',
            points_loader=dict(
                backend_args=None,
                coord_type='LIDAR',
                load_dim=5,
                type='LoadPointsFromFile',
                use_dim=[
                    0,
                    1,
                    2,
                    3,
                    4,
                ]),
            prepare=dict(
                filter_by_difficulty=[
                    -1,
                ],
                filter_by_min_points=dict(
                    barrier=5,
                    bicycle=5,
                    bus=5,
                    car=5,
                    construction_vehicle=5,
                    motorcycle=5,
                    pedestrian=5,
                    traffic_cone=5,
                    trailer=5,
                    truck=5)),
            rate=1.0,
            sample_groups=dict(
                barrier=2,
                bicycle=6,
                bus=4,
                car=2,
                construction_vehicle=7,
                motorcycle=6,
                pedestrian=2,
                traffic_cone=2,
                trailer=6,
                truck=3)),
        type='ObjectSample'),
    dict(
        rot_range=[
            -0.3925,
            0.3925,
        ],
        scale_ratio_range=[
            0.95,
            1.05,
        ],
        translation_std=[
            0,
            0,
            0,
        ],
        type='GlobalRotScaleTrans'),
    dict(
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5,
        sync_2d=False,
        type='RandomFlip3D'),
    dict(
        point_cloud_range=[
            -51.2,
            -51.2,
            -5.0,
            51.2,
            51.2,
            3.0,
        ],
        type='PointsRangeFilter'),
    dict(
        point_cloud_range=[
            -51.2,
            -51.2,
            -5.0,
            51.2,
            51.2,
            3.0,
        ],
        type='ObjectRangeFilter'),
    dict(
        classes=[
            'car',
            'truck',
            'construction_vehicle',
            'bus',
            'trailer',
            'barrier',
            'motorcycle',
            'bicycle',
            'pedestrian',
            'traffic_cone',
        ],
        type='ObjectNameFilter'),
    dict(type='PointShuffle'),
    dict(
        keys=[
            'points',
            'gt_bboxes_3d',
            'gt_labels_3d',
        ],
        type='Pack3DDetInputs'),
]
val_cfg = dict()
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='nuscenes_infos_val.pkl',
        backend_args=None,
        box_type_3d='LiDAR',
        data_prefix=dict(
            img='', pts='samples/LIDAR_TOP', sweeps='sweeps/LIDAR_TOP'),
        data_root='data/nuscenes/',
        metainfo=dict(classes=[
            'car',
            'truck',
            'construction_vehicle',
            'bus',
            'trailer',
            'barrier',
            'motorcycle',
            'bicycle',
            'pedestrian',
            'traffic_cone',
        ]),
        modality=dict(use_camera=False, use_lidar=True),
        pipeline=[
            dict(
                backend_args=None,
                coord_type='LIDAR',
                load_dim=5,
                type='LoadPointsFromFile',
                use_dim=5),
            dict(
                backend_args=None,
                pad_empty_sweeps=True,
                remove_close=True,
                sweeps_num=9,
                type='LoadPointsFromMultiSweeps',
                use_dim=[
                    0,
                    1,
                    2,
                    3,
                    4,
                ]),
            dict(
                flip=False,
                img_scale=(
                    1333,
                    800,
                ),
                pts_scale_ratio=1,
                transforms=[
                    dict(
                        rot_range=[
                            0,
                            0,
                        ],
                        scale_ratio_range=[
                            1.0,
                            1.0,
                        ],
                        translation_std=[
                            0,
                            0,
                            0,
                        ],
                        type='GlobalRotScaleTrans'),
                    dict(type='RandomFlip3D'),
                    dict(
                        point_cloud_range=[
                            -51.2,
                            -51.2,
                            -5.0,
                            51.2,
                            51.2,
                            3.0,
                        ],
                        type='PointsRangeFilter'),
                ],
                type='MultiScaleFlipAug3D'),
            dict(keys=[
                'points',
            ], type='Pack3DDetInputs'),
        ],
        test_mode=True,
        type='NuScenesDataset'),
    drop_last=False,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    ann_file='data/nuscenes/nuscenes_infos_val.pkl',
    backend_args=None,
    data_root='data/nuscenes/',
    metric='bbox',
    type='NuScenesMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='Det3DLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
voxel_size = [
    0.1,
    0.1,
    0.2,
]
