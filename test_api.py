# from mmdet.apis import init_detector, inference_detector
# import mmcv

# # 指定模型的配置文件和 checkpoint 文件路径
# config_file = 'yolox_tiny_8x8_300e_coco.py'
# checkpoint_file = 'yolox_tiny_8x8_300e_coco_20211124_171234-b4047906.pth'

# # 根据配置文件和 checkpoint 文件构建模型
# model = init_detector(config_file, checkpoint_file, device='cuda:0')

# # 测试单张图片并展示结果
# img = '/home/wudi/python_files/onsite/v2/e2e/results/cam_0_167.jpg'  # 或者 img = mmcv.imread(img)，这样图片仅会被读一次
# result = inference_detector(model, img)
# print(result)
# # # 在一个新的窗口中将结果可视化
# # model.show_result(img, result)
# # # 或者将可视化结果保存为图片
# # model.show_result(img, result, out_file='result.jpg')



from mmdet.apis import DetInferencer

# Initialize the DetInferencer
inferencer = DetInferencer('rtmdet_tiny_8xb32-300e_coco')

# Perform inference
result = inferencer('/home/wudi/python_files/onsite/v2/e2e/results/cam_0_167.jpg',pred_score_thr=0.5, print_result=True)
print(result)


# from mmdet.apis import init_detector, inference_detector
# import mmcv

# # 指定模型的配置文件和 checkpoint 文件路径
# config_file = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
# checkpoint_file = 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

# # 根据配置文件和 checkpoint 文件构建模型
# model = init_detector(config_file, checkpoint_file, device='cuda:0')

# # 测试单张图片并展示结果
# img = 'test.jpg'  # 或者 img = mmcv.imread(img)，这样图片仅会被读一次
# result = inference_detector(model, img)
# # 在一个新的窗口中将结果可视化
# model.show_result(img, result)
# # 或者将可视化结果保存为图片
# model.show_result(img, result, out_file='result.jpg')

# # 测试视频并展示结果
# video = mmcv.VideoReader('video.mp4')
# for frame in video:
#     result = inference_detector(model, frame)
#     model.show_result(frame, result, wait_time=1)