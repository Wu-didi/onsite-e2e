import os  
  
# 设置图片文件夹的路径  
images_folder = 'total_results'  
  
# 创建一个空列表来存储所有.jpg图片的相对路径  
image_paths = []  
  
# 遍历图片文件夹中的所有文件  
for root, dirs, files in os.walk(images_folder):  
    for file in files:  
        # 检查文件是否是一个.jpg图片  
        if file.lower().endswith('.jpg'):  
            image_path = os.path.join(root, file)  
            relative_path = os.path.relpath(image_path, start=os.getcwd())  # 获取相对路径  
            image_paths.append(relative_path)  
  
# 打开（或创建）train.txt文件，并写入.jpg图片的相对路径  
with open('train.txt', 'w') as f:  
    for path in image_paths:  
        f.write(path + '\n')  # 每一个路径占一行