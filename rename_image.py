import os  
import shutil  
  
def rename_and_move_images(source_folder, target_folder, prefix='image_'):  
    # 确保源文件夹存在  
    if not os.path.isdir(source_folder):  
        print(f"Error: The source folder {source_folder} does not exist.")  
        return  
      
    # 确保目标文件夹存在，如果不存在则创建  
    if not os.path.exists(target_folder):  
        os.makedirs(target_folder)  
      
    # 获取源文件夹中所有图片文件的列表  
    image_files = [f for f in os.listdir(source_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]  
      
    # 遍历图片文件，复制到新文件夹并重命名  
    for idx, image_file in enumerate(image_files):  
        # 构建新的文件名  
        idx = idx + 341
        new_filename = f"{prefix}{idx:04d}{os.path.splitext(image_file)[1]}"  
        # 构建源文件和目标文件的完整路径  
        source_path = os.path.join(source_folder, image_file)  
        target_path = os.path.join(target_folder, new_filename)  
        # 复制并重命名文件  
        shutil.copy2(source_path, target_path)  # copy2 保留元数据  
        print(f"Copied and renamed {source_path} to {target_path}")  
  
# 使用函数，例如将'/path/to/your/images'文件夹中的图片重命名并保存到'/path/to/new/folder'  
rename_and_move_images('/home/wudi/python_files/onsite/0519update/e2e/results', '/home/wudi/python_files/onsite/0519update/e2e/total_results')