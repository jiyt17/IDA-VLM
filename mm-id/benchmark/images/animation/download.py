import subprocess


def download_with_wget(url, save_path):
    try:
        # 使用subprocess调用wget命令下载图片
        subprocess.run(["wget", "-O", save_path, url], check=True)
        print("图片下载成功:", save_path)
    except subprocess.CalledProcessError as e:
        print("下载出错:", e)

animation = 'zhouhui'
src_txt = animation + '/link.txt'
link_list = []
with open(src_txt) as f:
    link_list = f.readlines()

link_list = [link.strip() for link in link_list]
name = ''
id = 1
for link in link_list:
    if link.endswith(':'):
        name = link[:-1].capitalize()
        id = 1
        continue
    download_with_wget(link, animation+'/'+name+f'_{id}.jpg')
    id += 1

