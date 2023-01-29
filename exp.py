import torch ,os
import torchvision
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import pdb
import torchvision.transforms.functional as F
import pandas as pd

def read_img(path):
    img=torchvision.io.read_image(path)
    return img/255.0

def get_label(path,df):
        name = path.split('/')[-1]
        label = df[df.Image==name]['count'].values[0]
        return label

def plot_imgs(imgs_path_list,df,r=8,c=8):
    _,axs = plt.subplots(r,c)
    axs=axs.flatten()
    # pdb.set_trace()
    for n,ax in enumerate(axs):
        # pdb.set_trace()
        img= read_img(imgs_path_list[n])
        label = get_label(imgs_path_list[n],df)
        pdb.set_trace()
        ax.imshow(F.to_pil_image(img))
        ax.axis('off')
        ax.set_title(label)

    
    plt.tight_layout()
    plt.show()

def conver_csv(path):
    df = pd.read_csv(path)
    print(df)

def main():
    base_img_folder = '/Users/sagar/Desktop/Ace/Brovo5/counting/train/images'
    paths = [os.path.join(base_img_folder,p) for p in os.listdir(base_img_folder)]
    path_to_txt = '/Users/sagar/Desktop/Ace/Brovo5/counting/train/train_ground_truth.txt'
    df=  pd.read_csv(path_to_txt)
    plot_imgs(paths,df)
  


if __name__=="__main__":
    main()