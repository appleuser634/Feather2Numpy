import pandas as pd
import numpy as np
import feather

import cv2

import sys
from tqdm import tqdm

#Dataframeを読み込みCVで扱いやすいようにリシェイプを行う関数
def load_dataframe(path):

    df = feather.read_dataframe(path)
    nd_array = df.values        
    
    print("\n読み込んだデーターセット",path)        
    print("データー母数:",len(nd_array))
    print("データー形状:",nd_array[0].shape)
    
    print("\n137x236のイメージに変換中...")    
    img_list = [np.reshape(n, (137,236)) for n in tqdm(nd_array)]

    print("変換後のデーター形状:",img_list[0].shape) 

    return img_list

#ぼかし処理を行う関数
def blur_image(img_list):

    blur_image_list = []

    for img in tqdm(img_list):
        
        blur = cv2.GaussianBlur(img,(5,5),0)
        blur_image_list.append(blur)
    
    return blur_image_list
    
#画像データーの水増しを行う関数
def augmentation(img_list):
   
    print("\n水平反転処理中...")
    horizontal_flip = [cv2.flip(img_list[n],0) for n in tqdm(range(5000))]
    print("水辺反転処枚数:",len(horizontal_flip))
    
    print("\n左右反転処理中...")
    vertical_flip = [cv2.flip(img_list[n],1) for n in tqdm(range(5000,10000))]
    print("左右反転処理枚数:",len(vertical_flip))
    
    print("\nぼかし処理中...")
    blur_image_list = blur_image(img_list)
    print("ぼかし処理枚数:",len(blur_image_list))
        
    return horizontal_flip,vertical_flip,blur_image_list

#各水増し処理のプレビューを行う関数
def preview(preview_list):

    for title,img in preview_list.items():

        cv2.imshow(title,img)
        cv2.waitKey(1)
    
    #プレビュー時にキーの待受を行う
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#各処理のフローをまとめる関数
def main(dataframe_path,save_status):
    
    img_list = load_dataframe(dataframe_path)
    
    print("\nイメージサイズを256x256に変換中...")    
    img_list = [cv2.resize(img,(256,256)) for img in tqdm(img_list)] 
    
    print("\nイメージをRGB変換中...")
    img_list = [cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) for img in tqdm(img_list)]
    
    #画像データーの水増し処理を実行
    horizontal_flip,vertical_flip,blur_image_list = augmentation(img_list)
    
    #水増し処理済みの画像のプレビューを行う
    preview_list = {"Normal":img_list[0],"horizontal_flip":horizontal_flip[0],"blur_image":blur_image_list[0]}
    preview(preview_list)
    
    #データーセットをまとめて保存
    if save_status == "n":
        
        img_list = np.concatenate([img_list,horizontal_flip,vertical_flip,blur_image_list]) 
        
        print("\nデーターセットをまとめて保存します")
        print("保存のため圧縮中...")
        
        np.savez_compressed('./augment_dataset.npz',img_list)
        print('保存しました -> ./augment_dataset.npz')
        print("データー枚数:",len(img_list))
    
    #データーセットを分けて保存
    elif save_status == "y":
        
        print("\nデーターセットを分けて保存します。")
        print("保存のため圧縮中...")
        
        np.savez_compressed('./resized_dataset.npz',img_list)
        print("\nリサイズ済みのデーターセットを保存しました -> ./resized_dataset.npz")
        print("データー枚数:",len(img_list))
    
        np.savez_compressed('./horizontal_flip_dataset.npz',horizontal_flip)
        print("\n水平反転済みのデーターセット保存しました　-> ./horizontal_flip_dataset.npz")
        print("データー枚数:",len(horizontal_flip))

        np.savez_compressed('./vertical_flip_dataset.npz',vertical_flip)
        print("\n左右反転済みのデーターセット保存しました　-> ./vertical_flip_dataset.npz")
        print("データー枚数:",len(vertical_flip))
    
        np.savez_compressed('./blur_dataset.npz',blur_image_list)
        print("\nぼかし処理済みのデーターセット保存しました　-> ./blur_dataset.npz")
        print("データー枚数:",len(blur_image_list))
    
    print("\nAugmentationを終了します")    

if __name__ == "__main__":

    #実行時の引数でDataframeのパスを指定できるようにした デフォルトはimage_dataset.feather
    if len(sys.argv) == 1:
        dataframe_path = "./image_dataset.feather" 
    else:
        dataframe_path = sys.argv[1]
    
    while True:   
        save_status = input("\nAugmentation手法ごとにデーターセットを分けますか？y/n ")
        
        if save_status == "y" or save_status == "n":
            break
        else:
            print("yかnを入力してください")
    
    main(dataframe_path,save_status)
