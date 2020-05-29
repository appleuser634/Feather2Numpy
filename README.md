

----------------

実行時のコマンドライン引数でデータセットのパスを指定できます。(指定なしではデフォルトでimage_dataset.featherが読み込まれます)

`python3 data_augmentation.py データセットへのパス`

また、実行時に標準入力でデーターセットの出力をAugmentation方法ごとに分けるか、まとめるかを選択できます。

Resize,AugmentationにはOpenCVを使用しています。

Augmentationの内容
- 水平反転処理
- 左右反転処理
- ぼかし処理

Augmentationを上記の内容にした理由として、現実に起こりうる状態を想定した方が良いと考えました。

各処理のfor文にリスト内包表記を使用したのは、速度向上のためです。

featherファイルには、サイズ137x236の画像の各ピクセル値がDataFrame形式で1万枚分格納されています。

1. DataFrameに格納されている各ピクセル値をndarray型の画像データに変換する
　※配布したDataFrameに格納されている画像データは、元の画像を1次元の配列に変換し、画像1枚を1行に格納しています。(137×236  = 32332のカラムが存在します。)
2. 画像を256 * 256にリサイズする
3. RGBカラー画像に変換する
4. Data Augmentationにより、データを3万枚に水増しする
5. ndarrayから.npz形式で保存する, またはDataFrameに変換してfeather形式で保存する
　　保存するファイルは、1つにまとめても、複数に分けても構いません