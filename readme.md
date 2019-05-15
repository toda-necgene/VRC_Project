# VRC(Voice Realtime Conversion)
## TL;DR
話者Aの声質を話者Bの声質にCycleGANを用いて変換するプログラムです。

1. 10分ぐらい録音して、16000Hz,16bit,モノラル,非圧縮で保存
2. 自分の声を"./dataset/train/A"に配置
3. 相手の声を"./dataset/train/B"に配置
```
$ python train.py
```
GTX1070なら1時間半ぐらいで終了
 
## インストール（環境構築）
以下のライブラリをインストールしてください。

### 変換時
- chainer
- numpy
- pyaudio
- pyworld

### 学習時
- chainer
- numpy
- pyworld
- matplotlib

## 学習の手順
"setting.json"ファイルで学習の設定が行なえます。
主に出力先の設定を記述します。
以下の手順で音声ファイルを用意します。

"dataset"ディレクトリを用意して、以下に"train","test","patch"を用意してください。

**16000Hz,int16,モノラル,非圧縮音声ファイル**(台本.txtの読み上げを推奨します)
を"dataset/train/A/"と"dataset/train/B/"に話者ごとに分けて入れてください。
なお、**同じ文章を読まなくても構いません**。ただし、データが偏った場合は正常に学習できない場合があります。

同条件で10秒程度のテスト用ファイル(話者A)を用意して、
"test.wav"を"dataset/test"ディレクトリ以下に保存してください。

以下のコマンドを実行してください。

```
$ python train.py
```
学習はGTX1070(OCなし)環境で1時間半ほどかかります。

ちなみにデータは量より質です。
大量に偏ったデータを読み込ませると学習に時間がかかるだけでなく性能が低下する場合もあります。

## 開発ポリシー
### 高速
CPUでのリアルタイム推論変換実行が可能にできること
### 安定
誰でも学習できるように安定的なネットワーク/パラメータを設定ポリシー
### 少データ
規定の文章であれば10分程度の録音で足りる

## ライセンス

MIT