# RVS(Realtime Vocoder System)
## 概要
　このプログラムは話者Aの声質を話者Bの声質に変換するプログラムです。
変換にCycleGANの技術を用いています。また、学習以外はCPUで動くように設計されています。
 
## 使い方
python3 VRC \[command\] \[option\]
 
\[command\]
 
--create_dataset データセットの作成を行います。（要GPU）
 
--train 学習を行います。（要GPU）先にデータセットの作成を行ってください。
 
--run   リアルタイム実行します。先に学習を済ませておいてください。
 
--test  学習済みモデルを用いてテストを行います。

\[options\]
 
 --path 設定ファイルの場所を指定します。（初期値は\"./setting.json\"）
  
## 依存ライブラリ
- tensorflow(1.9.0)
- numpy(1.14.2)
- cupy(4.0.0)
- pyaudio(0.2.11)
- matplotlib(2.1.2)
- hyperdash(0.15.2)
##ライセンス
MIT
