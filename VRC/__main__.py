import sys
if __name__ == '__main__':
    pass
if len(sys.argv)!=0 and sys.argv.__contains__("--train"):
    import train
elif len(sys.argv)!=0 and sys.argv.__contains__("--test"):
    from utils import model_test
elif len(sys.argv)!=0 and sys.argv.__contains__("--run"):
     import run
elif len(sys.argv)!=0 and sys.argv.__contains__("--create_dataset"):
    from utils import voice_to_datasets_cycle
else :
    print("コマンドがありません。")
    print("--コマンド説明--")
    print("python3 ./VRC [command]")
    print("[command]")
    print("--train          学習を行います。")
    print("--test           学習済みモデルの実行テストを行います。")
    print("--run            リアルタイム変換を行います。")
    print("--create_dataset データセットを音声から生成します。")
    print("各設定は\"setting.json\"ファイルを使用します。\nなお、詳しい使い方はリードミーをご覧ください。")