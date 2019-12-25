"""
製作者:TODA
設定ファイル関連の関数
"""
import json
def load_setting_from_json(path):
    """
    モデルに関する設定ファイルを読み出します。

    Returns
    -------
    _args: dict
    各パラメータを保持しています。
    """
    _args = dict()
    # 保存する任意のディレクトリー名（特に話者データを変えて実験する際におすすめ）
    _args["model_name"] = "VRC"
    # バージョン管理用パラメータ（特に発話データを変えて実験する際におすすめ）
    # model_nameの後にディレクトリ名として足されます
    _args["version"] = "1.0.0"
    # 学習済みモデルの保管場所
    # このディレクトリの中に上記で設定したディレクトリが存在しその中に学習済みパラメータが保存されます
    _args["checkpoint_dir"] = "./trained_model/"
    # テスト音声とパワースペクトラム画像の保管場所
    # このディレクトリの中に上記で設定したディレクトリが存在しその中に音声と画像が保存されます
    _args["wave_otp_dir"] = "./waves/"
    # 生成済み学習ファイルの使用フラグ
    # ファイルはデフォルトで"dataset/patch"に存在します。
    _args["use_old_dataset"] = False
    _args["use_predata"] = "no"
    # テスト音声生成フラグ
    _args["test"] = True
    # ミニバッチで扱うデータの数
    _args["batch_size"] = 1
    # 学習でパラメータを更新する回数
    _args["train_iteration"] = 10000
    # 出力をする頻度(イテレーション単位)
    _args["log_interval"] = 100
    # 入力で見る受容体データ数(15999なら約1秒)
    _args["input_size"] = 15999
    # GPU使用デバイス設定(負の値ならCPUを使う)
    _args["gpu"] = -1
    # Lineの通知を使う
    # APIキーは”line_api_token.txt”に格納（文字列素置き）
    _args["line_notify"] = False
    # loading json setting file
    # (more codes ./setting.json. manual is exist in ./setting-example.json)
    with open(path, "r") as setting_raw_txt:
        try:
            json_loaded = json.load(setting_raw_txt)
            keys = json_loaded.keys()
            for k in keys:
                if k in _args:
                    if isinstance(_args[k], type(json_loaded[k])):
                        _args[k] = json_loaded[k]
                    else:
                        print(" [W] Argumet \"" + k + "\" is incorrect data type. Please change to \"" + str(type(_args[k])) + "\"")
                elif k[0] == "#":
                    pass
                else:
                    print(" [W] Argument \"" + k + "\" is not exsits.")
            # initializing harvest directory
            _args["name_save"] = _args["checkpoint_dir"]+_args["model_name"] + _args["version"]
        except json.JSONDecodeError as er_message:
            print(" [W] JSONDecodeError: ", er_message)
            print(" [W] Use default setting")
    return _args
