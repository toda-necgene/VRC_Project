"""
製作者:TODA
設定ファイル関連の関数
"""
import json
def load_setting_from_json(path_setting):
    """
    モデルに関する設定ファイルを読み出します。

    Returns
    -------
    _args: dict
    各パラメータを保持しています。
    """
    _args = dict()
    # name options
    _args["model_name"] = "VRC"
    _args["version"] = "1.0.0"
    # saving options
    _args["checkpoint_dir"] = "./trained_models"
    _args["wave_otp_dir"] = "./harvests"
    #training-data options
    _args["use_old_dataset"] = False
    _args["train_data_dir"] = "./dataset/train"
    _args["test_data_dir"] = "./dataset/test"
    # learning details output options
    _args["test"] = True
    _args["real_sample_compare"] = False
    # learning options
    _args["batch_size"] = 128
    _args["train_iteration"] = 10000
    _args["log_interval"] = 5
    # architecture option
    _args["input_size"] = 4096
    _args["gpu"] = -1
    # loading json setting file
    # (more codes ./setting.json. manual is exist in ./setting-example.json)
    with open(path_setting, "r") as setting_raw_txt:
        try:
            json_loaded = json.load(setting_raw_txt)
            keys = json_loaded.keys()
            for j in keys:
                data = json_loaded[j]
                keys2 = data.keys()
                for k in keys2:
                    if k in _args:
                        if isinstance(_args[k], type(data[k])):
                            _args[k] = data[k]
                        else:
                            print(" [W] Argumet \"" + k + "\" is incorrect data type. Please change to \"" + str(type(_args[k])) + "\"")
                    elif k[0] == "#":
                        pass
                    else:
                        print(" [W] Argument \"" + k + "\" is not exsits.")
            # shapes properties
            _args["input_size_model"] = [_args["batch_size"], 52, 513]
            _args["input_size_test"] = [1, 52, 513]

            # initializing harvest directory
            _args["name_save"] = _args["checkpoint_dir"]+_args["model_name"] + _args["version"]
        except json.JSONDecodeError as er_message:
            print(" [W] JSONDecodeError: ", er_message)
            print(" [W] Use default setting")
    return _args
