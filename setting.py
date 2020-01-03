"""
setting file
"""
import os
def get_setting():
    """
    return settings as dictionary object
    
    Returns
    -------
    _args: dict


    Internal Params
     ----------
      checkpoint_dir: path(str)
                    the weight file output **parent** dir.
          model_name: str
                    a name of experiment without versions.
             version: str
                    a number of experiment.
                    finary path = os.path.join("checkpoint_dir",  "model_name" + "version")
     use_old_dataset: bool 
     train_iteration: int
                 gpu: int
                    if a negative number, use cpu not gpu.
          batch_size: list of int
                    it changes in order every 2000 itrs.
          input_size: int
                    the audio-wave frames of intput.
  f0_estimation_plan: "harvest" or "dio" default="dio"
                    algorithm name of f0 estimation.
                test: bool
        wave_otp_dir: path (str)
        log_interval: int
                    the intervals of test and save. 
         line_notify: path (str)

    """

    _args = {}
    _args["checkpoint_dir"] = "./trained_model/"
    _args["model_name"] = "VRC"
    _args["version"] = "20.01.01.02"
    _args["use_old_dataset"] = True
    _args["batch_size"] = [16, 16, 8]
    _args["train_iteration"] = 15000
    _args["input_size"] = 88199
    _args["f0_estimation_plan"] = "harvest"
    _args["gpu"] = 0
    _args["test"] = True
    _args["wave_otp_dir"] = "./waves/"
    _args["log_interval"] = 100
    _args["line_notify"] = "_line_api_token.txt"
    # _args["line_notify"] = ""
    _args["name_save"] = os.path.join(_args["checkpoint_dir"],  _args["model_name"] + _args["version"])
    
    return _args
