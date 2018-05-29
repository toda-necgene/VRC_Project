import sys
debug=False
if __name__ == '__main__':
    pass
if len(sys.argv)!=0 and sys.argv.__contains__("--train"):
    import train
elif len(sys.argv)!=0 and sys.argv.__contains__("--test"):
    from utils import model_test
elif len(sys.argv)!=0 and sys.argv.__contains__("--run"):
     import run
elif len(sys.argv)!=0 and sys.argv.__contains__("--voice-profile"):
    from utils import voice_profile