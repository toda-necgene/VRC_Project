import sys
debug=False
if __name__ == '__main__':
    pass
from .Model.model_seed import Model  as model
path="../setting.json"
if len(sys.argv)!=0 and sys.argv.__contains__("--debug"):
    debug=True
if len(sys.argv)!=0 and sys.argv.__contains__("--path"):
    i= sys.argv.index("--path")
    path=sys.argv[i+1]
net = model(path)
print(" [*] Building Model...")
net.build_model()
print(" [*] Built Model!!")
net.train()
