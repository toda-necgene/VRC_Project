from collections import namedtuple
import sys
import Model.model3 as model
debug=False
if __name__ == '__main__':
    pass
if len(sys.argv)!=0 and sys.argv.__contains__("--debug"):
    debug=True
net = model.Model(debug)
print(" [*]Building Model...")
net.build_model()
print(" [*]Built Model!!")
args=namedtuple('checkpoint_dir', 'train_size')
args.checkpoint_dir="./trained_models"
args.train_size=150

net.train(args)
