import numpy as np
from vrc_project.setting_loader import load_setting_from_json
def model_simplify(model_weight_name):
    def fix_weight(weight, u, axis):
        w = weight.reshape(weight.shape[axis], -1)
        for _ in range(5):
            v = np.dot(u, w) / (np.linalg.norm(np.dot(u, w)) + 1e-8)
            u = np.dot(w, v) / (np.linalg.norm(np.dot(w, v)) + 1e-8)
        sigma = np.matmul(np.matmul(u, w), v)
        w_fixed = w / sigma
        w_fixed = w_fixed.reshape(weight.shape)
        return w_fixed
    gen_ab = np.load(model_weight_name+"/gen_ab.npz")
    w_list = dict()
    for k in gen_ab.keys():
        if k.endswith("/W_u"):
            a = 0
            if k.startswith("d"):
                a=1
            w_list[k[:-2]] = fix_weight(gen_ab[k[:-2]], gen_ab[k], axis=a)
        elif k.endswith("/b"):
            w_list[k] = gen_ab[k]  
    np.savez(model_weight_name+"/gen_ab_fix.npz", **(w_list))