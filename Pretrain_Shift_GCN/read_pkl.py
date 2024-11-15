import pickle
path = '/root/autodl-tmp/FGE/Pretrain_Shift_GCN/work_dir/ntu60_ShiftGCN_joint_xview_seen55_unseen5/eval_results/epoch_6_0.1288965715932008.pkl'

f = open(path, 'rb')
data = pickle.load(f)

print(data)
print(len(data))