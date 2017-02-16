EMB_DIM = 300
HIDDEN_DIM = 300
SEQ_LENGTH = 3
START_TOKEN = 0
SEED = 88
generated_num = 10000
BATCH_SIZE = 16

feat_style = 0  # 0-text, 1-img, 2-txt+img, 3-txt+img+mm
category = 'homicide'
if feat_style == 0:
    positive_file = '../dataset/feat_pool/' + category + '_in_num_w2v.npy'
elif feat_style == 1:
    positive_file = '../dataset/feat_pool/' + category + '_in_num_img.npy'
elif feat_style == 2:
    positive_file = '../dataset/feat_pool/' + category + '_in_num_w2v_img.npy'
elif feat_style == 3:
    positive_file = '../dataset/feat_pool/' + category + '_in_num_w2v_img_mm.npy'

