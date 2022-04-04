import dlib
import os
import json
import face_alignment
from skimage import io
import numpy as np
import torch

path_to_shape_predictor = "Fill Here"
predictor = dlib.shape_predictor(os.path.join(path_to_shape_predictor, "shape_predictor_68_face_landmarks.dat"))
detector = dlib.get_frontal_face_detector()
# Dir of the images
ref_dir = "Fill Here"
fns_list = [i for i in os.listdir(ref_dir)]
fns = []
for fn in fns_list:
    for j in os.listdir(os.path.join(ref_dir, fn)):
        fns.append(os.path.join(fn, j))

p = {}
p_dlib = {}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device=device)

for j, f in enumerate(fns):
    try:
        input = io.imread(os.path.join(ref_dir, f))
        pred = fa.get_landmarks(input[:, :, :3])
        v = []
        if len(pred) > 0:
            print("P: Number of faces detected: {}".format(len(pred)))
        for pr in pred:
            v.append(pr.tolist())
        p[f] = v
    except:
        pass
       
    try:
        img = dlib.load_rgb_image(os.path.join(ref_dir, f))
        dets = detector(img, 1)
        p_dlib[f] = []
        print("P_Dlib: Number of faces detected: {}".format(len(dets)))
        for k, d in enumerate(dets):
            # Get the landmarks\parts for the face in box d.
            shape = predictor(img, d)
            # reye = [shape.part(i) for i in range(36, 42)]
            # leye = [shape.part(i) for i in range(42, 48)]
            flm = [[shape.part(i).x, shape.part(i).y] for i in range(68)]
            p_dlib[f].append(flm)
    except:
        pass


import json
import os


# Load original annotations files, including face annotations
anns = open(os.path.join('old_json', a))
anns = json.load(anns)
eyes_dict = p

fns_dict = {}
for img in anns['images']:
    fns_dict[img['id']] = img['file_name']

# Add annotations of eyes from deep-learning-based model into faces
new_anns = []
for ann in anns['annotations']:
    bbox = ann['bbox']
    if fns_dict[ann['image_id']][:-3]+'jpg' in eyes_dict.keys():
        for flm in eyes_dict[fns_dict[ann['image_id']][:-3]+'jpg']:
            if ann['category_id'] != 1:
                continue
            reye = flm[42:48]
            x_r = [e[0] for e in reye]
            y_r = [e[1] for e in reye]
            leye = flm[36:42]
            x_l = [e[0] for e in leye]
            y_l = [e[1] for e in leye]
            nose = flm[27:36]
            x_n = [e[0] for e in nose]
            dx_r = max(x_r) - min(x_r)
            dy_r = max(y_r) - min(y_r)
            dx_l = max(x_l) - min(x_l)
            dy_l = max(y_l) - min(y_l)

            if max(x_r) < bbox[0] + bbox[2] and min(x_l) > bbox[0] and max(y_r) < bbox[1] + bbox[3] and min(y_r) > bbox[1] and max(y_l) < bbox[1] + bbox[3] and min(y_l) > bbox[1]:
                area_r = (max(x_r) - min(x_r)) * (max(y_r) - min(y_r))
                area_l = (max(x_l) - min(x_l)) * (max(y_l) - min(y_l))
            else:
                continue
            if area_l > 1.5 * area_r or area_r > 1.5 * area_l or abs(min(x_l) - bbox[0]) > 3 * abs(bbox[0] + bbox[2] - max(x_r)) or 3 * abs(min(x_l) - bbox[0]) < abs(bbox[0] + bbox[2] - max(x_r)) or min(x_r) - max(x_l) < max(x_r) - min(x_r) or min(x_r) - max(x_l) < max(x_l) - min(x_l) or dy_r > dx_r or dy_l > dx_l:
                continue
            ann['has_reye'] = 1
            ann['reye_box'] = [min(x_r) - dx_r / 12, min(y_r) - dy_r / 5, max(x_r) - min(x_r) + 2 * dx_r / 12, max(y_r) - min(y_r) + dy_r * (1/5 + 1/7)]
            ann['has_leye'] = 1
            ann['leye_box'] = [min(x_l) - dx_l / 12, min(y_l) - dy_l / 5, max(x_l) - min(x_l) + 2 * dx_l / 12, max(y_l) - min(y_l) + dy_l * (1/5 + 1/7)]
    else:
        pass
    new_anns.append(ann)
anns['annotations'] = new_anns

# Add annotations of eyes from dlib model (=classical algorithm) into faces
eyes_dict_pdlib = p_dlib
anns_dlib = open(os.path.join('old_json', a))
anns_dlib = json.load(anns)

fns_dict = {}
for img in anns['images']:
    fns_dict[img['id']] = img['file_name']

new_anns_dlib = []
for ann in anns_dlib['annotations']:
    bbox = ann['bbox']
    if fns_dict[ann['image_id']][:-3]+'jpg' in eyes_dict_pdlib.keys():
        for flm in eyes_dict_pdlib[fns_dict[ann['image_id']][:-3]+'jpg']:
            if ann['category_id'] != 1:
                continue
            reye = flm[42:48]
            x_r = [e[0] for e in reye]
            y_r = [e[1] for e in reye]
            leye = flm[36:42]
            x_l = [e[0] for e in leye]
            y_l = [e[1] for e in leye]
            nose = flm[27:36]
            x_n = [e[0] for e in nose]
            dx_r = max(x_r) - min(x_r)
            dy_r = max(y_r) - min(y_r)
            dx_l = max(x_l) - min(x_l)
            dy_l = max(y_l) - min(y_l)

            if max(x_r) < bbox[0] + bbox[2] and min(x_l) > bbox[0] and max(y_r) < bbox[1] + bbox[3] and min(y_r) > bbox[1] and max(y_l) < bbox[1] + bbox[3] and min(y_l) > bbox[1]:
                area_r = (max(x_r) - min(x_r)) * (max(y_r) - min(y_r))
                area_l = (max(x_l) - min(x_l)) * (max(y_l) - min(y_l))
            else:
                continue
            if area_l > 1.5 * area_r or area_r > 1.5 * area_l or abs(min(x_l) - bbox[0]) > 3 * abs(bbox[0] + bbox[2] - max(x_r)) or 3 * abs(min(x_l) - bbox[0]) < abs(bbox[0] + bbox[2] - max(x_r)) or min(x_r) - max(x_l) < max(x_r) - min(x_r) or min(x_r) - max(x_l) < max(x_l) - min(x_l) or dy_r > dx_r or dy_l > dx_l:
                continue
            ann['has_reye'] = 1
            ann['reye_box'] = [min(x_r) - dx_r / 12, min(y_r) - dy_r / 5, max(x_r) - min(x_r) + 2 * dx_r / 12, max(y_r) - min(y_r) + dy_r * (1/5 + 1/7)]
            ann['has_leye'] = 1
            ann['leye_box'] = [min(x_l) - dx_l / 12, min(y_l) - dy_l / 5, max(x_l) - min(x_l) + 2 * dx_l / 12, max(y_l) - min(y_l) + dy_l * (1/5 + 1/7)]
    else:
        pass
    new_anns_dlib.append(ann)
anns_dlib['annotations'] = new_anns_dlib
        


combine_anns = []
for org_ann, dlib_ann in zip(anns['annotations'], eyes_dict_pdlib['annotations']):
    new_ann = org_ann
    if 'reye_box' not in org_ann.keys() and 'reye_box' not in dlib_ann.keys():
        pass
    elif 'reye_box' in org_ann.keys() and 'reye_box' not in dlib_ann.keys():
        new_ann = org_ann
    elif 'reye_box' not in org_ann.keys() and 'reye_box' in dlib_ann.keys():
        new_ann['has_reye'] = dlib_ann['has_reye']
        new_ann['reye_box'] = dlib_ann['reye_box']
    elif 'reye_box' in org_ann.keys() and 'reye_box' in dlib_ann.keys():
        new_x = max(org_ann['reye_box'][0], dlib_ann['reye_box'][0])
        new_y = min(org_ann['reye_box'][1], dlib_ann['reye_box'][1])
        new_dx = min(org_ann['reye_box'][0] + org_ann['reye_box'][2], dlib_ann['reye_box'][0] + dlib_ann['reye_box'][2]) - new_x
        new_dy = max(org_ann['reye_box'][1] + org_ann['reye_box'][3], dlib_ann['reye_box'][1] + dlib_ann['reye_box'][3]) - new_y
        if new_x < 0 or new_y < 0:
            new_ann = org_ann
        else:
            new_ann['reye_box'] = [new_x, new_y, new_dx, new_dy]
        
    if 'leye_box' not in org_ann.keys() and 'leye_box' not in dlib_ann.keys():
        pass
    elif 'leye_box' in org_ann.keys() and 'leye_box' not in dlib_ann.keys():
        pass
    elif 'leye_box' not in org_ann.keys() and 'leye_box' in dlib_ann.keys():
        new_ann['has_leye'] = dlib_ann['has_leye']
        new_ann['leye_box'] = dlib_ann['leye_box']
    elif 'leye_box' in org_ann.keys() and 'leye_box' in dlib_ann.keys():
        new_x = max(org_ann['leye_box'][0], dlib_ann['leye_box'][0])
        new_y = min(org_ann['leye_box'][1], dlib_ann['leye_box'][1])
        new_dx = min(org_ann['leye_box'][0] + org_ann['leye_box'][2], dlib_ann['leye_box'][0] + dlib_ann['leye_box'][2]) - new_x
        new_dy = max(org_ann['leye_box'][1] + org_ann['leye_box'][3], dlib_ann['leye_box'][1] + dlib_ann['leye_box'][3]) - new_y
        if new_x < 0 or new_y < 0:
            new_ann = org_ann
        else:
            new_ann['leye_box'] = [new_x, new_y, new_dx, new_dy]
    combine_anns.append(new_ann)
anns['annotations'] = combine_anns
with open(os.path.join(f'new_json_combined', a), 'w', encoding='utf-8') as ff:
    json.dump(anns, ff, indent=4)










