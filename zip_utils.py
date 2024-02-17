from torchvision.transforms import Compose, Resize, ToTensor, Normalize, InterpolationMode
import torch
import numpy as np
import matplotlib.pyplot as plt
import PIL
import PIL.Image as Image
import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure, color
from torch import nn
import os
import cv2
from CLIP_Surgery import clip
import torch.nn.functional as F
from collections import Counter
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

def _convert_image_to_rgb(image):
    return image.convert("RGB")

h_n_px = 224
BICUBIC = InterpolationMode.BICUBIC
H1_ToTensor = Compose([
        Resize(h_n_px, interpolation=BICUBIC),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

h_n_px = 512
BICUBIC = InterpolationMode.BICUBIC
H2_ToTensor = Compose([
        Resize((h_n_px,h_n_px), interpolation=BICUBIC),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

l_n_px = 2048
BICUBIC = InterpolationMode.BICUBIC
L_ToTensor = Compose([
        Resize(l_n_px, interpolation=BICUBIC),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

def bbox_area(box1, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if x1y1x2y2:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]

    else:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2


    # Union Area
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)

    # print(box1, box1.shape)
    # print(box2, box2.shape)
    return b1_area

def bbox_area(box1, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if x1y1x2y2:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]

    else:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2


    # Union Area
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)

    # print(box1, box1.shape)
    # print(box2, box2.shape)
    return b1_area

def mask2bbox(mask):
    # mask: [N, H, W]
    # bbox: [N, 4]
    N, H, W = mask.shape
    bbox = torch.zeros(N, 4, dtype=torch.float32, device=mask.device)
    for i in range(N):
        m = mask[i]

        idx = torch.nonzero(m)
        if idx.shape[0] == 0:
            continue
        bbox[i, 0] = idx[:, 1].min()
        bbox[i, 1] = idx[:, 0].min()
        bbox[i, 2] = idx[:, 1].max()
        bbox[i, 3] = idx[:, 0].max()
    return bbox

color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
def show_mask(mask, ax, random_color=False):
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='blue', facecolor=(0,0,0,0), lw=3)) 

def resize_pos(x1,y1,src_size,tar_size):
    w1=src_size[0]
    h1=src_size[1]
    w2=tar_size[1]
    h2=tar_size[0]
    y2=(h2/h1)*y1
    x2=(w2/w1)*x1
    return x2,y2
device = 'cuda:2'
def inference_on_one_image(semantic_model, clustering_model, image_root, class_name):
    # Load image
    I = Image.open(image_root).convert('RGB')
    image = cv2.imread(image_root)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor = H2_ToTensor(I).unsqueeze(0)
    tensor = tensor.to(device)
    print(tensor.shape)

    with torch.no_grad():
        text_features = clip.encode_text_with_prompt_ensemble(semantic_model, class_name, device)
        redundant_feats = clip.encode_text_with_prompt_ensemble(semantic_model, [''], device)
        image_features0 = semantic_model.encode_image(tensor)
        image_features0 = image_features0 / image_features0.norm(dim=1, keepdim=True)
        similarity0 = clip.clip_feature_surgery(image_features0, text_features, redundant_feats)
        similarity_map0 = clip.get_similarity_map(similarity0[:, 1:, :], image.shape[:2]).squeeze(0)
        similarity_map0 = similarity_map0.cpu().numpy()

    with torch.no_grad():
        tensor = L_ToTensor(I).unsqueeze(0)
        tensor = tensor.to(device)
        feature = clustering_model(tensor)[0]
        c, h, w = feature.shape
        feature_flat = feature.permute(1, 2, 0).reshape(-1, c)
        k = 20
        centroid_indices = torch.randint(feature_flat.shape[0], size=(k,))
        centroid_indices = torch.linspace(0, h*w-1, k)
        centroid_indices = torch.round(centroid_indices)
        centroid_indices = torch.unique(centroid_indices)
        centroid_indices = torch.tensor(centroid_indices, dtype=torch.long)
        centroid_indices = centroid_indices.to(device)
        new_centroids = feature_flat[centroid_indices]

        max_iterations = 100
        for i in range(max_iterations):
            distances = torch.cdist(feature_flat, new_centroids, p=2)
            cluster_indices = torch.argmin(distances, dim=1)
            old_centroids = new_centroids.clone()
            for j in range(k):
                mask = cluster_indices == j
                cluster_points = feature_flat[mask]
                if len(cluster_points) > 0:
                    new_centroids[j] = cluster_points.mean(dim=0)
                else:
                    new_centroids[j] = old_centroids[j]
            if torch.allclose(new_centroids, old_centroids, rtol=1e-5):
                break
        cluster_labels = cluster_indices.reshape(feature.shape[1], feature.shape[2])
        cluster_labels = cluster_labels.cpu().numpy()
        cluster_labels = cluster_labels + 1

    class_act = similarity_map0[..., 0]
    threshold = np.mean(class_act) * 0.7
    class_act_resize = F.interpolate(torch.tensor(class_act).unsqueeze(0).unsqueeze(0).float(), size=cluster_labels.shape, mode='nearest').squeeze(0).squeeze(0).cpu().numpy()
    class_act_resize = np.asarray(class_act_resize) > np.asarray(threshold)
    cluster_labels_class = cluster_labels * class_act_resize

    counter = Counter(np.asarray(cluster_labels_class.flatten()))
    overall_map = np.float64(cluster_labels == -1)
    for sub_class, sub_num in counter.most_common(20):
        if sub_class == 0 or sub_class in [-1]:
            continue
        else:
            specific_map = np.float64(cluster_labels == sub_class)
            specific_map_instance = measure.label(specific_map, connectivity=1, background=0)
            properties = measure.regionprops(specific_map_instance)
            for prop in properties:
                if prop.label == 0:
                    continue
                i = prop.label
                instance_map = np.float64(specific_map_instance == i)
                gain1 = np.sum(instance_map * class_act_resize) / np.sum(instance_map)
                loss1 = bbox_area(prop.bbox) / prop.area
                # print(loss1)
                if  (gain1 > 0.3 and loss1 < 3):
                    overall_map = overall_map + instance_map
                    
    labels = measure.label(overall_map != 0, connectivity=1, background=0)
    draw_labels = overall_map != 0
    draw_labels_resize = F.interpolate(torch.tensor(draw_labels).unsqueeze(0).unsqueeze(0).float(), size=(I.size[1], I.size[0]), mode='nearest').squeeze(0).squeeze(0).cpu().numpy()
    camp = plt.get_cmap('tab20', np.max(labels))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    show_mask(draw_labels_resize, plt.gca(), random_color=False)
    ax = plt.gca()
    ax.set_axis_off()

    plt.subplot(1, 2, 2)
    plt.imshow(image)
    ax = plt.gca()
    ax.set_axis_off()
    scale1 = I.size[0] / labels.shape[0]
    scale2 = I.size[1] / labels.shape[1]
    properties = measure.regionprops(labels)
    for prop in properties:
        if prop.area < 100:
            continue
        x1,y1 = resize_pos(prop.bbox[1] ,prop.bbox[0] ,cluster_labels.shape,I.size) 
        x2,y2 = resize_pos(prop.bbox[3] ,prop.bbox[2] ,cluster_labels.shape,I.size)
        show_box([x1,y1,x2,y2], plt.gca())
    return None
