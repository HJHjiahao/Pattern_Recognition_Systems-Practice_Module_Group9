import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

import numpy as np
import torch

from PIL import Image
from torchvision import transforms

import models.mask_net as mask_net
from models.mask_net.faster_rcnn import FastRCNNPredictor, TwoMLPHead
from models.mask_net.rpn import AnchorGenerator

from PIL import Image as PILImage

# Lung mask segmentation
def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model

def load_lung_mask_model():
	model = get_model_instance_segmentation(2)
	model.load_state_dict(torch.load('../lung_mask_model.pth'))

	return model


def get_lung_mask(img_path, model):
	device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

	img = Image.open(img_path).convert("RGB")

	t_ = transforms.Compose([transforms.ToTensor()])
	img = np.array(img)
	img = t_(img)

	model.eval().to(device)
	with torch.no_grad():
	    prediction = model([img.to(device)])

	mask = prediction[0]['masks'][0, 0].mul(255).byte().cpu().numpy()
	kernel = np.ones((16,16), np.uint8)

	mask2 = cv2.morphologyEx(np.uint8(mask), cv2.MORPH_CLOSE, kernel)

	_, mask2 = cv2.threshold(mask, 2, 255, cv2.THRESH_BINARY)
	mask2 = cv2.morphologyEx(np.uint8(mask2), cv2.MORPH_CLOSE, kernel)
	mask2 = mask2/255

	# row_img = img.mul(255).permute(1, 2, 0).byte().numpy()
	# row_img = cv2.bitwise_and(row_img, row_img, mask=np.uint8(mask2))
	return mask2

# Segmentation application
def load_segmentation_model():
	from models.mask_net.rpn_segmentation import AnchorGenerator

	device = torch.device('cpu')
	confidence_threshold = 0.05
	mask_threshold = 0.5
	backbone_name = 'resnet50'
	rpn_nms = 0.75
	roi_nms = 0.5
	truncation = '0'

	n_c = 3
	ckpt = torch.load('pretrained_models/segmentation_model_both_classes.pth', map_location=device)

	model_name = None
	if 'model_name' in ckpt.keys():
	    model_name = ckpt['model_name']

	sizes = ckpt['anchor_generator'].sizes
	aspect_ratios = ckpt['anchor_generator'].aspect_ratios
	anchor_generator = AnchorGenerator(sizes, aspect_ratios)
	print("Anchors: ", anchor_generator.sizes, anchor_generator.aspect_ratios)

	# create modules
	# this assumes FPN with 256 channels
	box_head = TwoMLPHead(in_channels=7 * 7 * 256, representation_size=128)
	if backbone_name == 'resnet50':
	    maskrcnn_heads = None
	    box_predictor = FastRCNNPredictor(in_channels=128, num_classes=n_c)
	    mask_predictor = MaskRCNNPredictor(in_channels=256, dim_reduced=256, num_classes=n_c)
	else:
	    #Backbone->FPN->boxhead->boxpredictor
	    box_predictor = FastRCNNPredictor(in_channels=128, num_classes=n_c)
	    maskrcnn_heads = MaskRCNNHeads(in_channels=256, layers=(128,), dilation=1)
	    mask_predictor = MaskRCNNPredictor(in_channels=128, dim_reduced=128, num_classes=n_c)

	# keyword arguments
	maskrcnn_args = {'num_classes': None, 'min_size': 512, 'max_size': 1024, 'box_detections_per_img': 100,
	                  'box_nms_thresh': roi_nms, 'box_score_thresh': confidence_threshold, 'rpn_nms_thresh': rpn_nms,
	                  'box_head': box_head, 'rpn_anchor_generator': anchor_generator, 'mask_head':maskrcnn_heads,
	                  'mask_predictor': mask_predictor, 'box_predictor': box_predictor}

	# Instantiate the segmentation model
	maskrcnn_model = mask_net.maskrcnn_resnet_fpn(backbone_name, truncation, pretrained_backbone=False, **maskrcnn_args)
	# Load weights
	maskrcnn_model.load_state_dict(ckpt['model_weights'])
	return maskrcnn_model

def get_segmentation(path, maskrcnn_model):
	import copy
	import matplotlib.pyplot as plt
	from matplotlib.patches import Rectangle
	import cv2

	ct_classes = {0: '__bgr', 1: 'GGO', 2: 'CL'}
	ct_colors = {1: 'red', 2: 'blue', 'mask_cols': np.array([[255, 0, 0], [0, 0, 255]])} 
	confidence_threshold = 0.05
	mask_threshold = 0.5

	maskrcnn_model.eval().to(device)

	im = PILImage.open(path)
	# convert image to RGB, remove the alpha channel
	if im.mode != 'RGB':
	    im = im.convert(mode='RGB')
	img = np.array(im)
	# copy image to make background for plotting
	bgr_img = copy.deepcopy(img)
	if img.shape[2] > 3:
	    img = img[:, :, :3]
	# torchvision transforms, the rest Mask R-CNN does internally
	t_ = transforms.Compose([
	    transforms.ToPILImage(),
	    transforms.ToTensor()])
	img = t_(img).to(device)
	out = maskrcnn_model([img])
	# scores + bounding boxes + labels + masks
	scores = out[0]['scores']
	bboxes = out[0]['boxes']
	classes = out[0]['labels']
	mask = out[0]['masks']
	# this is the array for all masks
	best_scores = scores[scores > confidence_threshold]
	# Are there any detections with confidence above the threshold?
	if len(best_scores):
	    best_idx = np.where(scores > confidence_threshold)
	    best_bboxes = bboxes[best_idx]
	    best_classes = classes[best_idx]
	    best_masks = mask[best_idx]
	    print('bm', best_masks.shape)
	    mask_array = np.zeros([best_masks[0].shape[1], best_masks[0].shape[2], 3], dtype=np.uint8)
	    fig, ax = plt.subplots(1, 1)
	    fig.set_size_inches(12, 6)
	    ax.axis("off")
	    # plot predictions
	    for idx, dets in enumerate(best_bboxes):
	        found_masks = best_masks[idx][0].detach().clone().to(device).numpy()
	        pred_class = best_classes[idx].item()
	        pred_col_n = ct_colors[pred_class]
	        pred_class_txt = ct_classes[pred_class]
	        pred_col = ct_colors['mask_cols'][pred_class - 1]
	        mask_array[found_masks > mask_threshold] = pred_col
	        rect = Rectangle((dets[0], dets[1]), dets[2] - dets[0], dets[3] - dets[1], linewidth=1,
	                          edgecolor=pred_col_n, facecolor='none', linestyle="--")
	        ax.text(dets[0] + 40, dets[1], '{0:}'.format(pred_class_txt), fontsize=10, color=pred_col_n)
	        ax.text(dets[0], dets[1], '{0:.2f}'.format(best_scores[idx]), fontsize=10, color=pred_col_n)
	        ax.add_patch(rect)

	    added_image = cv2.addWeighted(bgr_img, 0.5, mask_array, 0.75, gamma=0)
	    ax.imshow(added_image)
	    # fig.savefig(os.path.join(save_dir, model_name + "_" + str(num) + ".png"),
	    #             bbox_inches='tight', pad_inches=0.0)

	else:
	    print("No detections")

	return added_image


# Classification application covid-mask-net
def load_mask_net_model():
	from models.mask_net.rpn import AnchorGenerator

	s_features = 1024
	n_c = 3
	device = 'cuda'

	covid_mask_net_args = {'num_classes': None, 'min_size': 512, 'max_size': 1024, 'box_detections_per_img': 256,
	                        'box_nms_thresh': 0.75, 'box_score_thresh': -0.01, 'rpn_nms_thresh': 0.75}



	if torch.cuda.is_available() and device == 'cuda':
	    device = torch.device('cuda')
	else:
	    device = torch.device('cpu')
	# either 2+1 or 1+1 classes
	ckpt = torch.load('pretrained_models/classification_model_both_classes.pth', map_location=device)

	sizes = ckpt['anchor_generator'].sizes
	aspect_ratios = ckpt['anchor_generator'].aspect_ratios
	anchor_generator = AnchorGenerator(sizes, aspect_ratios)
	# Faster R-CNN interfaces, masks not implemented at this stage
	box_head = TwoMLPHead(in_channels=256*7*7, representation_size=128)
	box_predictor = FastRCNNPredictor(in_channels=128, num_classes=n_c)
	# Mask prediction is not necessary, keep it for future extensions
	covid_mask_net_args['rpn_anchor_generator'] = anchor_generator
	covid_mask_net_args['box_predictor'] = box_predictor
	covid_mask_net_args['box_head'] = box_head
	# representation size of the S classification module
	# these should be provided in the config
	covid_mask_net_args['s_representation_size'] = s_features

	covid_mask_net_model = mask_net.fasterrcnn_resnet_fpn("resnet50", "1", **covid_mask_net_args)
	covid_mask_net_model.load_state_dict(ckpt['model_weights'])
	return covid_mask_net_model

def get_mask_net_classificaction(path, covid_mask_net_model):
	covid_mask_net_model.eval().to(device)
	im = PILImage.open(path)
	img = np.array(im)

	if img.shape[2] > 3:
		img = img[:, :, :3]

	t_ = transforms.Compose([
	    transforms.ToPILImage(),
	    transforms.Resize(512),
	    transforms.ToTensor()])
	img = t_(img)
	if device == torch.device('cuda'):
	    img = img.to(device)

	out = covid_mask_net_model([img])
	pred_class = out[0]['final_scores'].argmax().item()
	
	if(pred_class == 0):
		return 'Normal'
	elif(pred_class == 1):
		return 'Pneumonia'
	else:
		return 'Covid19'

# # Classification application resnet
# def get_resnet_classificaction():


# if __name__ == '__main__':
