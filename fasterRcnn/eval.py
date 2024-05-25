import torch
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
from train import get_model
from torchvision.models.detection.image_list import ImageList
from torchvision.ops import nms
from dataset import VOCDetectionDataset

def plot_and_save_boxes(image, boxes, labels, scores, title, filename):
    plt.imshow(image)
    ax = plt.gca()
    for box, label, score in zip(boxes, labels, scores):
        x, y, w, h = box[0], box[1], box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x, y), w, h, fill=False, edgecolor='red', linewidth=2))
        ax.text(x, y, f'{label}: {score:.2f}', bbox=dict(facecolor='yellow', alpha=0.5))
    plt.title(title)
    plt.axis('off')
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

def apply_nms(boxes, scores, iou_threshold=0.5):
    keep = nms(boxes, scores, iou_threshold)
    return keep

def get_proposals_and_predictions(model, images):
    model.eval()
    with torch.no_grad():
        features = model.backbone(images.tensors)
        proposals, proposal_losses = model.rpn(images, features)
        detections, detector_losses = model.roi_heads(features, proposals, images.image_sizes)
    return proposals, detections

def predict_image_with_proposals(model, image_path, device, nms_threshold=0.5):
    transform = T.Compose([T.ToTensor()])
    img = Image.open(image_path)
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    images = ImageList(img_tensor, [(img_tensor.shape[-2], img_tensor.shape[-1])])
    
    proposals, detections = get_proposals_and_predictions(model, images)
    
    proposal_boxes = proposals[0].cpu()
    proposal_scores = torch.ones(proposal_boxes.shape[0])
    keep = apply_nms(proposal_boxes, proposal_scores, nms_threshold)
    proposal_boxes = proposal_boxes[keep].numpy()
    
    final_boxes = detections[0]['boxes'].cpu()
    final_scores = detections[0]['scores'].cpu()
    final_labels = [VOCDetectionDataset.classes[i] for i in detections[0]['labels'].cpu().numpy()]
    
    keep = apply_nms(final_boxes, final_scores, nms_threshold)
    final_boxes = final_boxes[keep].numpy()
    final_scores = final_scores[keep].numpy()
    final_labels = [final_labels[i] for i in keep]
    
    return img, proposal_boxes, final_boxes, final_labels, final_scores

if __name__ == "__main__":
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = get_model(21)
    model.load_state_dict(torch.load('fasterrcnn_voc2007.pth'))
    model.to(device)
    
    image = "person.jpg"
    image_path = f'image/{image}'
    img, proposal_boxes, final_boxes, final_labels, final_scores = predict_image_with_proposals(model, image_path, device)
    
    plot_and_save_boxes(img, proposal_boxes, ['Proposal'] * len(proposal_boxes), [1.0] * len(proposal_boxes), 'Proposal Boxes', f'{image}_proposal_boxes.png')
    plot_and_save_boxes(img, final_boxes, final_labels, final_scores, 'Final Boxes', f'{image}_final_boxes.png')
