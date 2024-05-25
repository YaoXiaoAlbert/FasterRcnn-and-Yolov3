import torch
import torchvision
from torchvision.transforms import functional as F
from PIL import Image

class VOCDetectionDataset(torchvision.datasets.VOCDetection):
    classes = [
        "__background__", "aeroplane", "bicycle", "bird", "boat",
        "bottle", "bus", "car", "cat", "chair", "cow",
        "diningtable", "dog", "horse", "motorbike", "person",
        "pottedplant", "sheep", "sofa", "train", "tvmonitor"
    ]
    def __init__(self, root, year, image_set, transform):
        super().__init__(root=root, year=year, image_set=image_set, transform=transform)
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
    
    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)
        if isinstance(img, torch.Tensor):
            img = F.to_pil_image(img)
        img = F.to_tensor(img)
        
        boxes = []
        labels = []
        for obj in target['annotation']['object']:
            xmin = int(obj['bndbox']['xmin'])
            ymin = int(obj['bndbox']['ymin'])
            xmax = int(obj['bndbox']['xmax'])
            ymax = int(obj['bndbox']['ymax'])
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_to_idx[obj['name']])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {'boxes': boxes, 'labels': labels}
        return img, target

def get_data_loader(root, year='2007', image_set='train', batch_size=2):
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    dataset = VOCDetectionDataset(root=root, year=year, image_set=image_set, transform=transform)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    return data_loader
