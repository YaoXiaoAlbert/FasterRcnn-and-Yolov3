import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from dataset import get_data_loader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def get_model(num_classes):
    weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def train_model(data_loader, num_classes=21, num_epochs=10, lr=0.005, momentum=0.9, weight_decay=0.0005, step_size=3, gamma=0.1):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = get_model(num_classes)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    writer = SummaryWriter()
    
    for epoch in range(num_epochs):
        model.train()
        i = 0
        for images, targets in data_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            if i % 10 == 0:
                writer.add_scalar('Loss/train', losses.item(), epoch * len(data_loader) + i)
            
            i += 1
        
        lr_scheduler.step()

    writer.close()
    torch.save(model.state_dict(), 'fasterrcnn_voc2007.pth')
    return model

if __name__ == "__main__":
    data_loader = get_data_loader(root='/home/yao/detection/fasterRCNN/datasets')
    num_classes=21
    num_epochs=10
    lr=0.005
    momentum=0.9
    weight_decay=0.0005
    step_size=3
    gamma=0.1
    model = train_model(data_loader, num_classes, num_epochs, lr, momentum, weight_decay, step_size, gamma)
