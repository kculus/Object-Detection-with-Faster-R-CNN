import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.transforms import functional as F
from torch.utils.data import random_split
import os
import json
from PIL import Image
import random

# Veri artırımı dönüşümleri tanımları
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class RandomHorizontalFlip(object):
    def __init__(self, flip_prob=0.5):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            width, _ = image.size
            boxes = target['boxes']
            boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
            target['boxes'] = boxes
        return image, target

class RandomRotation(object):
    def __init__(self, degrees):
        self.degrees = degrees

    def __call__(self, image, target):
        angle = random.uniform(-self.degrees, self.degrees)
        image = F.rotate(image, angle)
        # Bounding box'ları döndürmek karmaşık olduğu için sadece görüntüyü döndürmekte kalabilirsiniz
        return image, target

class ColorJitter(object):
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1):
        self.color_jitter = torchvision.transforms.ColorJitter(
            brightness=brightness, contrast=contrast, saturation=saturation, hue=hue
        )

    def __call__(self, image, target):
        image = self.color_jitter(image)
        return image, target

class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target

# CustomDataset sınıfı
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, annotation_file, img_dir, transforms=None, basic_transforms=None):
        self.img_dir = img_dir
        self.transforms = transforms
        self.basic_transforms = basic_transforms
        with open(annotation_file) as f:
            self.coco_data = json.load(f)
        self.images = self.coco_data['images']
        self.annotations = self.coco_data['annotations']
        self.category_map = {cat['id']: cat['name'] for cat in self.coco_data['categories']}

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        image = Image.open(img_path).convert("RGB")
        
        # Anotasyonları alın
        boxes = []
        labels = []
        for annot in self.annotations:
            if annot['image_id'] == img_info['id']:
                xmin, ymin, width, height = annot['bbox']
                boxes.append([xmin, ymin, xmin + width, ymin + height])
                labels.append(annot['category_id'])
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels

        # Nadir sınıfları kontrol edin
        rare_class_ids = [3, 4]  # `futbol_sahası` ve `silo` ID'leri (örnek olarak)
        if any(label.item() in rare_class_ids for label in labels):
            # Veri artırımı dönüşümlerini uygulayın
            if self.transforms is not None:
                image, target = self.transforms(image, target)
        else:
            # Temel dönüşümleri uygulayın
            if self.basic_transforms is not None:
                image = self.basic_transforms(image)

        return image, target

def collate_fn(batch):
    return tuple(zip(*batch))

if __name__ == "__main__":
    # Gelişmiş veri artırımı dönüşümleri
    data_transforms = Compose([
        RandomHorizontalFlip(0.5),
        RandomRotation(15),  # 15 dereceye kadar rastgele döndürme
        ColorJitter(0.2, 0.2, 0.2, 0.1),  # Renk ayarlamaları
        ToTensor(),
    ])

    # Temel dönüşümler (diğer sınıflar için)
    basic_transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])

    # Veri setini oluştur
    dataset = CustomDataset(
        annotation_file='annotations/train.json',
        img_dir='images',
        transforms=data_transforms,
        basic_transforms=basic_transforms
    )

    # Veri setini eğitim ve doğrulama olarak böl
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # DataLoader'ları oluştur
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=True, num_workers=2, collate_fn=collate_fn
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=2, collate_fn=collate_fn
    )

    # Modeli yükle ve özelleştir
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights)

    num_classes = len(dataset.category_map) + 1  # Background için +1
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

    # Modeli cihazınıza atayın
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    # Sadece son katmanı eğitmek için gövde katmanlarını dondurun
    for param in model.backbone.parameters():
        param.requires_grad = False

    # Optimizer ve öğrenme oranı güncelleyici
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # Eğitim döngüsü
    num_epochs = 5

    for epoch in range(num_epochs):
        # Eğitim modu
        model.train()
        running_loss = 0.0
        for images, targets in train_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # İleri yayılım
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            # Geri yayılım
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            running_loss += losses.item()

        # Öğrenme hızını güncelle
        lr_scheduler.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {running_loss / len(train_loader)}')

        # Doğrulama (Validation) modu
        model.train()  # Validation set üzerinde loss hesaplayabilmek için geçici olarak train moduna alıyoruz
        val_loss = 0.0
        with torch.no_grad():  # Gradyan hesaplamasını kapatıyoruz
            for images, targets in val_loader:
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                # Doğrulama veri setinde loss hesaplayın
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                val_loss += losses.item()

        print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss / len(val_loader)}')

    # Modeli kaydet
    torch.save(model.state_dict(), 'fasterrcnn_finetuned2.pth')
