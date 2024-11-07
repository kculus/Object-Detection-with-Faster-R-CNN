import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
import os
import json
from PIL import Image

# Modeli yüklemek için fonksiyon
def load_model(model_path, num_classes):
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights)

    # Son katmanı kendi veri setine göre ayarlama
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    
    # Model ağırlıklarını yükle
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Modeli değerlendirme moduna al
    return model

# Görüntü üzerinde tahmin yapma fonksiyonu
def infer(model, image_path, device):
    # Görüntüyü yükle ve ön işle
    image = Image.open(image_path).convert("RGB")
    image_tensor = torchvision.transforms.ToTensor()(image).unsqueeze(0).to(device)  # (1, C, H, W) formatına dönüştür

    # Model tahmini
    with torch.no_grad():
        predictions = model(image_tensor)
    
    return predictions

# Ana fonksiyon
def main():
    # Model ayarları
    model_path = 'fasterrcnn_finetuned2.pth'
    annotation_file = 'annotations/train.json'
    
    with open(annotation_file) as f:
        coco_data = json.load(f)
    
    num_classes = len(coco_data['categories']) + 1  # +1 background için
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # Modeli yükle
    model = load_model(model_path, num_classes).to(device)

    # Test resimlerinin bulunduğu klasör
    test_images_path = 'test-images'
    
    # image_file_name_to_image_id dosyasını yükle
    with open('image_file_name_to_image_id.json') as f:
        image_file_name_to_image_id = json.load(f)

    results = []
    for img_name in os.listdir(test_images_path):
        img_path = os.path.join(test_images_path, img_name)
        
        # Modeli kullanarak tahmin yap
        predictions = infer(model, img_path, device)

        # Tahmin sonuçlarını kaydet
        img_id = image_file_name_to_image_id.get(img_name)  # image_id'yi dosyadan al
        if img_id is None:
            print(f"Warning: {img_name} not found in image_file_name_to_image_id.json. Skipping.")
            continue

        bboxes = predictions[0]['boxes'].cpu().numpy()  # CPU'ya geç
        labels = predictions[0]['labels'].cpu().numpy()
        scores = predictions[0]['scores'].cpu().numpy()

        for bbox, label, score in zip(bboxes, labels, scores):
            # bbox'ları xyxy formatından xywh formatına dönüştür
            bbox[2], bbox[3] = bbox[2] - bbox[0], bbox[3] - bbox[1] 
            res = {
                'image_id': img_id,
                'category_id': int(label),  # Label 1'den başlıyorsa
                'bbox': list(bbox.astype('float64')),
                'score': float("{:.8f}".format(score))
            }
            results.append(res)

    # Sonuçları JSON dosyasına yaz
    with open('inference_results2.json', 'w') as f:
        json.dump(results, f, indent=4)

    print("Inference results saved to inference_results.json")

if __name__ == "__main__":
    main()
