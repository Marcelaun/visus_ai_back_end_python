import torch
import torch.nn as nn
from PIL import Image
import io
import logging
import timm
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

MODEL_PATH = "model_weights/best_model.pth"
DEVICE = torch.device("cpu") 
CLASS_NAMES = ['Normal', 'RD Leve', 'RD Moderada', 'RD Severa', 'RD Proliferativa']
GRAVIDADE = {
    'Normal': 0,
    'RD Leve': 1,
    'RD Moderada': 2,
    'RD Severa': 3,
    'RD Proliferativa': 4,
}
logger = logging.getLogger(__name__)

# Transformações IDÊNTICAS ao código de teste/treinamento (usando Albumentations)
transforms_pipeline = A.Compose([
    A.Resize(512, 512),  # Exatamente como no treinamento
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])


class EfficientNetB4_DR(nn.Module):
    """Arquitetura IDÊNTICA ao treinamento"""
    def __init__(self, num_classes=5, pretrained=False, dropout_rate=0.3):
        super(EfficientNetB4_DR, self).__init__()
        self.backbone = timm.create_model(
            'efficientnet_b4',
            pretrained=pretrained,
            num_classes=0,
            global_pool='avg',
            drop_rate=0.0
        )
        num_features = self.backbone.num_features

        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        output = self.classifier(features)
        return output


def load_model():
    """Carrega o modelo treinado COMPLETO (backbone + classificador)."""
    try:
        logger.info(f"Carregando modelo de: {MODEL_PATH}")
        
        # 1. Cria a arquitetura do modelo SEM pesos pré-treinados
        #    Isso é crucial! Queremos usar APENAS os pesos do seu treinamento
        model = EfficientNetB4_DR(
            num_classes=len(CLASS_NAMES),
            pretrained=False,  # <-- NÃO carrega pesos do ImageNet!
            dropout_rate=0.3
        )
        
        # 2. Carrega o checkpoint
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
        logger.info("Checkpoint carregado com sucesso")

        # 3. Extrai o state_dict
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            logger.info("Chave 'model_state_dict' encontrada no checkpoint")
        else:
            state_dict = checkpoint
            logger.info("Usando checkpoint direto como state_dict")

        # 4. Remove prefixo 'module.' se existir (caso tenha usado DataParallel)
        if any(key.startswith('module.') for key in state_dict.keys()):
            logger.info("Removendo prefixo 'module.' de DataParallel...")
            state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}

        # 5. Carrega TODOS os pesos (backbone + classificador) com strict=True
        #    Agora não há conflito porque o modelo foi criado sem pesos pré-treinados
        model.load_state_dict(state_dict, strict=True)
        logger.info("✅ State_dict carregado completamente (strict=True)")
        
        # 6. Coloca em modo de avaliação
        model.eval()
        model.to(DEVICE)
        
        # 7. Log das métricas do checkpoint
        if 'best_kappa' in checkpoint:
            logger.info(f"📊 Kappa do modelo: {checkpoint['best_kappa']:.4f}")
        if 'best_accuracy' in checkpoint:
            logger.info(f"📊 Acurácia do modelo: {checkpoint['best_accuracy']:.4f}")
        if 'epoch' in checkpoint:
            logger.info(f"📊 Época do treinamento: {checkpoint['epoch']}")
        
        logger.info("✅ Modelo carregado e pronto para inferência!")
        return model
    
    except FileNotFoundError:
        logger.error(f"❌ Arquivo do modelo não encontrado: {MODEL_PATH}")
        return None
    except Exception as e:
        logger.error(f"❌ Erro ao carregar o modelo: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def predict_image(model, image_bytes):
    """Realiza predição em uma imagem."""
    try:
        # 1. Carrega a imagem como array numpy (igual ao Colab)
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image_np = np.array(image)  # Converte para numpy array
        
        # 2. Aplica as transformações do Albumentations
        transformed = transforms_pipeline(image=image_np)
        image_tensor = transformed['image'].unsqueeze(0).to(DEVICE)
        
        # 3. Realiza a predição
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            conf, pred_idx = torch.max(probabilities, 0)
            
            predicted_class = CLASS_NAMES[pred_idx.item()]
            confidence_percent = conf.item() * 100

            # 4. Prepara todas as probabilidades
            all_probabilities = [
                {
                    "label": CLASS_NAMES[i], 
                    "value": round(probabilities[i].item() * 100, 2)
                }
                for i in range(len(CLASS_NAMES))
            ]

            result = {
                "diagnosis": predicted_class,
                "confidence": round(confidence_percent, 2),
                "probabilities": all_probabilities,
                "gravity_score": GRAVIDADE[predicted_class]
            }
            
            logger.info(f"Predição: {predicted_class} ({confidence_percent:.2f}%)")
            return result

    except Exception as e:
        logger.error(f"Erro durante a predição: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {"error": str(e)}