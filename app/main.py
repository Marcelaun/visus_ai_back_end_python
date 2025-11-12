import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from contextlib import asynccontextmanager
from typing import List
import logging # 1. Importe o módulo de logging

# Importa nossas funções do outro arquivo
from . import model_loader 

# 2. Configure o logger
# Isso vai formatar nossas mensagens com Data/Hora, Nível e a Mensagem
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)


# Dicionário para guardar nosso modelo carregado
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Carrega o modelo de IA UMA VEZ na inicialização
    logger.info("Iniciando a aplicação...")
    ml_models["retinopathy_model"] = model_loader.load_model()
    yield
    # Limpa o modelo ao desligar
    logger.info("Desligando a aplicação...")
    ml_models.clear()

# Inicializa o app FastAPI com o 'lifespan'
app = FastAPI(lifespan=lifespan)

@app.get("/")
def read_root():
    return {"status": "Serviço de IA da VisusAI está online."}

@app.post("/predict")
async def create_upload_file(file: UploadFile = File(...)):
    """
    Endpoint principal. Recebe um upload de imagem e retorna
    a predição do modelo.
    """
    if not ml_models.get("retinopathy_model"):
        logger.error("Tentativa de predição sem modelo carregado.")
        raise HTTPException(status_code=500, detail="Modelo de IA não foi carregado.")

    logger.info(f"Recebida 1 nova imagem para predição: {file.filename}")

    # Lê os bytes da imagem enviada
    image_bytes = await file.read()
    
    # Faz a predição
    prediction = model_loader.predict_image(
        model=ml_models["retinopathy_model"],
        image_bytes=image_bytes
    )
    
    if "error" in prediction:
        logger.warning(f"Erro ao processar a imagem {file.filename}: {prediction['error']}")
        raise HTTPException(status_code=400, detail=prediction["error"])

    # 3. Loga o resultado da predição
    logger.info(f"Resultado da predição: {prediction['diagnosis']} (Confiança: {prediction['confidence']:.2f}%)")
    
    return prediction

# --- O que seu Laravel vai usar ---
@app.post("/predict_batch")
async def create_upload_files(files: List[UploadFile] = File(...)):
    """
    Endpoint BÔNUS. Recebe VÁRIAS imagens (como discutimos)
    e retorna o resultado MAIS GRAVE.
    """
    if not ml_models.get("retinopathy_model"):
        logger.error("Tentativa de predição em batch sem modelo carregado.")
        raise HTTPException(status_code=500, detail="Modelo de IA não foi carregado.")

    # 3. Loga o recebimento
    logger.info(f"Recebido batch de {len(files)} imagens para predição.")

    results = []
    
    for file in files:
        image_bytes = await file.read()
        prediction = model_loader.predict_image(
            model=ml_models["retinopathy_model"],
            image_bytes=image_bytes
        )
        if "error" in prediction:
            logger.warning(f"Erro ao processar imagem {file.filename} no batch: {prediction['error']}")
            continue # Pula esta imagem se der erro
            
        results.append(prediction)

    if not results:
        logger.error("Nenhuma imagem no batch pôde ser processada.")
        raise HTTPException(status_code=400, detail="Nenhuma imagem pôde ser processada.")
    
    # Encontra o resultado com o maior "gravity_score"
    final_result = max(results, key=lambda res: res["gravity_score"])

    # 3. Loga o veredito final
    logger.info(f"Veredito final do batch (mais grave): {final_result['diagnosis']}")

    return final_result