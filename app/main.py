import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from contextlib import asynccontextmanager
from typing import List
import logging

# Importa nossas funções do model_loader (mesmo diretório)
from . import model_loader

# Configure o logger
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Dicionário para guardar nosso modelo carregado
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carrega o modelo na inicialização e libera recursos ao desligar."""
    # Startup
    logger.info("🚀 Iniciando a aplicação...")
    logger.info("📦 Carregando modelo de IA...")
    
    model = model_loader.load_model()
    
    if model is None:
        logger.error("❌ ERRO CRÍTICO: Não foi possível carregar o modelo!")
        raise RuntimeError("Falha ao carregar o modelo de IA")
    
    ml_models["retinopathy_model"] = model
    logger.info("✅ Modelo carregado com sucesso!")
    logger.info("🎯 API pronta para receber requisições")
    
    yield
    
    # Shutdown
    logger.info("🔄 Desligando a aplicação...")
    ml_models.clear()
    logger.info("✅ Aplicação encerrada")

# Inicializa o app FastAPI com o 'lifespan'
app = FastAPI(
    title="VisusAI - API de Detecção de Retinopatia Diabética",
    description="API para classificação de retinopatia diabética usando EfficientNet-B4",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/")
def read_root():
    """Endpoint raiz - Informações da API."""
    return {
        "status": "online",
        "message": "Serviço de IA da VisusAI está online",
        "model_loaded": "retinopathy_model" in ml_models and ml_models["retinopathy_model"] is not None,
        "endpoints": {
            "predict": "/predict - POST - Upload de uma imagem",
            "predict_batch": "/predict_batch - POST - Upload de múltiplas imagens"
        }
    }

@app.get("/health")
def health_check():
    """Verifica o status da API."""
    model_loaded = "retinopathy_model" in ml_models and ml_models["retinopathy_model"] is not None
    
    return {
        "status": "healthy" if model_loaded else "unhealthy",
        "model_loaded": model_loaded
    }

@app.post("/predict")
async def create_upload_file(file: UploadFile = File(...)):
    """
    Endpoint principal. Recebe uma imagem e retorna a predição.
    
    Args:
        file: Arquivo de imagem (PNG, JPG, JPEG)
        
    Returns:
        JSON com diagnóstico, confiança e probabilidades
    """
    # Valida se o modelo está carregado
    if not ml_models.get("retinopathy_model"):
        logger.error("❌ Tentativa de predição sem modelo carregado")
        raise HTTPException(
            status_code=503, 
            detail="Modelo de IA não foi carregado. Reinicie o servidor."
        )

    # Valida o tipo de arquivo
    if not file.content_type or not file.content_type.startswith('image/'):
        logger.warning(f"⚠️  Arquivo inválido recebido: {file.content_type}")
        raise HTTPException(
            status_code=400,
            detail=f"Tipo de arquivo inválido. Esperado imagem, recebido: {file.content_type}"
        )

    logger.info(f"📸 Recebida imagem para predição: {file.filename}")

    # Lê os bytes da imagem
    try:
        image_bytes = await file.read()
        
        # Validação de tamanho
        if len(image_bytes) > 10 * 1024 * 1024:  # 10MB
            raise HTTPException(status_code=400, detail="Imagem muito grande (máx: 10MB)")
        
        if len(image_bytes) < 1024:  # 1KB
            raise HTTPException(status_code=400, detail="Imagem muito pequena")
        
    except Exception as e:
        logger.error(f"❌ Erro ao ler arquivo: {e}")
        raise HTTPException(status_code=400, detail="Erro ao ler arquivo de imagem")
    
    # Faz a predição
    prediction = model_loader.predict_image(
        model=ml_models["retinopathy_model"],
        image_bytes=image_bytes
    )
    
    # Verifica se houve erro
    if "error" in prediction:
        logger.warning(f"⚠️  Erro ao processar {file.filename}: {prediction['error']}")
        raise HTTPException(status_code=400, detail=prediction["error"])

    # Loga o resultado
    logger.info(
        f"✅ Resultado: {prediction['diagnosis']} "
        f"(Confiança: {prediction['confidence']:.2f}%)"
    )
    
    return prediction

@app.post("/predict_batch")
async def create_upload_files(files: List[UploadFile] = File(...)):
    """
    Endpoint para múltiplas imagens. Retorna TODOS os resultados + resumo.
    
    Args:
        files: Lista de arquivos de imagem
        
    Returns:
        JSON com todos os resultados, mais grave e estatísticas
    """
    # Valida se o modelo está carregado
    if not ml_models.get("retinopathy_model"):
        logger.error("❌ Tentativa de predição batch sem modelo carregado")
        raise HTTPException(
            status_code=503,
            detail="Modelo de IA não foi carregado. Reinicie o servidor."
        )
    
    # Valida quantidade de arquivos
    if len(files) == 0:
        raise HTTPException(status_code=400, detail="Nenhum arquivo enviado")
    
    if len(files) > 20:
        raise HTTPException(
            status_code=400,
            detail=f"Muitos arquivos. Máximo: 20, enviados: {len(files)}"
        )

    logger.info(f"📦 Recebido batch de {len(files)} imagens")

    results = []
    errors = []
    
    for idx, file in enumerate(files):
        # Valida tipo de arquivo
        if not file.content_type or not file.content_type.startswith('image/'):
            logger.warning(f"⚠️  Arquivo {idx+1} inválido: {file.content_type}")
            errors.append({
                "filename": file.filename,
                "error": f"Tipo de arquivo inválido: {file.content_type}"
            })
            continue
        
        try:
            image_bytes = await file.read()
            
            # Validação de tamanho
            if len(image_bytes) > 10 * 1024 * 1024:
                errors.append({
                    "filename": file.filename,
                    "error": "Imagem muito grande (máx: 10MB)"
                })
                continue
            
            if len(image_bytes) < 1024:
                errors.append({
                    "filename": file.filename,
                    "error": "Imagem muito pequena"
                })
                continue
            
            prediction = model_loader.predict_image(
                model=ml_models["retinopathy_model"],
                image_bytes=image_bytes
            )
            
            if "error" in prediction:
                logger.warning(f"⚠️  Erro em {file.filename}: {prediction['error']}")
                errors.append({
                    "filename": file.filename,
                    "error": prediction["error"]
                })
                continue
            
            # Adiciona informações extras ao resultado
            prediction["filename"] = file.filename
            prediction["index"] = idx
            results.append(prediction)
            logger.info(f"✅ [{idx+1}/{len(files)}] {file.filename}: {prediction['diagnosis']}")
            
        except Exception as e:
            logger.error(f"❌ Erro ao processar {file.filename}: {e}")
            errors.append({
                "filename": file.filename,
                "error": str(e)
            })
            continue

    # Verifica se processou alguma imagem
    if not results:
        logger.error("❌ Nenhuma imagem do batch pôde ser processada")
        raise HTTPException(
            status_code=400,
            detail="Nenhuma imagem pôde ser processada"
        )
    
    # Encontra o resultado com maior gravidade
    most_severe = max(results, key=lambda res: res["gravity_score"])
    
    # Calcula distribuição dos diagnósticos
    diagnosis_counts = {}
    for result in results:
        diag = result["diagnosis"]
        diagnosis_counts[diag] = diagnosis_counts.get(diag, 0) + 1

    # Monta resposta completa
    response = {
        "summary": {
            "total_images": len(files),
            "successful": len(results),
            "failed": len(errors),
            "most_severe": {
                "diagnosis": most_severe["diagnosis"],
                "confidence": most_severe["confidence"],
                "gravity_score": most_severe["gravity_score"],
                "filename": most_severe["filename"]
            },
            "diagnosis_distribution": diagnosis_counts
        },
        "results": results,
        "errors": errors if errors else []
    }

    logger.info(
        f"🎯 Batch completo: {len(results)} sucessos, {len(errors)} erros. "
        f"Mais grave: {most_severe['diagnosis']} ({most_severe['filename']})"
    )

    return response


@app.post("/predict_batch/most_severe")
async def predict_batch_most_severe(files: List[UploadFile] = File(...)):
    """
    Endpoint alternativo. Retorna APENAS o diagnóstico MAIS GRAVE (comportamento original).
    Útil quando você quer apenas saber o pior caso.
    
    Args:
        files: Lista de arquivos de imagem
        
    Returns:
        JSON com apenas o diagnóstico mais grave
    """
    # Reutiliza a lógica do endpoint principal
    full_response = await create_upload_files(files)
    
    # Retorna apenas o mais grave
    most_severe_result = full_response["summary"]["most_severe"]
    
    # Busca o resultado completo correspondente
    for result in full_response["results"]:
        if result["filename"] == most_severe_result["filename"]:
            return result
    
    # Fallback (não deve acontecer)
    return most_severe_result


if __name__ == "__main__":
    # Para rodar diretamente: python -m app.main
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )