"""
Multi-modal AI Service

This service integrates vision, audio, and text processing capabilities:
- Vision processing using computer vision models
- Audio processing for speech recognition and synthesis
- Text processing with NLP models
- Multi-modal fusion and understanding
- Cross-modal retrieval and generation
- Integration with existing agent workflows
"""

import asyncio
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union, BinaryIO
from dataclasses import dataclass, asdict
from enum import Enum
import json
import numpy as np
import base64
import io
from pathlib import Path
import tempfile

# Vision processing
import cv2
from PIL import Image
import torch
import torchvision.transforms as transforms
from transformers import (
    CLIPProcessor, CLIPModel,
    BlipProcessor, BlipForConditionalGeneration,
    ViTImageProcessor, ViTForImageClassification
)

# Audio processing
import librosa
import soundfile as sf
import speech_recognition as sr
from transformers import (
    Wav2Vec2Processor, Wav2Vec2ForCTC,
    SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
)

# Text processing
from transformers import (
    AutoTokenizer, AutoModel,
    pipeline as hf_pipeline
)

# Multi-modal models
from transformers import (
    LlavaNextProcessor, LlavaNextForConditionalGeneration,
    GPT4VisionConfig
)

# Q Platform imports
from shared.pulsar_client import shared_pulsar_client
from shared.q_vectorstore_client.client import VectorStoreClient
from shared.q_knowledgegraph_client.client import KnowledgeGraphClient
from shared.vault_client import VaultClient

logger = logging.getLogger(__name__)

class ModalityType(Enum):
    """Types of modalities"""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    MULTIMODAL = "multimodal"

class ProcessingTask(Enum):
    """Processing task types"""
    CLASSIFICATION = "classification"
    DETECTION = "detection"
    SEGMENTATION = "segmentation"
    CAPTIONING = "captioning"
    TRANSCRIPTION = "transcription"
    SYNTHESIS = "synthesis"
    TRANSLATION = "translation"
    FUSION = "fusion"
    RETRIEVAL = "retrieval"
    GENERATION = "generation"

class ProcessingStatus(Enum):
    """Processing status"""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class MultiModalRequest:
    """Multi-modal processing request"""
    request_id: str
    agent_id: str
    modality: ModalityType
    task: ProcessingTask
    input_data: Dict[str, Any]
    parameters: Dict[str, Any]
    status: ProcessingStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None

@dataclass
class MultiModalAsset:
    """Multi-modal asset storage"""
    asset_id: str
    modality: ModalityType
    content_type: str
    file_path: str
    metadata: Dict[str, Any]
    embeddings: Optional[np.ndarray] = None
    created_at: datetime = None
    tags: List[str] = None

@dataclass
class CrossModalMapping:
    """Cross-modal relationship mapping"""
    mapping_id: str
    source_modality: ModalityType
    target_modality: ModalityType
    source_asset_id: str
    target_asset_id: str
    similarity_score: float
    mapping_type: str
    created_at: datetime

class MultiModalAIService:
    """
    Multi-modal AI Service for Q Platform
    """
    
    def __init__(self, 
                 storage_path: str = "data/multimodal",
                 models_path: str = "models/multimodal"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.models_path = Path(models_path)
        self.models_path.mkdir(parents=True, exist_ok=True)
        
        # Processing queues
        self.processing_queue: Dict[str, MultiModalRequest] = {}
        self.completed_requests: Dict[str, MultiModalRequest] = {}
        
        # Asset storage
        self.multimodal_assets: Dict[str, MultiModalAsset] = {}
        self.cross_modal_mappings: Dict[str, CrossModalMapping] = {}
        
        # Models
        self.models = {}
        self.processors = {}
        
        # Vector store client for embeddings
        self.vector_store_client = None
        
        # Configuration
        self.config = {
            "max_image_size": (1024, 1024),
            "max_audio_duration": 300,  # 5 minutes
            "supported_image_formats": ['.jpg', '.jpeg', '.png', '.bmp', '.tiff'],
            "supported_audio_formats": ['.wav', '.mp3', '.flac', '.ogg'],
            "batch_size": 8,
            "embedding_dim": 512
        }
        
        # Performance tracking
        self.multimodal_metrics = {
            "requests_processed": 0,
            "images_processed": 0,
            "audio_processed": 0,
            "text_processed": 0,
            "multimodal_fusions": 0,
            "cross_modal_retrievals": 0,
            "processing_time": 0.0
        }
        
        # Background tasks
        self.background_tasks: Set[asyncio.Task] = set()
        
    async def initialize(self):
        """Initialize the multi-modal AI service"""
        logger.info("Initializing Multi-modal AI Service")
        
        # Initialize models
        await self._initialize_models()
        
        # Setup vector store client
        await self._setup_vector_store()
        
        # Setup Pulsar topics
        await self._setup_pulsar_topics()
        
        # Load existing assets
        await self._load_multimodal_assets()
        
        # Start background tasks
        self.background_tasks.add(asyncio.create_task(self._processing_loop()))
        self.background_tasks.add(asyncio.create_task(self._performance_monitoring()))
        
        logger.info("Multi-modal AI Service initialized successfully")
    
    async def shutdown(self):
        """Shutdown the multi-modal AI service"""
        logger.info("Shutting down Multi-modal AI Service")
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self.background_tasks:
            await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        # Save assets
        await self._save_multimodal_assets()
        
        logger.info("Multi-modal AI Service shut down successfully")
    
    # ===== MODEL INITIALIZATION =====
    
    async def _initialize_models(self):
        """Initialize all multi-modal models"""
        
        try:
            # CLIP for vision-language understanding
            self.processors["clip"] = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.models["clip"] = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            
            # BLIP for image captioning
            self.processors["blip"] = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            self.models["blip"] = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
            
            # ViT for image classification
            self.processors["vit"] = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
            self.models["vit"] = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
            
            # Wav2Vec2 for speech recognition
            self.processors["wav2vec2"] = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
            self.models["wav2vec2"] = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
            
            # SpeechT5 for text-to-speech
            self.processors["speech_t5"] = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
            self.models["speech_t5"] = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
            self.models["speech_t5_vocoder"] = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
            
            # Text processing pipelines
            self.models["sentiment"] = hf_pipeline("sentiment-analysis")
            self.models["summarization"] = hf_pipeline("summarization")
            self.models["translation"] = hf_pipeline("translation_en_to_es")
            
            # Move models to GPU if available
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            for model_name, model in self.models.items():
                if hasattr(model, 'to'):
                    self.models[model_name] = model.to(device)
            
            logger.info("Multi-modal models initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize models: {e}")
            raise
    
    # ===== PROCESSING METHODS =====
    
    async def process_multimodal_request(
        self,
        agent_id: str,
        modality: ModalityType,
        task: ProcessingTask,
        input_data: Dict[str, Any],
        parameters: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Process multi-modal request
        
        Args:
            agent_id: ID of requesting agent
            modality: Type of modality
            task: Processing task
            input_data: Input data
            parameters: Additional parameters
            
        Returns:
            Request ID
        """
        request_id = f"multimodal_{uuid.uuid4().hex[:12]}"
        
        # Create request
        request = MultiModalRequest(
            request_id=request_id,
            agent_id=agent_id,
            modality=modality,
            task=task,
            input_data=input_data,
            parameters=parameters or {},
            status=ProcessingStatus.QUEUED,
            created_at=datetime.utcnow()
        )
        
        # Add to processing queue
        self.processing_queue[request_id] = request
        
        logger.info(f"Queued multi-modal request: {request_id}")
        
        return request_id
    
    async def _processing_loop(self):
        """Main processing loop for multi-modal requests"""
        
        while True:
            try:
                # Process queued requests
                for request_id, request in list(self.processing_queue.items()):
                    if request.status == ProcessingStatus.QUEUED:
                        await self._process_request(request)
                
                await asyncio.sleep(1)  # Small delay
                
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                await asyncio.sleep(5)
    
    async def _process_request(self, request: MultiModalRequest):
        """Process individual request"""
        
        try:
            request.status = ProcessingStatus.PROCESSING
            request.started_at = datetime.utcnow()
            
            logger.info(f"Processing request: {request.request_id}")
            
            # Route to appropriate processor
            if request.modality == ModalityType.IMAGE:
                result = await self._process_image_request(request)
            elif request.modality == ModalityType.AUDIO:
                result = await self._process_audio_request(request)
            elif request.modality == ModalityType.TEXT:
                result = await self._process_text_request(request)
            elif request.modality == ModalityType.MULTIMODAL:
                result = await self._process_multimodal_request(request)
            else:
                raise ValueError(f"Unsupported modality: {request.modality}")
            
            # Store result
            request.result = result
            request.status = ProcessingStatus.COMPLETED
            request.completed_at = datetime.utcnow()
            
            # Move to completed
            self.completed_requests[request.request_id] = request
            del self.processing_queue[request.request_id]
            
            # Update metrics
            self.multimodal_metrics["requests_processed"] += 1
            
            # Publish completion event
            await shared_pulsar_client.publish(
                "q.ml.multimodal.request.completed",
                {
                    "request_id": request.request_id,
                    "agent_id": request.agent_id,
                    "modality": request.modality.value,
                    "task": request.task.value,
                    "processing_time": (request.completed_at - request.started_at).total_seconds(),
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            
            logger.info(f"Completed request: {request.request_id}")
            
        except Exception as e:
            request.status = ProcessingStatus.FAILED
            request.error_message = str(e)
            request.completed_at = datetime.utcnow()
            
            # Move to completed
            self.completed_requests[request.request_id] = request
            del self.processing_queue[request.request_id]
            
            logger.error(f"Failed to process request {request.request_id}: {e}")
    
    # ===== IMAGE PROCESSING =====
    
    async def _process_image_request(self, request: MultiModalRequest) -> Dict[str, Any]:
        """Process image request"""
        
        if request.task == ProcessingTask.CLASSIFICATION:
            return await self._classify_image(request.input_data, request.parameters)
        elif request.task == ProcessingTask.CAPTIONING:
            return await self._caption_image(request.input_data, request.parameters)
        elif request.task == ProcessingTask.DETECTION:
            return await self._detect_objects(request.input_data, request.parameters)
        else:
            raise ValueError(f"Unsupported image task: {request.task}")
    
    async def _classify_image(self, input_data: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Classify image using ViT"""
        
        # Load image
        image = await self._load_image(input_data)
        
        # Process with ViT
        inputs = self.processors["vit"](image, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.models["vit"](**inputs)
            predictions = outputs.logits
            predicted_class_idx = predictions.argmax(-1).item()
            confidence = torch.nn.functional.softmax(predictions, dim=-1).max().item()
        
        # Get class name
        class_name = self.models["vit"].config.id2label[predicted_class_idx]
        
        # Update metrics
        self.multimodal_metrics["images_processed"] += 1
        
        return {
            "task": "classification",
            "predicted_class": class_name,
            "confidence": confidence,
            "class_id": predicted_class_idx
        }
    
    async def _caption_image(self, input_data: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Generate image caption using BLIP"""
        
        # Load image
        image = await self._load_image(input_data)
        
        # Process with BLIP
        inputs = self.processors["blip"](image, return_tensors="pt")
        
        with torch.no_grad():
            out = self.models["blip"].generate(**inputs, max_length=100)
            caption = self.processors["blip"].decode(out[0], skip_special_tokens=True)
        
        # Update metrics
        self.multimodal_metrics["images_processed"] += 1
        
        return {
            "task": "captioning",
            "caption": caption,
            "confidence": 0.8  # Placeholder
        }
    
    async def _detect_objects(self, input_data: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Detect objects in image"""
        
        # Load image
        image = await self._load_image(input_data)
        
        # Simple object detection using OpenCV (placeholder)
        # In production, would use YOLO or similar
        image_np = np.array(image)
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        
        # Detect edges as a simple form of object detection
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Extract bounding boxes
        objects = []
        for contour in contours[:10]:  # Limit to 10 objects
            x, y, w, h = cv2.boundingRect(contour)
            if w > 20 and h > 20:  # Filter small objects
                objects.append({
                    "bbox": [x, y, x+w, y+h],
                    "confidence": 0.7,  # Placeholder
                    "class": "object"  # Placeholder
                })
        
        # Update metrics
        self.multimodal_metrics["images_processed"] += 1
        
        return {
            "task": "detection",
            "objects": objects,
            "count": len(objects)
        }
    
    async def _load_image(self, input_data: Dict[str, Any]) -> Image.Image:
        """Load image from various sources"""
        
        if "base64" in input_data:
            # Decode base64 image
            image_data = base64.b64decode(input_data["base64"])
            image = Image.open(io.BytesIO(image_data))
        elif "file_path" in input_data:
            # Load from file
            image = Image.open(input_data["file_path"])
        elif "url" in input_data:
            # Load from URL (placeholder)
            raise NotImplementedError("URL loading not implemented")
        else:
            raise ValueError("No valid image source provided")
        
        # Resize if needed
        if image.size[0] > self.config["max_image_size"][0] or image.size[1] > self.config["max_image_size"][1]:
            image = image.resize(self.config["max_image_size"], Image.Resampling.LANCZOS)
        
        # Convert to RGB
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        return image
    
    # ===== AUDIO PROCESSING =====
    
    async def _process_audio_request(self, request: MultiModalRequest) -> Dict[str, Any]:
        """Process audio request"""
        
        if request.task == ProcessingTask.TRANSCRIPTION:
            return await self._transcribe_audio(request.input_data, request.parameters)
        elif request.task == ProcessingTask.SYNTHESIS:
            return await self._synthesize_speech(request.input_data, request.parameters)
        else:
            raise ValueError(f"Unsupported audio task: {request.task}")
    
    async def _transcribe_audio(self, input_data: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Transcribe audio using Wav2Vec2"""
        
        # Load audio
        audio_array, sample_rate = await self._load_audio(input_data)
        
        # Resample if needed
        if sample_rate != 16000:
            audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=16000)
            sample_rate = 16000
        
        # Process with Wav2Vec2
        inputs = self.processors["wav2vec2"](audio_array, sampling_rate=sample_rate, return_tensors="pt")
        
        with torch.no_grad():
            logits = self.models["wav2vec2"](inputs.input_values).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = self.processors["wav2vec2"].decode(predicted_ids[0])
        
        # Update metrics
        self.multimodal_metrics["audio_processed"] += 1
        
        return {
            "task": "transcription",
            "transcription": transcription,
            "confidence": 0.8,  # Placeholder
            "duration": len(audio_array) / sample_rate
        }
    
    async def _synthesize_speech(self, input_data: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize speech from text using SpeechT5"""
        
        text = input_data.get("text", "")
        
        # Process with SpeechT5
        inputs = self.processors["speech_t5"](text=text, return_tensors="pt")
        
        # Generate speech
        with torch.no_grad():
            # Create speaker embeddings (placeholder)
            speaker_embeddings = torch.randn(1, 512)
            
            spectrogram = self.models["speech_t5"].generate_speech(
                inputs["input_ids"], 
                speaker_embeddings, 
                vocoder=self.models["speech_t5_vocoder"]
            )
        
        # Convert to audio array
        audio_array = spectrogram.cpu().numpy()
        
        # Save audio file
        output_path = self.storage_path / f"speech_{uuid.uuid4().hex[:8]}.wav"
        sf.write(output_path, audio_array, 16000)
        
        # Update metrics
        self.multimodal_metrics["audio_processed"] += 1
        
        return {
            "task": "synthesis",
            "audio_path": str(output_path),
            "duration": len(audio_array) / 16000,
            "sample_rate": 16000
        }
    
    async def _load_audio(self, input_data: Dict[str, Any]) -> Tuple[np.ndarray, int]:
        """Load audio from various sources"""
        
        if "base64" in input_data:
            # Decode base64 audio
            audio_data = base64.b64decode(input_data["base64"])
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                tmp_file.write(audio_data)
                tmp_file.flush()
                audio_array, sample_rate = librosa.load(tmp_file.name, sr=None)
        elif "file_path" in input_data:
            # Load from file
            audio_array, sample_rate = librosa.load(input_data["file_path"], sr=None)
        else:
            raise ValueError("No valid audio source provided")
        
        # Limit duration
        max_samples = self.config["max_audio_duration"] * sample_rate
        if len(audio_array) > max_samples:
            audio_array = audio_array[:max_samples]
        
        return audio_array, sample_rate
    
    # ===== TEXT PROCESSING =====
    
    async def _process_text_request(self, request: MultiModalRequest) -> Dict[str, Any]:
        """Process text request"""
        
        if request.task == ProcessingTask.CLASSIFICATION:
            return await self._classify_text(request.input_data, request.parameters)
        elif request.task == ProcessingTask.TRANSLATION:
            return await self._translate_text(request.input_data, request.parameters)
        else:
            raise ValueError(f"Unsupported text task: {request.task}")
    
    async def _classify_text(self, input_data: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Classify text sentiment"""
        
        text = input_data.get("text", "")
        
        # Process with sentiment analysis
        result = self.models["sentiment"](text)[0]
        
        # Update metrics
        self.multimodal_metrics["text_processed"] += 1
        
        return {
            "task": "classification",
            "sentiment": result["label"],
            "confidence": result["score"]
        }
    
    async def _translate_text(self, input_data: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Translate text"""
        
        text = input_data.get("text", "")
        
        # Process with translation
        result = self.models["translation"](text)[0]
        
        # Update metrics
        self.multimodal_metrics["text_processed"] += 1
        
        return {
            "task": "translation",
            "translated_text": result["translation_text"],
            "confidence": 0.8  # Placeholder
        }
    
    # ===== MULTIMODAL PROCESSING =====
    
    async def _process_multimodal_request(self, request: MultiModalRequest) -> Dict[str, Any]:
        """Process multimodal request"""
        
        if request.task == ProcessingTask.FUSION:
            return await self._multimodal_fusion(request.input_data, request.parameters)
        elif request.task == ProcessingTask.RETRIEVAL:
            return await self._cross_modal_retrieval(request.input_data, request.parameters)
        else:
            raise ValueError(f"Unsupported multimodal task: {request.task}")
    
    async def _multimodal_fusion(self, input_data: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Fuse multiple modalities using CLIP"""
        
        # Load image and text
        image = await self._load_image(input_data)
        text = input_data.get("text", "")
        
        # Process with CLIP
        inputs = self.processors["clip"](text=[text], images=[image], return_tensors="pt", padding=True)
        
        with torch.no_grad():
            outputs = self.models["clip"](**inputs)
            logits_per_image = outputs.logits_per_image
            logits_per_text = outputs.logits_per_text
            
            # Calculate similarity
            similarity = torch.cosine_similarity(
                outputs.image_embeds, 
                outputs.text_embeds, 
                dim=1
            ).item()
        
        # Update metrics
        self.multimodal_metrics["multimodal_fusions"] += 1
        
        return {
            "task": "fusion",
            "similarity": similarity,
            "image_embeddings": outputs.image_embeds.cpu().numpy().tolist(),
            "text_embeddings": outputs.text_embeds.cpu().numpy().tolist()
        }
    
    async def _cross_modal_retrieval(self, input_data: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Perform cross-modal retrieval"""
        
        query_modality = input_data.get("query_modality", "text")
        target_modality = input_data.get("target_modality", "image")
        
        # Generate query embedding
        if query_modality == "text":
            query_embedding = await self._get_text_embedding(input_data.get("text", ""))
        elif query_modality == "image":
            query_embedding = await self._get_image_embedding(input_data)
        else:
            raise ValueError(f"Unsupported query modality: {query_modality}")
        
        # Search in target modality
        results = await self._search_similar_assets(query_embedding, target_modality)
        
        # Update metrics
        self.multimodal_metrics["cross_modal_retrievals"] += 1
        
        return {
            "task": "retrieval",
            "query_modality": query_modality,
            "target_modality": target_modality,
            "results": results
        }
    
    async def _get_text_embedding(self, text: str) -> np.ndarray:
        """Get text embedding using CLIP"""
        
        inputs = self.processors["clip"](text=[text], return_tensors="pt", padding=True)
        
        with torch.no_grad():
            text_embeds = self.models["clip"].get_text_features(**inputs)
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        
        return text_embeds.cpu().numpy()
    
    async def _get_image_embedding(self, input_data: Dict[str, Any]) -> np.ndarray:
        """Get image embedding using CLIP"""
        
        image = await self._load_image(input_data)
        inputs = self.processors["clip"](images=[image], return_tensors="pt")
        
        with torch.no_grad():
            image_embeds = self.models["clip"].get_image_features(**inputs)
            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        
        return image_embeds.cpu().numpy()
    
    async def _search_similar_assets(self, query_embedding: np.ndarray, target_modality: str) -> List[Dict[str, Any]]:
        """Search for similar assets"""
        
        # Filter assets by modality
        target_assets = [
            asset for asset in self.multimodal_assets.values()
            if asset.modality.value == target_modality and asset.embeddings is not None
        ]
        
        if not target_assets:
            return []
        
        # Calculate similarities
        similarities = []
        for asset in target_assets:
            similarity = np.dot(query_embedding.flatten(), asset.embeddings.flatten())
            similarities.append({
                "asset_id": asset.asset_id,
                "similarity": similarity,
                "metadata": asset.metadata
            })
        
        # Sort by similarity
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        
        return similarities[:10]  # Return top 10
    
    # ===== ASSET MANAGEMENT =====
    
    async def store_multimodal_asset(
        self,
        modality: ModalityType,
        content_type: str,
        data: Union[str, bytes, Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
    ) -> str:
        """Store multimodal asset"""
        
        asset_id = f"asset_{uuid.uuid4().hex[:12]}"
        
        # Store data
        file_path = self.storage_path / f"{asset_id}.{content_type.split('/')[-1]}"
        
        if isinstance(data, str):
            with open(file_path, 'w') as f:
                f.write(data)
        elif isinstance(data, bytes):
            with open(file_path, 'wb') as f:
                f.write(data)
        else:
            with open(file_path, 'w') as f:
                json.dump(data, f)
        
        # Generate embeddings
        embeddings = await self._generate_embeddings(modality, file_path)
        
        # Create asset
        asset = MultiModalAsset(
            asset_id=asset_id,
            modality=modality,
            content_type=content_type,
            file_path=str(file_path),
            metadata=metadata or {},
            embeddings=embeddings,
            created_at=datetime.utcnow(),
            tags=tags or []
        )
        
        self.multimodal_assets[asset_id] = asset
        
        logger.info(f"Stored multimodal asset: {asset_id}")
        
        return asset_id
    
    async def _generate_embeddings(self, modality: ModalityType, file_path: Path) -> np.ndarray:
        """Generate embeddings for asset"""
        
        if modality == ModalityType.TEXT:
            with open(file_path, 'r') as f:
                text = f.read()
            return await self._get_text_embedding(text)
        elif modality == ModalityType.IMAGE:
            image = Image.open(file_path)
            return await self._get_image_embedding({"image": image})
        else:
            # Return random embeddings for other modalities
            return np.random.randn(512)
    
    # ===== UTILITY METHODS =====
    
    async def _setup_vector_store(self):
        """Setup vector store client"""
        
        # This would initialize the vector store client
        # For now, just log
        logger.info("Vector store client setup completed")
    
    async def _setup_pulsar_topics(self):
        """Setup Pulsar topics for multimodal AI"""
        
        topics = [
            "q.ml.multimodal.request.created",
            "q.ml.multimodal.request.completed",
            "q.ml.multimodal.asset.stored",
            "q.ml.multimodal.fusion.completed"
        ]
        
        logger.info("Multimodal AI Pulsar topics configured")
    
    async def _load_multimodal_assets(self):
        """Load existing multimodal assets"""
        
        assets_file = self.storage_path / "assets.json"
        if assets_file.exists():
            try:
                with open(assets_file, 'r') as f:
                    assets_data = json.load(f)
                
                for asset_data in assets_data:
                    asset = MultiModalAsset(**asset_data)
                    # Convert embeddings back to numpy array
                    if asset.embeddings is not None:
                        asset.embeddings = np.array(asset.embeddings)
                    self.multimodal_assets[asset.asset_id] = asset
                
                logger.info(f"Loaded {len(self.multimodal_assets)} multimodal assets")
            except Exception as e:
                logger.error(f"Failed to load multimodal assets: {e}")
    
    async def _save_multimodal_assets(self):
        """Save multimodal assets to storage"""
        
        assets_file = self.storage_path / "assets.json"
        try:
            assets_data = []
            for asset in self.multimodal_assets.values():
                asset_dict = asdict(asset)
                # Convert numpy array to list for JSON serialization
                if asset_dict["embeddings"] is not None:
                    asset_dict["embeddings"] = asset_dict["embeddings"].tolist()
                assets_data.append(asset_dict)
            
            with open(assets_file, 'w') as f:
                json.dump(assets_data, f, indent=2, default=str)
            
            logger.info(f"Saved {len(self.multimodal_assets)} multimodal assets")
        except Exception as e:
            logger.error(f"Failed to save multimodal assets: {e}")
    
    # ===== BACKGROUND TASKS =====
    
    async def _performance_monitoring(self):
        """Monitor multimodal AI performance"""
        
        while True:
            try:
                # Update metrics
                total_processing_time = sum(
                    (req.completed_at - req.started_at).total_seconds()
                    for req in self.completed_requests.values()
                    if req.started_at and req.completed_at
                )
                
                if self.completed_requests:
                    self.multimodal_metrics["processing_time"] = total_processing_time
                
                await asyncio.sleep(300)  # Update every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in performance monitoring: {e}")
                await asyncio.sleep(300)
    
    # ===== PUBLIC API METHODS =====
    
    async def get_request_status(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Get status of multimodal request"""
        
        # Check active queue
        if request_id in self.processing_queue:
            request = self.processing_queue[request_id]
        elif request_id in self.completed_requests:
            request = self.completed_requests[request_id]
        else:
            return None
        
        return {
            "request_id": request_id,
            "status": request.status.value,
            "modality": request.modality.value,
            "task": request.task.value,
            "created_at": request.created_at,
            "started_at": request.started_at,
            "completed_at": request.completed_at,
            "result": request.result,
            "error_message": request.error_message
        }
    
    async def get_request_result(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Get result of completed request"""
        
        if request_id in self.completed_requests:
            request = self.completed_requests[request_id]
            if request.status == ProcessingStatus.COMPLETED:
                return request.result
        
        return None
    
    async def list_multimodal_assets(self, modality: Optional[ModalityType] = None) -> List[Dict[str, Any]]:
        """List multimodal assets"""
        
        assets = list(self.multimodal_assets.values())
        
        if modality:
            assets = [asset for asset in assets if asset.modality == modality]
        
        return [
            {
                "asset_id": asset.asset_id,
                "modality": asset.modality.value,
                "content_type": asset.content_type,
                "metadata": asset.metadata,
                "tags": asset.tags,
                "created_at": asset.created_at
            }
            for asset in assets
        ]
    
    async def get_multimodal_metrics(self) -> Dict[str, Any]:
        """Get multimodal AI metrics"""
        
        return {
            "service_metrics": self.multimodal_metrics,
            "active_requests": len(self.processing_queue),
            "completed_requests": len(self.completed_requests),
            "stored_assets": len(self.multimodal_assets),
            "cross_modal_mappings": len(self.cross_modal_mappings)
        }

# Global instance
multimodal_ai_service = MultiModalAIService() 