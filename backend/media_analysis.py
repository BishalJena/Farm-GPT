"""
Media Analysis Service for Agricultural Image and Document Processing
Simple implementation for hackathon demo using LlamaVisionService
"""

import base64
import io
import json
import logging
from typing import Dict, List, Optional, Any, Literal
from datetime import datetime, timezone
from PIL import Image
import httpx
from pydantic import BaseModel
import os
from llama_vision_service import LlamaVisionService

logger = logging.getLogger(__name__)

class MediaAnalysis(BaseModel):
    id: str
    user_id: str
    file_name: str
    file_type: str
    analysis_type: Literal['pest', 'soil', 'crop_health', 'document']
    diagnosis: str
    confidence_score: float
    severity: Literal['low', 'medium', 'high']
    treatment: str
    cost_estimate: float
    nearby_dealers: List[Dict[str, Any]]
    created_at: datetime

class MediaAnalysisService:
    def __init__(self, openrouter_api_key: str):
        self.api_key = openrouter_api_key
        self.llama_vision = LlamaVisionService(openrouter_api_key)
        
        # Supported formats
        self.supported_image_formats = {'jpeg', 'jpg', 'png', 'webp', 'heic'}
        self.supported_document_formats = {'pdf'}
        
        # File size limits (in bytes)
        self.max_image_size = 10 * 1024 * 1024  # 10MB
        self.max_document_size = 5 * 1024 * 1024  # 5MB

    def validate_file(self, file_data: bytes, filename: str) -> Dict[str, Any]:
        """Validate uploaded file format and size"""
        file_extension = filename.lower().split('.')[-1]
        file_size = len(file_data)
        
        # Check format
        if file_extension in self.supported_image_formats:
            file_type = 'image'
            max_size = self.max_image_size
        elif file_extension in self.supported_document_formats:
            file_type = 'document'
            max_size = self.max_document_size
        else:
            return {
                'valid': False,
                'error': f'Unsupported format. Supported: {", ".join(self.supported_image_formats | self.supported_document_formats)}'
            }
        
        # Check size
        if file_size > max_size:
            return {
                'valid': False,
                'error': f'File too large. Max size: {max_size // (1024*1024)}MB'
            }
        
        return {
            'valid': True,
            'file_type': file_type,
            'file_extension': file_extension,
            'file_size': file_size
        }

    def compress_image(self, image_data: bytes) -> bytes:
        """Simple image compression for large files"""
        try:
            image = Image.open(io.BytesIO(image_data))
            
            # Resize if too large
            max_dimension = 1024
            if max(image.size) > max_dimension:
                ratio = max_dimension / max(image.size)
                new_size = tuple(int(dim * ratio) for dim in image.size)
                image = image.resize(new_size, Image.Resampling.LANCZOS)
            
            # Compress
            output = io.BytesIO()
            image.save(output, format='JPEG', quality=85, optimize=True)
            return output.getvalue()
        except Exception as e:
            logger.error(f"Image compression failed: {e}")
            return image_data

    async def analyze_image(self, image_data: bytes, filename: str, user_id: str) -> MediaAnalysis:
        """Analyze agricultural image using LlamaVisionService"""
        try:
            # Compress if needed
            if len(image_data) > 2 * 1024 * 1024:  # 2MB threshold
                image_data = self.compress_image(image_data)
            
            # Use LlamaVisionService for analysis
            analysis_data = await self.llama_vision.analyze_agricultural_image(image_data)
            
            # Add mock dealer data for hackathon demo
            mock_dealers = [
                {"name": "Punjab Agri Store", "contact": "+91-9876543210", "distance": "2.5 km", "products": ["Pesticides", "Fertilizers"]},
                {"name": "Farmer's Choice", "contact": "+91-9876543211", "distance": "4.1 km", "products": ["Organic Solutions", "Seeds"]},
                {"name": "Krishi Kendra", "contact": "+91-9876543212", "distance": "6.8 km", "products": ["Equipment", "Chemicals"]}
            ]
            
            # Extract cost estimate from string format
            cost_estimate = 0.0
            cost_str = analysis_data.get('cost_estimate', '₹200-500 per acre')
            try:
                # Extract first number from cost string
                import re
                numbers = re.findall(r'\d+', cost_str)
                if numbers:
                    cost_estimate = float(numbers[0])
            except:
                cost_estimate = 250.0  # Default fallback
            
            return MediaAnalysis(
                id=f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                user_id=user_id,
                file_name=filename,
                file_type='image',
                analysis_type=analysis_data.get('analysis_type', 'crop_health'),
                diagnosis=analysis_data.get('diagnosis', 'Analysis completed'),
                confidence_score=analysis_data.get('confidence_score', 0.7),
                severity=analysis_data.get('severity', 'medium'),
                treatment=analysis_data.get('treatment', 'Consult agricultural expert'),
                cost_estimate=cost_estimate,
                nearby_dealers=mock_dealers,
                created_at=datetime.now(timezone.utc)
            )

        except Exception as e:
            logger.error(f"Image analysis failed: {e}")
            # Return a fallback analysis
            return MediaAnalysis(
                id=f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                user_id=user_id,
                file_name=filename,
                file_type='image',
                analysis_type='crop_health',
                diagnosis=f"Unable to complete full analysis: {str(e)}. Please ensure image is clear and shows agricultural content.",
                confidence_score=0.5,
                severity='medium',
                treatment="Please consult with a local agricultural expert or try uploading a clearer image.",
                cost_estimate=0.0,
                nearby_dealers=[],
                created_at=datetime.now(timezone.utc)
            )

    async def analyze_document(self, document_data: bytes, filename: str, user_id: str) -> MediaAnalysis:
        """Simple document analysis - for demo purposes"""
        try:
            # For hackathon demo, we'll simulate document analysis
            # In production, this would use OCR and document parsing
            
            return MediaAnalysis(
                id=f"doc_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                user_id=user_id,
                file_name=filename,
                file_type='document',
                analysis_type='soil',
                diagnosis="Document uploaded successfully. Soil report analysis shows moderate NPK levels.",
                confidence_score=0.8,
                severity='medium',
                treatment="Based on soil report: Apply balanced NPK fertilizer (10:26:26) at 200kg/hectare",
                cost_estimate=250.0,
                nearby_dealers=[
                    {"name": "Fertilizer Hub", "contact": "+91-9876543210", "distance": "1.8 km", "products": ["NPK Fertilizer", "Soil Amendments"]},
                    {"name": "Agri Solutions", "contact": "+91-9876543211", "distance": "3.2 km", "products": ["Organic Fertilizer", "Soil Testing"]}
                ],
                created_at=datetime.now(timezone.utc)
            )

        except Exception as e:
            logger.error(f"Document analysis failed: {e}")
            return MediaAnalysis(
                id=f"doc_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                user_id=user_id,
                file_name=filename,
                file_type='document',
                analysis_type='document',
                diagnosis="Document received but analysis is currently unavailable.",
                confidence_score=0.3,
                severity='low',
                treatment="Please try again later or contact support.",
                cost_estimate=0.0,
                nearby_dealers=[],
                created_at=datetime.now(timezone.utc)
            )