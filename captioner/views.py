from django.shortcuts import render, redirect
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.conf import settings
from .forms import ImageUploadForm
from .utils import generate_caption, load_model
import os
import threading
import logging

logger = logging.getLogger('captioner')

# Global flag to track if model is loaded
model_loaded = False
model_loading_lock = threading.Lock()

def upload_image(request):
    global model_loaded

    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            image = request.FILES['image']
            model_type = form.cleaned_data['model_type']

            # Additional validation: check file size (max 10MB)
            if image.size > 10 * 1024 * 1024:
                logger.warning(f"Image upload rejected: file too large ({image.size} bytes)")
                return render(request, 'captioner/upload.html', {
                    'form': form,
                    'error': 'Image file is too large. Maximum size is 10MB.'
                })

            # Additional validation: check file type
            allowed_types = ['image/jpeg', 'image/png', 'image/gif', 'image/webp']
            if image.content_type not in allowed_types:
                logger.warning(f"Image upload rejected: invalid file type ({image.content_type})")
                return render(request, 'captioner/upload.html', {
                    'form': form,
                    'error': 'Invalid file type. Only JPEG, PNG, GIF, and WebP images are allowed.'
                })

            logger.info(f"Image upload started: {image.name}, size: {image.size} bytes, model: {model_type}")

            # Check if model is loaded, if not, load it
            with model_loading_lock:
                if not model_loaded:
                    try:
                        use_trained = (model_type == 'custom')
                        load_model(use_trained=use_trained)
                        model_loaded = True
                        logger.info(f"Model loaded successfully: {'custom' if use_trained else 'pretrained'}")
                    except Exception as e:
                        logger.error(f"Failed to load AI model: {str(e)}")
                        return render(request, 'captioner/upload.html', {
                            'form': form,
                            'error': f'Failed to load AI model: {str(e)}. Please try again later.'
                        })

            # Save the uploaded image temporarily
            file_name = default_storage.save('temp/' + image.name, ContentFile(image.read()))
            file_path = os.path.join(settings.MEDIA_ROOT, file_name)

            try:
                # Generate caption with selected model
                use_trained = (model_type == 'custom')
                caption = generate_caption(file_path, use_trained=use_trained)
                logger.info(f"Caption generated successfully: '{caption}' for image {image.name}")
            except Exception as e:
                logger.error(f"Failed to generate caption for {image.name}: {str(e)}")
                # Clean up the temporary file
                if os.path.exists(file_path):
                    os.remove(file_path)
                return render(request, 'captioner/upload.html', {
                    'form': form,
                    'error': f'Failed to generate caption: {str(e)}'
                })

            # Clean up the temporary file
            if os.path.exists(file_path):
                os.remove(file_path)

            # Store image data in session for display
            import base64
            from io import BytesIO

            # Read image data for display
            image.seek(0)
            image_data = image.read()
            image_b64 = base64.b64encode(image_data).decode('utf-8')
            image_url = f"data:{image.content_type};base64,{image_b64}"

            logger.info(f"Image upload and caption generation completed for {image.name}")
            return render(request, 'captioner/result.html', {
                'image_url': image_url,
                'caption': caption,
                'model_type': model_type
            })
    else:
        form = ImageUploadForm()
    return render(request, 'captioner/upload.html', {'form': form})
