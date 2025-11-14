# Image Caption Generator Mini Project

This is a Django-based web application that generates captions for images using deep learning techniques. The project utilizes a pre-trained model to analyze images and produce descriptive captions.

## Features

- Upload images and generate captions automatically
- Web-based interface for easy interaction
- Uses deep learning models for accurate caption generation
- Built with Django framework

## Installation

1. Clone the repository:
   `
   git clone https://github.com/jasin263/IMAGE-CAPTION-GENERATOR--Deep-Learning.git
   cd IMAGE-CAPTION-GENERATOR--Deep-Learning
   `

2. Create a virtual environment:
   `
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   `

3. Install dependencies:
   `
   pip install -r requirements_train.txt
   `

4. Run migrations:
   `
   python manage.py migrate
   `

5. Start the development server:
   `
   python manage.py runserver
   `

6. Open your browser and navigate to http://127.0.0.1:8000/

## Usage

1. Upload an image using the web interface
2. The application will process the image and generate a caption
3. View the generated caption on the results page

## Project Structure

- captioner/ - Main Django app containing views, models, and templates
- Flickr8k_Dataset/ - Dataset used for training the model
- Flickr8k_text/ - Text annotations for the dataset
- image_caption_generator/ - Django project settings
- logs/ - Application logs
- media/ - Uploaded media files

## Technologies Used

- Django
- Python
- Deep Learning (TensorFlow/Keras)
- HTML/CSS/JavaScript

## Contributing

Feel free to fork this repository and submit pull requests for any improvements or bug fixes.

## License

This project is open source and available under the [MIT License](LICENSE).
