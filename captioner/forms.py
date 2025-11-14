from django import forms

class ImageUploadForm(forms.Form):
    MODEL_CHOICES = [
        ('pretrained', 'Pre-trained BLIP Model (General Purpose)'),
        ('custom', 'Custom Trained Model (Domain Specific)'),
    ]

    image = forms.ImageField(label='Select an image', help_text='Upload an image to generate a caption.')

    model_type = forms.ChoiceField(
        choices=MODEL_CHOICES,
        initial='pretrained',
        label='Select Model Type',
        widget=forms.RadioSelect(attrs={'class': 'form-check-input'}),
        help_text='Choose between general-purpose pre-trained model or your custom trained model'
    )
