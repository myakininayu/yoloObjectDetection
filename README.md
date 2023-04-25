# Object detection using Yolo model and OpenCV

## Installation (venv)

```bash
pip install -r requirements.txt
```

## Usage

```bash
# expected 3 parameters
# python main.py filepath class
python main.py 1.mp4 cat 
```
In console you'll see logs (whether video is detected and objects are found).
After detecting you'll see a new videofile consisting of cropped images (output_filename.mp4 = input_filename + object_class.mp4)

## Examples
<img width="445" alt="image" src="https://user-images.githubusercontent.com/79317792/234410240-7c9a1b8e-4b6b-4f44-a301-423d3fbc936b.png"> <img width="485" alt="image" src="https://user-images.githubusercontent.com/79317792/234410938-363a1285-4423-41c7-b250-a95aff9465ba.png">



