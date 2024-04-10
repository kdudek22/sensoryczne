# This is the backend part of the app

## Endpoints
The base url for the api is 127.0.0.1:8000/api

### GET /videos
Response is a list of videos with attributes:
- id - the id of the record
- date - timestamp of when the recording was saved
- detection - the label of the image, right now it`s a string, later we will have to thin about multpile detections in a
single recording etc.
- video_url - the url at which the video is available, currently its simply hosted on the server

Example response:

```json
[
    {
        "id": "1",
        "date": "2024-04-10 16:10:20.827212+00:00",
        "detection": "Sarna",
        "video_url": "http://127.0.0.1:8000/media/videos/car.mp4"
    }
]
```