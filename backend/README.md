# This is the backend part of the app

### How to add a video

Currently, there are no endpoints for posting data, you have to do it using the admin panel.
To do so, run ```python manage.py createsuperuser``` in the folder with the manage.py file.
Then you have to provide some account details(you do not have to pass a real email). After you have created the account
open the url ```127.0.0.1:8000/admin``` and login into your account. Then open the videos tab, and create a new instance
by filling the form, and pressing save.

## Endpoints
The base url for the api is 127.0.0.1:8000/api

### GET /videos
Response is a list of videos with attributes:
- id - the id of the record
- date - timestamp of when the recording was saved
- detection - the label of the image, right now it`s a string, later we will have to thin about multiple detections in a
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

### POST /videos
This post, saves a video with a detection to the database
Body:
```json
{
    "detection": "cat",
    "video": "file.mp4"
}
```
