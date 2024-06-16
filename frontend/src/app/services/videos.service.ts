import { HttpClient } from '@angular/common/http';
import { Injectable, signal } from '@angular/core';
import { VideoIncidentData } from '../interfaces/video-incident-data';
import { catchError, of, retry } from 'rxjs';

export const baseUrl = 'http://34.116.207.218:8000/api/';
const fetchInterval = 60 * 1000;

@Injectable({
  providedIn: 'root'
})
export class VideosService {
  videoIncidents = signal<VideoIncidentData[]>([]);
  
  constructor(private httpClient: HttpClient) { 
    this.fetchVideoIncidents();
    setInterval(() => this.fetchVideoIncidents(), fetchInterval);
  }

  fetchVideoIncidents() {
    this.httpClient.get<VideoIncidentData[]>(baseUrl + `videos`)
      .pipe(
        retry(3),
        catchError((error) => {
          console.error('Error fetching video incidents:', error);
          return of([]);
        })
      )
      .subscribe((videoIncidents) => {
        console.log('API Response:', videoIncidents);
        this.videoIncidents.set(videoIncidents);
        console.log('set videoIncisdents', this.videoIncidents);
      });
  }
}
