import { Component } from '@angular/core';
import { VideosService } from '../../services/videos.service';
import { VideoIncidentComponent } from '../video-incident/video-incident.component';

@Component({
  selector: 'app-video-incidents-list',
  standalone: true,
  imports: [VideoIncidentComponent],
  templateUrl: './video-incidents-list.component.html',
  styleUrl: './video-incidents-list.component.scss'
})
export class VideoIncidentsListComponent {
  constructor(private videosService: VideosService) {}

  getVideIncidentsList() {
    return this.videosService.videoIncidents();
  }
}
