import { Component, input } from '@angular/core';
import { VideoIncidentData } from '../../interfaces/video-incident-data';

@Component({
  selector: 'app-video-incident',
  standalone: true,
  imports: [],
  templateUrl: './video-incident.component.html',
  styleUrl: './video-incident.component.scss'
})
export class VideoIncidentComponent {
  videoIncidentData = input<VideoIncidentData>({} as VideoIncidentData);

  constructor() {
  }

  convertDate(date: string) {
    return new Date(date).toLocaleDateString("pl-PL",  {
      year: 'numeric',
      month: 'numeric',
      day: 'numeric',
  });
  }

  convertTime(date: string) {
    return new Date(date).toLocaleTimeString("pl-PL",  {
      hour: 'numeric',
      minute: 'numeric',
      second: 'numeric'
  });
  }
}
