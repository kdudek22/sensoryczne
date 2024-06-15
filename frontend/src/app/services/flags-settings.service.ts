import { Injectable, signal } from '@angular/core';
import { FlagData } from '../interfaces/flag-data';
import { baseUrl } from './videos.service';
import { HttpClient } from '@angular/common/http';
import { catchError, of, retry } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class FlagsSettingsService {
  flags = signal<FlagData[]>([]);

  constructor(private httpClient: HttpClient) { 
    this.fetchClassesFlags()
  }

  fetchClassesFlags() {
    this.httpClient.get<FlagData[]>(baseUrl + `classes`)
      .pipe(
        retry(3),
        catchError((error) => {
          console.error('Error fetching classes flags:', error);
          return of([]);
        })
      )
      .subscribe((videoIncidents) => {
        console.log('API Response:', videoIncidents);
        this.flags.set(videoIncidents);
        console.log('set classesFlags', this.flags);
      });
  }

  updateClassesFlags(flags: FlagData[]) {
    this.httpClient.post(baseUrl + `classes`, flags)
      .pipe(
        retry(3),
        catchError((error) => {
          console.error('Error updating classes flags:', error);
          return of([]);
        })
        
      );
  }
}
