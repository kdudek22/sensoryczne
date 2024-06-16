import { Injectable, signal } from '@angular/core';
import { FlagData } from '../interfaces/flag-data';
import { baseUrl } from './videos.service';
import { HttpClient, HttpHeaders, HttpParams } from '@angular/common/http';
import { catchError, of, retry, tap } from 'rxjs';

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
    console.log('Updating classes flags:', flags);
    const headers = new HttpHeaders({ 'Content-Type': 'application/json' });
    console.log('trying to update classes flags:', flags, headers, baseUrl + 'classes')
    this.httpClient.post(baseUrl + `classes/`, flags, { headers })
      .pipe(
        tap({
          next: (response) => {
            console.log('HTTP POST Request successful:', {
              url: baseUrl + 'classes/',
              payload: flags,
              response,
            });
          },
          error: (error) => {
            console.error('HTTP POST Request failed:', {
              url: baseUrl + 'classes/',
              payload: flags,
              error,
            });
          },
        }),
        retry(3),
        catchError((error) => {
          console.error('Error updating classes flags:', error);
          return of([]);
        })
        
      ).subscribe();
  }
}
