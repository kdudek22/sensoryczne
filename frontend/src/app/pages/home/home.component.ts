import { Component } from '@angular/core';
import { VideoIncidentsListComponent } from '../../components/video-incidents-list/video-incidents-list.component';
import { NavbarComponent } from '../../components/navbar/navbar.component';
import { OptionsComponent } from '../../components/options/options.component';

@Component({
  selector: 'app-home',
  standalone: true,
  imports: [VideoIncidentsListComponent, NavbarComponent, OptionsComponent],
  templateUrl: './home.component.html',
  styleUrl: './home.component.scss'
})
export class HomeComponent {

}
