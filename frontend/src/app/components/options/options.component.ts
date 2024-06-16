import { Component } from '@angular/core';
import { FlagsSettingsService } from '../../services/flags-settings.service';
import { NgFor } from '@angular/common';
import { FlagData } from '../../interfaces/flag-data';
import {MatCheckboxChange, MatCheckboxModule} from '@angular/material/checkbox';

@Component({
  selector: 'app-options',
  standalone: true,
  imports: [NgFor, MatCheckboxModule],
  templateUrl: './options.component.html',
  styleUrl: './options.component.scss'
})
export class OptionsComponent {

  constructor(public flagSettingsService: FlagsSettingsService) {}

  onSubmit(event: MatCheckboxChange, flagData: FlagData): void {
    flagData.is_active = event.checked;
    console.log("Changing flag data: ", flagData);
    this.flagSettingsService.updateClassesFlags([flagData]);
  }
}