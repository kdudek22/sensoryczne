import { Component } from '@angular/core';
import { FormGroup, FormBuilder, FormArray } from '@angular/forms';
import { FlagsSettingsService } from '../../services/flags-settings.service';
import { NgFor } from '@angular/common';

@Component({
  selector: 'app-options',
  standalone: true,
  imports: [NgFor],
  templateUrl: './options.component.html',
  styleUrl: './options.component.scss'
})
export class OptionsComponent {
  flagsForm: FormGroup;

  constructor(private fb: FormBuilder, private flagSettingsService: FlagsSettingsService) {
    this.flagsForm = this.fb.group({
      flags: this.fb.array([]),
    });
    const keyValueFlags = flagSettingsService.flags().map(flag => ({ name: flag.name, value: flag.is_active }));
    const flagsFormArray = this.fb.array(keyValueFlags);

    this.flagsForm.setControl('flags', flagsFormArray);
  }

  setFlags(flags: any[]): void {
    const flagsFGs = flags.map(flag => this.fb.group({
      name: [flag.name],
      value: [flag.value]
    }));
    const flagsFormArray = this.fb.array(flagsFGs);
    this.flagsForm.setControl('flags', flagsFormArray);
  }

  get flags(): FormArray {
    return this.flagsForm.get('flags') as FormArray;
  }

  onSubmit(): void {
    if (this.flagsForm.valid) {
      this.flagSettingsService.updateClassesFlags(this.flagsForm.value.flags);
    }
  }
}