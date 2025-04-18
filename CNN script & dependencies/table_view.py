import customtkinter 
from customtkinter import filedialog
from tkinter import ttk
import pandas as pd
import os
import librosa
import sounddevice as sd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np


class edit_tab(customtkinter.CTkFrame):
    def __init__(self, parent):
        super().__init__(parent)
        self.pack()

        self.excel_frame = customtkinter.CTkFrame(self)
        self.excel_frame.pack(padx=10, pady=10)
        self.excel_label = customtkinter.CTkLabel(self.excel_frame, text='Edit Excel Table')
        self.excel_label.pack(padx=10, pady=10)
        self.select_button = customtkinter.CTkButton(self.excel_frame, text='Select Table', command=self.file_select)
        self.select_button.pack(padx=10, pady=10)

        self.excel_file = ''

    def file_select(self):
        filetypes = (
            ('Excel Files', '*.xlsx'),
        )
        self.excel_file = filedialog.askopenfilename(initialdir='./', filetypes=filetypes)

        self.start_table()
        
    def start_table(self):
        self.table_view = table_window(self.excel_file)

class table_window(customtkinter.CTkToplevel):
    def __init__(self, excel_file, window_size = '500x400'):
        super().__init__()
        self.title('View Table')
        self.geometry(window_size)

        matplotlib.use('TkAgg')
        
        self.table_file = excel_file
        self.data_table = pd.read_excel(self.table_file)

        self.table = ttk.Treeview(self, columns=('time', 'true_positive'))
        self.table.heading('#0', text='ID')
        self.table.heading('time', text='Time')
        self.table.heading('true_positive', text='True Positive')
        
        self.fill_table()
        self.set_events()
        
        self.table.pack(side='top', fill='both', expand=True, padx=10, pady=10)

        self.input_frame = customtkinter.CTkFrame(self)
        self.input_frame.pack(side='bottom', padx=10, pady=10)

        self.view_button = customtkinter.CTkButton(self.input_frame, text='View', command=self.view)
        self.view_button.grid(column=0, row=0, padx=10, pady=10)

        self.listen_button = customtkinter.CTkButton(self.input_frame, text='Listen', command=self.listen)
        self.listen_button.grid(column=1, row=0, padx=10, pady=10)

        self.true_checkbox = customtkinter.CTkCheckBox(self.input_frame, text='T/F', command=self.checkbox_toggled)
        self.true_checkbox.grid(column=2, row=0, padx=10, pady=10)

        self.save_button = customtkinter.CTkButton(self.input_frame, text='Save', command=self.save)
        self.save_button.grid(column=2, row=1, padx=10, pady=10)

        self.selected_label = customtkinter.CTkLabel(self.input_frame, text='Selected:')
        self.selected_label.grid(column=0, row=1, padx=10, pady=10)

    #table headers
    #(columns=['label', 'date', 'time_stamp', 'start_offset', 'end_offset', 'power', 'distance', 'true_positive'])
    def fill_table(self):
        for i, row in self.data_table.iterrows():
            values = (row['time_stamp'], row['true_positive'])
            self.table.insert('', 'end', text=row['label'], values=values)

    def set_events(self):
        #event for a cell selection
        self.table.bind('<<TreeviewSelect>>', self.selected)

    #performed when a cell is selected
    def selected(self, event):
        #get the data
        data = self.table.item(self.table.selection()[0])
        self.selected_label.configure(text=f"Selected: {data['text']}")

        #update checkbox
        if data['values'][1] == 'T':
            self.true_checkbox.select()
        else:
            self.true_checkbox.deselect()

    #checkbox is toggled
    def checkbox_toggled(self):
        table_val = self.table.item(self.table.selection()[0])['values']

        #new state
        update_val = 'T'
        if self.true_checkbox.get() == 1:
            update_val = 'T'
        else:
            update_val = 'F'

        #update values
        self.table.item(self.table.selection()[0], values=(table_val[0], update_val))
        table_index = int(self.table.item(self.table.selection()[0])['text'])
        self.data_table.loc[table_index, 'true_positive'] = update_val

        #update text to reflect change
        if self.save_button.cget('text') == 'Saved!':
            self.save_button.configure(text='Save')

    #make spectrogram and display
    def view(self):
        self.view_button.configure(state='disabled')
        sample_rate=24000
        n_fft=1024
        hop_length=512

        idx = int(self.table.item(self.table.selection()[0])['text'])
        row = self.data_table.loc[idx]
        data, sr = librosa.load(row['file'], mono=True, offset=row['start_offset']-0.5, duration=1)
        mel_spec = librosa.feature.melspectrogram(y=data, sr=sr, n_fft=n_fft, hop_length=hop_length)
        mel_spec = librosa.power_to_db(mel_spec)
        mel_spec /= 80 * 255 #normalize and rerange to 0-255
        mel_spec = np.flip(mel_spec, axis=0)
        plt.imshow(mel_spec, cmap='viridis')
        plt.show()

        self.view_button.configure(state='normal')

    #play audio clip around prediction
    def listen(self):
        idx = int(self.table.item(self.table.selection()[0])['text'])
        row = self.data_table.loc[idx]
        data, sr = librosa.load(row['file'], mono=True, offset=row['start_offset']-0.5, duration=1)
        sd.play(data, sr)

    #save excel file
    def save(self):
        self.data_table.to_excel(self.table_file, header=True, index=False)
        self.save_button.configure(text='Saved!')