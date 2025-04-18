import customtkinter 
from customtkinter import filedialog
from audio_model import shotgun_detector
from shotgun_dataset import shotgun_training_dataset, shotgun_inference_dataset
import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import librosa
import os
import threading
from sklearn.metrics import f1_score, precision_score, recall_score, jaccard_score
import matplotlib.pyplot as plt
from table_view import edit_tab

class shotgun_detector_app(customtkinter.CTk):
    def __init__(self, window_size = '400x400'):
        super().__init__()
        self.title('Sound Event Detector')
        self.geometry(window_size)

        #make tabs
        self.tabview = customtkinter.CTkTabview(self)
        self.tabview.pack(expand=True, fill='both', padx=10, pady=10)
        
        tab_1 = self.tabview.add('Predict')
        tab_2 = self.tabview.add('Edit')
        tab_3 = self.tabview.add('Train')

        self.prediction_tab = prediction_tab(tab_1)
        self.edit_tab = edit_tab(tab_2)
        self.train_tab = train_tab(tab_3)


class train_tab():
    def __init__(self, parent):

        self.top_frame = customtkinter.CTkFrame(parent)
        self.top_frame.pack()
        
        #dataset
        self.data_frame = customtkinter.CTkFrame(self.top_frame)
        self.data_frame.pack(side='left', padx=10, pady=10, expand=True)
        self.dataset_label = customtkinter.CTkLabel(self.data_frame, text='Dataset')
        self.dataset_label.pack(padx=10, pady=10)
        self.dataset_button = customtkinter.CTkButton(self.data_frame, text='Select Dataset', command=self.dataset_select)
        self.dataset_button.pack(padx=10, pady=10)

        self.model_frame = customtkinter.CTkFrame(self.top_frame)
        self.model_frame.pack(side='left', padx=10, pady=10, expand=True)
        self.model_frame_label = customtkinter.CTkLabel(self.model_frame, text='Training Parameters')
        self.model_frame_label.pack(padx=10, pady=10)
        self.train_button = customtkinter.CTkButton(self.model_frame, text='Train Model', command=self.start_training_thread)
        self.train_button.pack(padx=10, pady=10)

        self.epoch_frame = customtkinter.CTkFrame(self.model_frame)
        self.epoch_frame.pack(padx=10, pady=10, expand=True)
        self.epoch_label = customtkinter.CTkLabel(self.epoch_frame, text='Epochs')
        self.epoch_label.pack(side='left', padx=10, pady=10)
        self.epoch_entry = customtkinter.CTkEntry(self.epoch_frame, placeholder_text='10')
        self.epoch_entry.pack(side='left', padx=10, pady=10)

        #progress bar
        self.progress_frame = customtkinter.CTkFrame(parent)
        self.progress_bar = customtkinter.CTkProgressBar(self.progress_frame, orientation='horizontal')
        self.progress_bar.pack(padx=10, pady=10)
        self.progress_bar.set(0)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def show_progress_bar(self):
        self.progress_frame.pack(side='bottom', padx=10, pady=10, expand=True)

    def hide_progress_bar(self):
        self.progress_frame.pack_forget()

    def dataset_select(self):
        filetypes = [
            ('Excel', '*.xlsx'),
            ('Csv', '*.csv')
        ]
        self.data_dir = filedialog.askopenfilename(initialdir='training', initialfile='shotgun_data.xlsx', filetypes=filetypes)

    #start up thread so that app doesn't lock up
    def start_training_thread(self):
        thread = threading.Thread(target=self.train)
        thread.daemon = True
        thread.start()

    #read in dataset from xml file
    def make_dataset(self):
        #get audio files
        datasets = pd.read_excel(self.data_dir)

        #make datasets
        supervised_dataset = []
        for i, row in datasets.iterrows():
            dataset = shotgun_training_dataset(row['Audiofile'], row['Labels'])
            supervised_dataset.append(dataset)
            
        #concat into one dataset
        supervised_dataset = torch.utils.data.ConcatDataset(supervised_dataset)

        #split into traing and test sets
        splits = torch.utils.data.random_split(supervised_dataset, [0.8, 0.2])

        train_dataloader = DataLoader(splits[0], batch_size=64, shuffle=True)
        test_dataloader = DataLoader(splits[1], batch_size=64, shuffle=True)

        return train_dataloader, test_dataloader

    def train(self):
        #disable buttons so multiple threads can't be started up
        self.train_button.configure(state='disabled')
        self.dataset_button.configure(state='disabled')
        epochs = int(self.epoch_entry.get())

        if epochs <= 0:
            print('Number of epochs must be an integer and must be positive')
            self.train_button.configure(state='normal')
            self.dataset_button.configure(state='normal')
            return
        
        #get a save destination
        save_destination = filedialog.asksaveasfilename(initialdir='model_weights', filetypes=[('PyTorch weight', '.pth')])

        #get datasets
        train_dataloader, test_dataloader = self.make_dataset()

        #create new model
        self.model = shotgun_detector((128, 188)).to(self.device)
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(86).to(self.device))
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)

        losses = []
        vlosses = []
        
        self.show_progress_bar()

        #train
        for epoch in range(epochs):
            prog_low_bound = epoch / epochs
            prog_high_bound = (epoch + 1) / epochs
            prog_mid_bound = (prog_low_bound + prog_high_bound) / 2
            avg_loss, avg_f1, avg_prec, avg_rec, avg_iou = self.train_step(self.model, optimizer, loss_fn, train_dataloader, prog_low_bound, prog_mid_bound)
            vavg_loss, vavg_f1, vavg_prec, vavg_rec, vavg_iou = self.test_step(self.model, loss_fn, test_dataloader, prog_mid_bound, prog_high_bound)
            losses.append(avg_loss)
            vlosses.append(vavg_loss)

            print(f'Epoch: {epoch+1} - Loss: {avg_loss}, f1: {avg_f1}, Precision: {avg_prec}, Recall: {avg_rec}, IoU: {avg_iou}')
            print(f'Validation - Loss: {vavg_loss}, f1: {vavg_f1}, Precision: {vavg_prec}, Recall: {vavg_rec}, IoU: {vavg_iou}\n')

        self.hide_progress_bar()

        #re-enable buttons
        self.train_button.configure(state='normal')
        self.dataset_button.configure(state='normal')

        #save model weights to selected destination
        torch.save(self.model.state_dict(), save_destination)


    def test_step(self, model, loss_fn, test_dataloader, prog_mid_bound, prog_high_bound):
        with torch.no_grad():
            sigmoid = torch.nn.Sigmoid()
            running_loss = 0
            running_f1 = 0
            running_precision = 0
            running_recall = 0
            running_iou = 0
            steps = 0
            total_steps = len(test_dataloader)
            prog_states = np.linspace(prog_mid_bound, prog_high_bound, total_steps+1)

            for i, vdata in enumerate(test_dataloader):
                #get batch on device
                inputs, labels = vdata
            
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
        
                #get predictions
                out = model(inputs)
        
                loss = loss_fn(out, labels)

                predictions = torch.round(sigmoid(out))
                predictions = predictions.to('cpu').detach().numpy().flatten()
        
                #add metrics
                running_loss += loss.item()
                labels = labels.to('cpu').numpy().flatten()
                running_f1 += f1_score(labels, predictions, zero_division=0)
                running_precision += precision_score(labels, predictions, zero_division=0)
                running_recall += recall_score(labels, predictions, zero_division=0)
                running_iou += jaccard_score(labels, predictions, zero_division=0)

                #update progress bar
                self.progress_bar.set(prog_states[i])
                steps += 1

            #get average metrics
            vavg_loss = running_loss / steps
            vavg_f1 = running_f1 / steps
            vavg_prec = running_precision / steps
            vavg_rec = running_recall / steps
            vavg_iou = running_iou / steps

            return vavg_loss, vavg_f1, vavg_prec, vavg_rec, vavg_iou


    def train_step(self, model, optimizer, loss_fn, dataloader, prog_low_bound, prog_mid_bound):
        running_loss = 0
        running_f1 = 0
        running_precision = 0
        running_recall = 0
        running_iou = 0

        sigmoid = torch.nn.Sigmoid()

        steps = 0
        total_steps = len(dataloader)
        prog_states = np.linspace(prog_low_bound, prog_mid_bound, total_steps+1)
            
        for i, data in enumerate(dataloader):
            #get batch on device
            inputs, labels = data
            
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            optimizer.zero_grad()

            #get predictions
            out = model(inputs)

            #update weights and optimizer
            loss = loss_fn(out, labels)
            loss.backward()

            optimizer.step()

            #add metrics
            predictions = torch.round(sigmoid(out))
            predictions = predictions.to('cpu').detach().numpy().flatten()

            running_loss += loss.item()
            labels = labels.to('cpu').numpy().flatten()
            running_f1 += f1_score(labels, predictions, zero_division=0)
            running_precision += precision_score(labels, predictions, zero_division=0)
            running_recall += recall_score(labels, predictions, zero_division=0)
            running_iou += jaccard_score(labels, predictions, zero_division=0)

            self.progress_bar.set(prog_states[i])
            steps += 1

        #get average metrics
        avg_loss = running_loss / steps
        avg_f1 = running_f1 / steps
        avg_prec = running_precision / steps
        avg_recall = running_recall / steps
        avg_iou = running_iou / steps

        return avg_loss, avg_f1, avg_prec, avg_recall, avg_iou

        
class prediction_tab(customtkinter.CTkFrame):

    def __init__(self, parent):
        super().__init__(parent)
        #input frame
        self.top_frame = customtkinter.CTkFrame(parent)
        self.top_frame.pack()

        #input selection
        self.input_frame = customtkinter.CTkFrame(self.top_frame)
        self.input_frame.pack(side='left', padx=10, pady=10, expand=True)
        self.input_frame_label = customtkinter.CTkLabel(self.input_frame, text='Audio Input')
        self.input_frame_label.pack(padx=10, pady=10, expand=True)
        self.inference_filenames = ()
        self.inference_file_button = customtkinter.CTkButton(self.input_frame, text='Select Files', command=self.inference_file_select)
        self.inference_file_button.pack(padx=10, pady=10)
        self.files_selected = 0
        self.file_select_label = customtkinter.CTkLabel(self.input_frame, text=f'{self.files_selected} Files Selected')
        self.file_select_label.pack(padx=10, pady=10)

        #model frame
        available_models = os.listdir('model_weights')
        self.model_frame = customtkinter.CTkFrame(self.top_frame)
        self.model_frame.pack(side='left', padx=10, pady=10, expand=True)
        self.model_frame_label = customtkinter.CTkLabel(self.model_frame, text='Model')
        self.model_frame_label.pack(padx=10, pady=10)
        self.model_selet = customtkinter.CTkComboBox(self.model_frame, values=available_models)
        self.model_selet.pack(padx=10, pady=10)
        self.run_model_button = customtkinter.CTkButton(self.model_frame, text='Run', command=self.start_prediction_thread)
        self.run_model_button.pack(padx=10, pady=10)
        self.cuda_checkbox = customtkinter.CTkCheckBox(self.model_frame, text='Use GPU', command=self.cuda_toggled)
        self.cuda_checkbox.pack(padx=10, pady=10)

        #progress bar
        self.progress_frame = customtkinter.CTkFrame(parent)
        self.progress_bar = customtkinter.CTkProgressBar(self.progress_frame, orientation='horizontal')
        self.progress_bar.pack(padx=10, pady=10)
        self.progress_bar.set(0)

        #load model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = shotgun_detector((128, 188)).to(self.device)
        self.model.load_state_dict(torch.load(os.path.join('model_weights', 'app_trained_10_hours_50_epochs.pth'), map_location=torch.device(self.device)))
        if self.device == 'cuda':
            self.cuda_checkbox.select()
        else:
            self.cuda_checkbox.deselect()

    def update_models(self):
        self.model_select['values'] = os.listdir('model_weights')

    def cuda_toggled(self):
        state = self.cuda_checkbox.get()

        match state: #update selected device
            case 1:
                if not torch.cuda.is_available(): #cuda unavailable so warn
                    window = customtkinter.CTkToplevel(self)
                    window.title('GPU not found')
                    label = customtkinter.CTkLabel(window, text='GPU was selected but a compatible device was detected. \nPlease check your hardware and GPU drivers before proceeding.')
                    label.pack(padx=10, pady=10)
                    button = customtkinter.CTkButton(window, text='Ok', command=lambda : window.destroy())
                    button.pack(padx=10, pady=10)
                    window.grab_set()

                #send to device
                self.device = 'cuda'
            case 0:
                #send to host
                self.device = 'cpu'
        
    def show_progress_bar(self):
        self.progress_frame.pack(side='bottom', padx=10, pady=10, expand=True)

    def hide_progress_bar(self):
        self.progress_frame.pack_forget()

    def inference_file_select(self):
        # Allow user to select a folder instead of individual files
        folder_selected = filedialog.askdirectory(initialdir='./')
        
        if not folder_selected:
            return  # In case the user cancels the dialog
        
        # Recursively find all .flac and .wav files in the selected folder and subfolders
        self.inference_filenames = []
        for root, dirs, files in os.walk(folder_selected):
            for file in files:
                if file.endswith(('.flac', '.wav')):
                    self.inference_filenames.append(os.path.join(root, file))
        
        # Update the file select label with the number of files found
        self.files_selected = len(self.inference_filenames)
        self.file_select_label.configure(text=f'{self.files_selected} Files Selected')
        


    #call predictions function on a thread to stop lock up
    def start_prediction_thread(self):
        self.model.to(self.device)
        self.model.load_state_dict(torch.load(os.path.join('model_weights', self.model_selet.get()), map_location=torch.device(self.device)))
        thread = threading.Thread(target=self.make_predictions)
        thread.daemon = True
        thread.start()

    def make_predictions(self):
        #disable buttons and show progress bar
        self.show_progress_bar()
        self.run_model_button.configure(state='disabled')
        self.inference_file_button.configure(state='disabled')
        num_files = len(self.inference_filenames)

        #constant values to estimate distance. Values measured from loudness experiments from 100 to 2000 meters
        dist_intercept = 22.473940046108346 
        dist_slope = -0.011855389957245197

        for file_i, file in enumerate(self.inference_filenames):
            #get dataset
            dataset = shotgun_inference_dataset(file)
            dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
            num_iterations = len(dataloader)

            name_splits = os.path.basename(file).split('_')
            # Validate that we have the expected number of parts
            if len(name_splits) < 6:
                print(f"Skipping file: {file} - invalid format")
                return
            # Extract the date of audio and time of start of audio file
            date_of_audio = name_splits[4]  # 20240104
            time_of_start = name_splits[5]  # 151002
            
            date = f'{date_of_audio[6:8]}/{date_of_audio[4:6]}/{date_of_audio[0:4]}'  # Format as DD/MM/YYYY


            try:
                with torch.no_grad():
                    #get iterations for progress bar
                    num_iterations = len(dataloader)
                    sigmoid = torch.nn.Sigmoid()

                    all_predictions = []
                    for step_i, inputs in enumerate(dataloader):
                        prog_states = np.linspace((file_i)/num_files, (file_i+1)/num_files, num_iterations+2)
                        progress = prog_states[step_i+1]
                        
                        self.progress_bar.set(progress)

                        #send data to device
                        inputs = inputs.to(self.device)

                        #get predictions
                        out = self.model(inputs)
                        pred_probs = sigmoid(out)
                        predictions = (pred_probs > 0.90).float() #only accept our most confident predictions
                        predictions = predictions.to('cpu').numpy()
                        all_predictions.append(predictions)
                
                predictions = np.concatenate(all_predictions, axis=0)

                sample_rate=24000
                n_fft=1024
                hop_length=512

                #get frame starts for 4 second long frames
                frame_starts = np.arange(0, len(predictions)) * 4

                #bin offsets for individual frames
                frame_offsets = np.arange(0, len(predictions[0])).astype(float)
                #frame_offsets = librosa.frames_to_time(frame_offsets, sr=sr, hop_length=hop_length)
                frame_offsets *= (4/len(predictions[0]))

                #export start stops to table
                table = pd.DataFrame(columns=['label', 'file', 'date', 'time_stamp', 'start_offset', 'end_offset', 'power', 'distance', 'true_positive'])
                entry_id = 0
                for i, p in enumerate(predictions):
                    t_0 = frame_starts[i]

                    new_entry = True
                    start_time = 0
                    end_time = 0
                    
                    for idx, pred in enumerate(p):
                        if pred == 1 and new_entry: #start of new identification
                            new_entry = False
                            start_time = t_0 + frame_offsets[idx]
                            end_time = start_time
                        elif (pred == 0 or idx == len(p)-1) and not new_entry: #end of identification
                            new_entry = True
                            end_time = t_0 + frame_offsets[idx]

                            #read in clip
                            data, sr = librosa.load(file, sr=24000, offset=start_time-.1, duration=end_time-start_time+.1) 
                            S = librosa.feature.melspectrogram(y=data, sr=sr, n_fft=1024, hop_length=512)
                            S = librosa.power_to_db(S)
                            S = S[:60,:]
                            power = np.max(S)
                            #extimate distance from power
                            distance = (power - dist_intercept) / dist_slope
                            distance = "<100" if distance < 0 else distance #don't have any measurements for <100 meters

                            #conver time for table
                            # Extract hours, minutes, and seconds from the start time
                            hours = int(time_of_start[0:2])
                            minutes = int(time_of_start[2:4])
                            seconds = int(time_of_start[4:6])
                            
                            # Convert and adjust start_time using the extracted values
                            minutes += int(start_time // 60)
                            seconds += int(start_time % 60)
                            if minutes >= 60:
                                hours += 1
                                minutes -= 60
                                
                            # Format the time for the output table
                            time_formatted = f'{hours:02d}:{minutes:02d}:{seconds:02d}'

                            #collect entry for table
                            # Collect entry for the table
                            entry = [f'{entry_id}', file, date, time_formatted, f'{start_time:.6f}', f'{end_time:.6f}', f'{power}', f'{distance}', 'T']
                            if entry[1] != entry[2]: #exclude predictions of length 0
                                entry_id += 1
                                table.loc[-1] = entry
                                table.index += 1
                
                #export each table to a file
                out_dir = 'outputs'
                if not os.path.isdir(out_dir):
                    os.mkdir(out_dir)
                file_out_name = os.path.basename(file).split('.')[0] + '.audacity_annotations.txt'
                out_path = os.path.join(out_dir, file_out_name)
                table.to_csv(out_path, sep='\t', columns=['start_offset', 'end_offset', 'label'], header=False, index=False)

                file_out_name = os.path.basename(file).split('.')[0] + '.xlsx'
                out_path = os.path.join(out_dir, file_out_name)
                table.to_excel(out_path, header=True, index=False)
            except Exception:
                print(f'Skipped {file}, it may be  corrupted')

        #re-enable buttons and reset progress bar
        self.run_model_button.configure(state='normal')
        self.inference_file_button.configure(state='normal')
        self.hide_progress_bar()
        self.progress_bar.set(0)


def main():
    gui = shotgun_detector_app()
    gui.mainloop()


if __name__ == '__main__':
    main()