# Pipeline implementation for integration with the annotation tool
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torchaudio
import torchvision.transforms as transforms
import os
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoFeatureExtractor, ASTForAudioClassification

class UserModel(nn.Module):
    # Create user model for transfer learning
    def __init__(self, classes):
        super(UserModel, self).__init__()
        in_features = 527
        self.linear1 = torch.nn.Linear(in_features, 100)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(100, len(classes)) # This needs to be dynamically set
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return x

class AudioDataset(Dataset):
    # ANNOTATIONS, AUDIO_DIR, mel_spectrogram, 16000, False
    def __init__(self, annotations_file, audio_dir, sr, val):
        super(AudioDataset, self).__init__()
        print(f"[DATASET] Initializing AudioDataset - Validation: {val}, Audio dir: {audio_dir}")
        self.val = val
        self.annotations = self._filter_annotations(pd.read_csv(annotations_file))
        print(f"[DATASET] Loaded {len(self.annotations)} annotations from {annotations_file}")
        self.audio_dir = audio_dir
        self.target_rate = sr
        self.resize = transforms.Resize((224,224))
        print(f"[DATASET] Loading feature extractor...")
        self.feature_extractor = AutoFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        print(f"[DATASET] Feature extractor loaded successfully")

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        audio_path = self._get_audio_path(index)
        label = self._get_audio_label(index)
        signal, sr = torchaudio.load(audio_path)
        resample = torchaudio.transforms.Resample(sr, self.target_rate)
        signal = resample(signal[0])
        feature = self.feature_extractor(signal, sampling_rate=self.target_rate, return_tensors="pt")
        feature = feature['input_values']
        return feature,label

    def _get_audio_path(self, index):
        path = os.path.join(self.audio_dir, self.annotations.iloc[index,1])
        return path

    def _get_audio_label(self, index):
        return self.annotations.iloc[index,3]

    def _get_validation(self, index):
        return self.annotations.iloc[index,2]

    def _filter_annotations(self, data):
        return data.loc[data['Validation'] == self.val]
    
    def shuffle(self):
        pass


def encode(labels, classes):
    target = []
    for label in labels:
        y = [0 for i in range(len(classes))] # This needs to be dynamically set
        y[classes.index(label)] = 1
        target.append(y)
    target = torch.Tensor(target)
    return target


def train(user, training_set, validation_set, classes, num_epoch=25, batch_size=12):
    print(f"[TRAIN] Starting training process for user {user.id}")
    print(f"[TRAIN] Training samples: {len(training_set)}, Validation samples: {len(validation_set)}")
    print(f"[TRAIN] Classes: {classes}")
    print(f"[TRAIN] Epochs: {num_epoch}, Batch size: {batch_size}")
    
    # Enhanced device detection with detailed logging
    print(f"[TRAIN] Checking available devices...")
    print(f"[TRAIN] PyTorch version: {torch.__version__}")
    print(f"[TRAIN] CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"[TRAIN] CUDA device count: {torch.cuda.device_count()}")
        print(f"[TRAIN] Current CUDA device: {torch.cuda.current_device()}")
        print(f"[TRAIN] CUDA device name: {torch.cuda.get_device_name(0)}")
        print(f"[TRAIN] CUDA capability: {torch.cuda.get_device_capability(0)}")
    else:
        print(f"[TRAIN] CUDA not available - checking PyTorch installation...")
        print(f"[TRAIN] PyTorch built with CUDA: {torch.version.cuda}")
    
    print(f"[TRAIN] MPS available: {torch.backends.mps.is_available()}")
    
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"[TRAIN] Selected CUDA device: {device}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"[TRAIN] Selected MPS device: {device}")
    else:
        device = torch.device("cpu")
        print(f"[TRAIN] Selected CPU device: {device}")
    
    print(f"[TRAIN] Final device: {device}")
    
    # Test GPU functionality if using CUDA
    if device.type == 'cuda':
        try:
            test_tensor = torch.tensor([1.0, 2.0, 3.0]).to(device)
            print(f"[TRAIN] GPU test successful - tensor created on GPU: {test_tensor.device}")
        except Exception as e:
            print(f"[TRAIN] GPU test failed: {e}")
            print(f"[TRAIN] Falling back to CPU")
            device = torch.device("cpu")
            print(f"[TRAIN] Device changed to: {device}")
    
    print(f"[TRAIN] Creating user model with {len(classes)} classes")
    usermodel = UserModel(classes)
    usermodel = usermodel.to(device)
    print(f"[TRAIN] User model created and moved to device")

    MODEL_PATH = f"./static/{user.id}/model/model.pth"
    if not os.path.exists(MODEL_PATH):
        print(f"[TRAIN] Model file not found, downloading pre-trained model...")
        model = ASTForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        print(f"[TRAIN] Pre-trained model downloaded successfully")
    else:
        print(f"[TRAIN] Loading existing model from {MODEL_PATH}")
        # Use weights_only=False for compatibility with older model files
        model = torch.load(MODEL_PATH, weights_only=False)
        print(f"[TRAIN] Existing model loaded successfully")
        
    model = model.to(device)
    print(f"[TRAIN] Model moved to device: {device}")
    
    # Log GPU memory usage if using CUDA
    if device.type == 'cuda':
        print(f"[TRAIN] GPU memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"[TRAIN] GPU memory cached: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")

    criterion = nn.CrossEntropyLoss()
    print(f"[TRAIN] Loss function initialized: CrossEntropyLoss")

    train_loss = []
    train_accuracy = []
    val_loss = []
    val_accuracy = [] 
    best_acc = 0

    learning_rate = 1e-5
    print(f"[TRAIN] Initial learning rate: {learning_rate}")

    # Recommended hyper-parameters - epoch:25, lr:1e-5 (halving every 5 epochs after epoch 10), batch:12
    print(f"[TRAIN] Starting training loop for {num_epoch} epochs...")
    for epoch in range(num_epoch):
        print(f"[TRAIN] ========== EPOCH {epoch + 1}/{num_epoch} ==========")
        optimizer = optim.Adam(list(model.parameters()) + list(usermodel.parameters()), lr=learning_rate) # Removed model.parameters()
        print(f"[TRAIN] Optimizer created with learning rate: {learning_rate}")

        if epoch > 10 and (epoch - 10) % 5 == 0:
            learning_rate = learning_rate/2
            print(f"[TRAIN] Learning rate reduced to: {learning_rate}")

        running_loss = 0
        running_corrects = 0
        val_running_loss = 0
        corrects = 0
        total = 0
        i = 0

        GT = []
        pred = []
        
        print(f"[TRAIN] Starting training phase for epoch {epoch + 1}...")

        for data, label in training_set:
            data = data.to(device)
            data = torch.squeeze(data,1)
            optimizer.zero_grad()

            embeddings = model(data).logits
            outputs = usermodel(embeddings)
            y = encode([label], classes) # list wrapper as label not batched
            y = y.to(device)

            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            total += len(torch.argmax(y,dim=1))
            corrects += (torch.argmax(y,dim=1) == torch.argmax(outputs,dim=1)).sum()
            running_corrects = 100*corrects/total

            accuracy = running_corrects.item()
            running_loss += loss.item()
            
            if (i % 100 == 1) and (i != 1):
                print(f"[TRAIN] [{i}/{len(training_set)}] - Training Accuracy: {accuracy:.2f}, Training loss: {running_loss/i:.2f}")
            i += 1

        train_loss.append(running_loss)
        train_accuracy.append(accuracy)
        print(f"[TRAIN] Training phase completed - Final Accuracy: {accuracy:.2f}, Final Loss: {running_loss/len(training_set):.2f}")

        print(f"[TRAIN] Starting validation phase for epoch {epoch + 1}...")
        val_running = 0
        val_total = 0
        val_corrects = 0
        val_running_accuracy = 0.0  # Initialize validation accuracy
        i = 0

        if len(validation_set) == 0:
            print(f"[TRAIN] No validation samples available, skipping validation phase")
            val_running_loss = 0.0
            val_running_accuracy = 0.0
        else:
            for data, label in validation_set:
                data = data.to(device)
                data = torch.squeeze(data,1)

                embeddings = model(data).logits
                outputs = usermodel(embeddings)

                y = encode([label], classes)
                y = y.to(device)
                loss = criterion(outputs, y)

                val_total += len(torch.argmax(y,dim=1))
                val_corrects += (torch.argmax(y,dim=1) == torch.argmax(outputs,dim=1)).sum()
                val_running = 100*val_corrects/val_total
                val_running_accuracy = val_running.item()

                val_running_loss += loss.item()

                GT.append(list(y[0].cpu()).index(1))
                pred.append(torch.argmax(outputs,dim=1).cpu().item())

                if (i % 100 == 1) and (i != 1):
                    print(f"[TRAIN] Val Accuracy: {val_running_accuracy:.2f}, Val loss: {val_running_loss/i:.2f}")
                i += 1

        val_loss.append(val_running_loss)
        val_accuracy.append(val_running_accuracy)
        
        if len(validation_set) > 0:
            print(f"[TRAIN] Validation phase completed - Final Val Accuracy: {val_running_accuracy:.2f}, Final Val Loss: {val_running_loss/len(validation_set):.2f}")
            print(f'[TRAIN] EPOCH {epoch + 1} SUMMARY - Training Accuracy: {accuracy:.2f}, Training loss: {running_loss/len(training_set):.2f}, Val Accuracy: {val_running_accuracy:.2f}, Val loss: {val_running_loss/len(validation_set):.2f}')
        else:
            print(f"[TRAIN] Validation phase skipped - No validation samples")
            print(f'[TRAIN] EPOCH {epoch + 1} SUMMARY - Training Accuracy: {accuracy:.2f}, Training loss: {running_loss/len(training_set):.2f}, Val Accuracy: N/A, Val loss: N/A')

        if (val_running_accuracy > best_acc):
            best_acc = val_running_accuracy
            print(f'[TRAIN] New best validation accuracy: {best_acc:.2f} - Saving models')
            # torch.save(enhance, "./models/enhance.pth")
            torch.save(model, f"./static/{user.id}/model/model.pth")
            torch.save(usermodel, f"./static/{user.id}/model/usermodel.pth")
            print(f'[TRAIN] Models saved to ./static/{user.id}/model/')
        else:
            print(f'[TRAIN] Validation accuracy {val_running_accuracy:.2f} not better than best {best_acc:.2f} - not saving models')
    
    print(f"[TRAIN] ========== TRAINING COMPLETED ==========")
    print(f"[TRAIN] Final best validation accuracy: {best_acc:.2f}")
    print(f"[TRAIN] Training completed for user {user.id}")
    return train_loss, train_accuracy, val_loss, val_accuracy

def predict(filename, user, classes, target_rate=16000):
    print(f"[PREDICT] Starting prediction for file: {filename}")
    print(f"[PREDICT] CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"[PREDICT] Using CUDA device: {device}")
    else:
        device = torch.device("cpu")
        print(f"[PREDICT] Using CPU device: {device}")
    PATH = f'./static/{user.id}/seg/{filename}'
    signal, sr = torchaudio.load(PATH)
    resample = torchaudio.transforms.Resample(sr, target_rate)
    signal = resample(signal[0])
    feature_extractor = AutoFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
    feature = feature_extractor(signal, sampling_rate=target_rate, return_tensors="pt")
    feature = feature['input_values'].to(device)

    MODEL_PATH = f'./static/{user.id}/model/model.pth'

    # Load trained models
    if os.path.exists(MODEL_PATH):
        model = torch.load(f"./static/{user.id}/model/model.pth", weights_only=False).to(device)
        usermodel = torch.load(f"./static/{user.id}/model/usermodel.pth", weights_only=False).to(device)

        embeddings = model(feature).logits
        outputs = usermodel(embeddings)

        index = torch.argmax(outputs,dim=1).item()
        prediction = classes[index]
        confidence = outputs[0][index].item()

        return prediction, confidence
    else:
        return 'unknown', 0
    
def embeddings(filename, user, target_rate=16000):
    # Output embeddings of feature extractor block
    print(f"[EMBEDDINGS] Starting embeddings extraction for file: {filename}")
    print(f"[EMBEDDINGS] CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"[EMBEDDINGS] Using CUDA device: {device}")
    else:
        device = torch.device("cpu")
        print(f"[EMBEDDINGS] Using CPU device: {device}")
    PATH = f'./static/{user.id}/seg/{filename}'
    signal, sr = torchaudio.load(PATH)
    resample = torchaudio.transforms.Resample(sr, target_rate)
    signal = resample(signal[0])
    feature_extractor = AutoFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
    feature = feature_extractor(signal, sampling_rate=target_rate, return_tensors="pt")
    feature = feature['input_values'].to(device) 

    MODEL_PATH = f'./static/{user.id}/model/model.pth'

    # Load trained models
    if os.path.exists(MODEL_PATH):
        model = torch.load(f"./static/{user.id}/model/model.pth", weights_only=False).to(device)
        embeddings = model(feature).logits.cpu().detach().numpy()
        return embeddings
    else:
        # Make directory
        os.mkdir(f'./static/{user.id}/model/')

        # Create model
        model = ASTForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593").to(device)
        embeddings = model(feature).logits.cpu().detach().numpy()
        torch.save(model, MODEL_PATH)
        return embeddings


def pipeline(user, classes, num_epoch=25, batch_size=12):
    print(f"[PIPELINE] Starting training pipeline for user {user.id}")
    print(f"[PIPELINE] Classes to train: {classes}")
    
    AUDIO_DIR = f"./static/{user.id}/seg/"
    ANNOTATIONS = f"./static/{user.id}/model/annotations.csv"
    
    print(f"[PIPELINE] Audio directory: {AUDIO_DIR}")
    print(f"[PIPELINE] Annotations file: {ANNOTATIONS}")

    print(f"[PIPELINE] Creating training dataset...")
    training = AudioDataset(ANNOTATIONS, AUDIO_DIR, 16000, False)
    print(f"[PIPELINE] Training dataset created with {len(training)} samples")
    
    print(f"[PIPELINE] Creating validation dataset...")
    validation = AudioDataset(ANNOTATIONS, AUDIO_DIR, 16000, True)
    print(f"[PIPELINE] Validation dataset created with {len(validation)} samples")
    
    print(f"[PIPELINE] Starting model training...")
    loss, acc, val_loss, val_acc = train(user, training, validation, classes, num_epoch, batch_size)
    print(f"[PIPELINE] Training completed successfully")

    return loss, acc, val_loss, val_acc