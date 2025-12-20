"""
Egzersiz Sınıflandırıcı Modelleri

Bu modül, keypoint'lerden egzersiz sınıfını tahmin eden modelleri içerir:
- ImageClassifier: Görseller için MLP modeli
- SequenceClassifier: Videolar için LSTM modeli
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageClassifier(nn.Module):
    """
    Görseller için Egzersiz Sınıflandırıcı (MLP)
    
    Input: (batch_size, 132) - Keypoint'ler
    Output: (batch_size, num_classes) - Egzersiz sınıfı olasılıkları
    """
    
    def __init__(self, input_dim=132, num_classes=23, hidden_dims=[256, 128, 64], dropout=0.3):
        """
        Args:
            input_dim: Keypoint boyutu (132)
            num_classes: Egzersiz sınıf sayısı (23)
            hidden_dims: Gizli katman boyutları
            dropout: Dropout oranı
        """
        super(ImageClassifier, self).__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # MLP katmanları
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Çıktı katmanı
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        """
        Args:
            x: (batch_size, 132) - Keypoint'ler
        
        Returns:
            logits: (batch_size, num_classes) - Sınıf logit'leri
        """
        return self.network(x)
    
    def predict(self, x):
        """
        Tahmin yap ve sınıf döndür
        
        Args:
            x: (batch_size, 132) veya (132,)
        
        Returns:
            predicted_class: Sınıf indeksi
            probabilities: Sınıf olasılıkları
        """
        self.eval()
        with torch.no_grad():
            if x.dim() == 1:
                x = x.unsqueeze(0)  # (132,) -> (1, 132)
            
            logits = self.forward(x)
            probabilities = F.softmax(logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1)
            
            return predicted_class.item(), probabilities[0].cpu().numpy()


class SequenceClassifier(nn.Module):
    """
    Videolar için Egzersiz Sınıflandırıcı (LSTM)
    
    Input: (batch_size, sequence_length, 132) - Zaman serisi keypoint'ler
    Output: (batch_size, num_classes) - Egzersiz sınıfı olasılıkları
    """
    
    def __init__(
        self,
        input_dim=132,
        num_classes=23,
        hidden_dim=128,
        num_layers=2,
        dropout=0.3,
        bidirectional=False
    ):
        """
        Args:
            input_dim: Keypoint boyutu (132)
            num_classes: Egzersiz sınıf sayısı (23)
            hidden_dim: LSTM gizli katman boyutu
            num_layers: LSTM katman sayısı
            dropout: Dropout oranı
            bidirectional: Çift yönlü LSTM kullan
        """
        super(SequenceClassifier, self).__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # LSTM katmanları
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # LSTM çıktı boyutu
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        # Sınıflandırıcı katmanları
        self.fc1 = nn.Linear(lstm_output_dim, 64)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        """
        Args:
            x: (batch_size, sequence_length, 132) - Zaman serisi keypoint'ler
        
        Returns:
            logits: (batch_size, num_classes) - Sınıf logit'leri
        """
        # LSTM işle
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Son zaman adımının çıktısını al
        # lstm_out shape: (batch_size, sequence_length, hidden_dim)
        # Son frame'i al
        last_output = lstm_out[:, -1, :]  # (batch_size, hidden_dim)
        
        # Sınıflandırıcı
        x = F.relu(self.fc1(last_output))
        x = self.dropout(x)
        logits = self.fc2(x)
        
        return logits
    
    def predict(self, x):
        """
        Tahmin yap ve sınıf döndür
        
        Args:
            x: (batch_size, sequence_length, 132) veya (sequence_length, 132)
        
        Returns:
            predicted_class: Sınıf indeksi
            probabilities: Sınıf olasılıkları
        """
        self.eval()
        with torch.no_grad():
            if x.dim() == 2:
                x = x.unsqueeze(0)  # (seq_len, 132) -> (1, seq_len, 132)
            
            logits = self.forward(x)
            probabilities = F.softmax(logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1)
            
            return predicted_class.item(), probabilities[0].cpu().numpy()


def create_image_classifier(num_classes=23, **kwargs):
    """ImageClassifier oluştur"""
    return ImageClassifier(num_classes=num_classes, **kwargs)


def create_sequence_classifier(num_classes=23, **kwargs):
    """SequenceClassifier oluştur"""
    return SequenceClassifier(num_classes=num_classes, **kwargs)


if __name__ == "__main__":
    # Test
    print("="*60)
    print("MODEL MİMARİSİ TESTİ")
    print("="*60)
    
    # ImageClassifier test
    print("\n[ImageClassifier Testi]")
    image_model = ImageClassifier(num_classes=23)
    print(f"   Model parametre sayisi: {sum(p.numel() for p in image_model.parameters()):,}")
    
    # Test input
    batch_size = 8
    test_input = torch.randn(batch_size, 132)
    output = image_model(test_input)
    print(f"   Input shape: {test_input.shape}")
    print(f"   Output shape: {output.shape}")
    
    # Prediction test
    pred_class, probs = image_model.predict(test_input[0])
    print(f"   Tahmin edilen sinif: {pred_class}")
    print(f"   En yuksek olasilik: {probs[pred_class]:.4f}")
    
    # SequenceClassifier test
    print("\n[SequenceClassifier Testi]")
    sequence_model = SequenceClassifier(num_classes=23)
    print(f"   Model parametre sayısı: {sum(p.numel() for p in sequence_model.parameters()):,}")
    
    # Test input
    sequence_length = 60
    test_input = torch.randn(batch_size, sequence_length, 132)
    output = sequence_model(test_input)
    print(f"   Input shape: {test_input.shape}")
    print(f"   Output shape: {output.shape}")
    
    # Prediction test
    pred_class, probs = sequence_model.predict(test_input[0])
    print(f"   Tahmin edilen sınıf: {pred_class}")
    print(f"   En yüksek olasılık: {probs[pred_class]:.4f}")
    
    print("\n[OK] Model testleri basarili!")

