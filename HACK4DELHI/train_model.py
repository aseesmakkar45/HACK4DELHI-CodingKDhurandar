import numpy as np
import os
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle

class ImageClassifier:
    def __init__(self, img_size=(64, 64)):
        
        self.img_size = img_size
        self.model = LogisticRegression(max_iter=1000, random_state=42)
        self.label_names = {}
        
    def load_images_from_folder(self, root_folder):
        
        images = []
        labels = []
        
        
        classes = [d for d in os.listdir(root_folder) 
                  if os.path.isdir(os.path.join(root_folder, d))]
        
        
        self.label_names = {i: class_name for i, class_name in enumerate(classes)}
        
        print(f"Found {len(classes)} classes: {classes}")
        
        for label_idx, class_name in enumerate(classes):
            class_folder = os.path.join(root_folder, class_name)
            
            
            img_files = [f for f in os.listdir(class_folder) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
            
            print(f"Loading {len(img_files)} images from class '{class_name}'...")
            
            for img_file in img_files:
                img_path = os.path.join(class_folder, img_file)
                try:
                    
                    img = Image.open(img_path)
                    
                    
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    
                    img = img.resize(self.img_size)
                    
                    
                    img_array = np.array(img).flatten()
                    
                    
                    img_array = img_array / 255.0
                    
                    images.append(img_array)
                    labels.append(label_idx)
                    
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
        
        return np.array(images), np.array(labels)
    
    def train(self, X_train, y_train):
        
        print("\nTraining logistic regression model...")
        self.model.fit(X_train, y_train)
        print("Training complete!")
        
    def evaluate(self, X_test, y_test):
        
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\nTest Accuracy: {accuracy * 100:.2f}%")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, 
                                   target_names=list(self.label_names.values())))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        return accuracy
    
    def predict(self, image_path):
        """Predict class for a single image"""
        img = Image.open(image_path)
        
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        img = img.resize(self.img_size)
        img_array = np.array(img).flatten() / 255.0
        img_array = img_array.reshape(1, -1)
        
        prediction = self.model.predict(img_array)[0]
        probability = self.model.predict_proba(img_array)[0]
        
        return self.label_names[prediction], probability
    
    def save_model(self, filename='image_classifier.pkl'):
        
        model_data = {
            'model': self.model,
            'label_names': self.label_names,
            'img_size': self.img_size
        }
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"\nModel saved to {filename}")
    
    def load_model(self, filename='image_classifier.pkl'):
        
        with open(filename, 'rb') as f:
            model_data = pickle.load(f)
        self.model = model_data['model']
        self.label_names = model_data['label_names']
        self.img_size = model_data['img_size']
        print(f"\nModel loaded from {filename}")



if __name__ == "__main__":
    
    classifier = ImageClassifier(img_size=(64, 64))
    
    

    root_folder = "C:\\Users\\HP\\Desktop\\HACK4DELHI\\my images"  

    print("Loading images...")
    X, y = classifier.load_images_from_folder(root_folder)
    
    print(f"\nLoaded {len(X)} images with shape {X.shape}")
    print(f"Number of classes: {len(classifier.label_names)}")
    
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    
    classifier.train(X_train, y_train)
    
    
    classifier.evaluate(X_test, y_test)
    
    
    classifier.save_model('my_image_classifier.pkl')
    
    