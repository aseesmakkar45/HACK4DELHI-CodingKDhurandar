import numpy as np
import os
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle

class SimpleImageClassifier:
    def __init__(self, img_size=(64, 64)):
        
        self.img_size = img_size
        self.model = LogisticRegression(max_iter=2000, random_state=42)
        self.label_names = {}
        
    def load_images(self, root_folder):
        
        images = []
        labels = []
        
        
        classes = sorted([d for d in os.listdir(root_folder) 
                         if os.path.isdir(os.path.join(root_folder, d))])
        
        if len(classes) != 2:
            print(f"⚠️  Warning: Found {len(classes)} classes, but expected 2!")
            print(f"   Classes found: {classes}")
        
        
        self.label_names = {i: class_name for i, class_name in enumerate(classes)}
        
        print("\n" + "="*60)
        print(f"📂 Found classes: {classes[0]} vs {classes[1]}")
        print("="*60)
        
        for label_idx, class_name in enumerate(classes):
            class_folder = os.path.join(root_folder, class_name)
            
            
            img_files = [f for f in os.listdir(class_folder) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
            
            print(f"📸 Loading {len(img_files)} images from '{class_name}'...")
            
            for img_file in img_files:
                img_path = os.path.join(class_folder, img_file)
                try:
                    img = Image.open(img_path)
                    
                    
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    
                    img = img.resize(self.img_size)
                    
                    
                    img_array = np.array(img).flatten() / 255.0
                    
                    images.append(img_array)
                    labels.append(label_idx)
                    
                except Exception as e:
                    print(f"⚠️  Error loading {img_file}: {e}")
        
        return np.array(images), np.array(labels)
    
    def train(self, X_train, y_train):
        
        print("\n🎓 Training logistic regression model...")
        self.model.fit(X_train, y_train)
        
        
        train_pred = self.model.predict(X_train)
        train_acc = accuracy_score(y_train, train_pred)
        print(f"✅ Training complete! Training accuracy: {train_acc*100:.2f}%")
        
    def evaluate(self, X_test, y_test):
        
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print("\n" + "="*60)
        print("📊 TEST RESULTS")
        print("="*60)
        print(f"🎯 Test Accuracy: {accuracy * 100:.2f}%")
        
        if accuracy >= 0.9:
            print("💎 Excellent performance!")
        elif accuracy >= 0.75:
            print("👍 Good performance!")
        elif accuracy >= 0.6:
            print("⚠️  Fair - consider adding more training images")
        else:
            print("❌ Poor - you need more/better training data")
        
        print("\n📈 Classification Report:")
        print(classification_report(y_test, y_pred, 
                                   target_names=list(self.label_names.values())))
        
        print("\n🔢 Confusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        print(f"\nClass 0 ({self.label_names[0]}): Correct={cm[0][0]}, Wrong={cm[0][1]}")
        print(f"Class 1 ({self.label_names[1]}): Correct={cm[1][1]}, Wrong={cm[1][0]}")
        
        return accuracy
    
    def predict(self, image_path):
        
        img = Image.open(image_path)
        
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        img = img.resize(self.img_size)
        img_array = np.array(img).flatten() / 255.0
        img_array = img_array.reshape(1, -1)
        
        prediction = self.model.predict(img_array)[0]
        probabilities = self.model.predict_proba(img_array)[0]
        
        return self.label_names[prediction], probabilities
    
    def save_model(self, filename='trained_model.pkl'):
        
        model_data = {
            'model': self.model,
            'label_names': self.label_names,
            'img_size': self.img_size
        }
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"\n💾 Model saved to: {filename}")
    
    def load_model(self, filename='trained_model.pkl'):
        
        with open(filename, 'rb') as f:
            model_data = pickle.load(f)
        self.model = model_data['model']
        self.label_names = model_data['label_names']
        self.img_size = model_data['img_size']
        print(f"✅ Model loaded from: {filename}")



if __name__ == "__main__":
    print("\n" + "="*60)
    print("🤖 2-CLASS IMAGE CLASSIFIER TRAINING")
    print("="*60)
    
    
    root_folder = r"C:\Users\HP\Desktop\HACK4DELHI\my_images"
    
    
    classifier = SimpleImageClassifier(img_size=(64, 64))
    
    print(f"\n📁 Looking for images in: {root_folder}\n")
    
    if not os.path.exists(root_folder):
        print(f"❌ ERROR: Folder not found!")
        print(f"   Create this folder: {root_folder}")
        print(f"   Then add 2 subfolders with your class names")
        exit()
    
    X, y = classifier.load_images(root_folder)
    
    if len(X) == 0:
        print("\n❌ No images found! Check your folder structure.")
        exit()
    
    print(f"\n✅ Loaded {len(X)} total images")
    print(f"   Images per class: {dict(zip(*np.unique(y, return_counts=True)))}")
    
    
    if len(X) < 10:
        print("\n⚠️  WARNING: Very few images! Accuracy may be unreliable.")
        print("   Recommendation: Get at least 30-50 images per class")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n📊 Data split:")
    print(f"   Training: {len(X_train)} images")
    print(f"   Testing: {len(X_test)} images")
    
    
    classifier.train(X_train, y_train)
    
    
    classifier.evaluate(X_test, y_test)
    
    
    model_filename = 'my_2class_model.pkl'
    classifier.save_model(model_filename)
    
    print("\n" + "="*60)
    print("✨ TRAINING COMPLETE!")
    print("="*60)
    print(f"\n📝 Your model is saved as: {model_filename}")
    print("\n🎯 To use it for predictions:")
    print("""
    from train_model import SimpleImageClassifier
    
    # Load the model
    classifier = SimpleImageClassifier()
    classifier.load_model('my_2class_model.pkl')
    
    # Predict new image
    result, confidence = classifier.predict('path/to/image.jpg')
    print(f"Prediction: {result}")
    print(f"Confidence: {confidence}")
    """)
    print("="*60 + "\n")