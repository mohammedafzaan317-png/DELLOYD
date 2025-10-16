import os
# Set environment variables BEFORE importing TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress all TensorFlow logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN custom operations

import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

# Optional: Also suppress Python warnings
import warnings
warnings.filterwarnings('ignore')

# Load pretrained VGG16 model
model = VGG16(weights='imagenet')

def classify_single_image(img_path):
    """Classify a single image and return the result"""
    if not os.path.exists(img_path):
        return "NOT_FOUND", [], 0.0
    
    try:
        # Load and preprocess image
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Predict
        preds = model.predict(img_array, verbose=0)
        decoded = decode_predictions(preds, top=3)[0]

        # Determine classification
        labels = [label.lower() for (_, label, _) in decoded]
        top_prob = decoded[0][2]  # Probability of top prediction
        
        if any("cat" in l for l in labels):
            return "CAT", decoded, top_prob
        elif any("dog" in l for l in labels):
            return "DOG", decoded, top_prob
        else:
            return "UNKNOWN", decoded, top_prob
            
    except Exception as e:
        return "ERROR", str(e), 0.0

def batch_classify_images():
    """Classify multiple images (up to 40)"""
    print("🐱🐶 Batch Image Classification Tool")
    print("=" * 50)
    
    # Get input method
    print("\nChoose input method:")
    print("1. Enter individual image paths")
    print("2. Enter a folder path")
    choice = input("Enter choice (1 or 2): ").strip()
    
    image_paths = []
    
    if choice == "1":
        print(f"\nEnter up to 40 image paths (press Enter without typing to finish):")
        for i in range(40):
            path = input(f"Image {i+1}: ").strip()
            if not path:  # Empty input means done
                break
            image_paths.append(path)
            
    elif choice == "2":
        folder_path = input("\nEnter folder path: ").strip()
        if not os.path.exists(folder_path):
            print("❌ Folder not found!")
            return
            
        # Get all image files from folder
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        for file in os.listdir(folder_path):
            if os.path.splitext(file)[1].lower() in valid_extensions:
                image_paths.append(os.path.join(folder_path, file))
                
        # Limit to 40 images
        image_paths = image_paths[:40]
        print(f"📁 Found {len(image_paths)} images in folder")
        
    else:
        print("❌ Invalid choice!")
        return
    
    if not image_paths:
        print("❌ No images to process!")
        return
    
    print(f"\n🔄 Processing {len(image_paths)} images...")
    print("=" * 60)
    
    # Classification results
    results = {
        "CAT": [],
        "DOG": [],
        "UNKNOWN": [],
        "ERROR": [],
        "NOT_FOUND": []
    }
    
    # Process each image
    for i, img_path in enumerate(image_paths, 1):
        print(f"\n[{i}/{len(image_paths)}] Processing: {os.path.basename(img_path)}")
        
        classification, details, confidence = classify_single_image(img_path)
        
        if classification == "NOT_FOUND":
            print("   ❌ Image not found")
            results["NOT_FOUND"].append(img_path)
        elif classification == "ERROR":
            print(f"   💥 Error: {details}")
            results["ERROR"].append((img_path, details))
        else:
            # Display top prediction
            top_label = details[0][1] if details else "Unknown"
            top_conf = details[0][2] if details else 0
            print(f"   🔍 Top prediction: {top_label} ({top_conf*100:.2f}%)")
            print(f"   ✅ Classification: {classification}")
            
            results[classification].append((img_path, top_label, confidence))
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 CLASSIFICATION SUMMARY")
    print("=" * 60)
    print(f"🐱 CATS: {len(results['CAT'])} images")
    print(f"🐶 DOGS: {len(results['DOG'])} images")
    print(f"❓ UNKNOWN: {len(results['UNKNOWN'])} images")
    print(f"💥 ERRORS: {len(results['ERROR'])} images")
    print(f"🔍 NOT FOUND: {len(results['NOT_FOUND'])} images")
    print(f"📁 TOTAL PROCESSED: {len(image_paths)} images")
    
    # Detailed breakdown
    if results['CAT']:
        print(f"\n🐱 CAT Images:")
        for img_path, label, conf in results['CAT']:
            print(f"   ✓ {os.path.basename(img_path)} - {label} ({conf*100:.1f}%)")
    
    if results['DOG']:
        print(f"\n🐶 DOG Images:")
        for img_path, label, conf in results['DOG']:
            print(f"   ✓ {os.path.basename(img_path)} - {label} ({conf*100:.1f}%)")
    
    if results['UNKNOWN']:
        print(f"\n❓ UNKNOWN Images:")
        for img_path, label, conf in results['UNKNOWN']:
            print(f"   ? {os.path.basename(img_path)} - {label} ({conf*100:.1f}%)")
    
    if results['ERROR']:
        print(f"\n💥 Images with Errors:")
        for img_path, error in results['ERROR']:
            print(f"   ✗ {os.path.basename(img_path)} - {error}")
    
    if results['NOT_FOUND']:
        print(f"\n🔍 Images Not Found:")
        for img_path in results['NOT_FOUND']:
            print(f"   ! {os.path.basename(img_path)}")

# Run the batch classification
if __name__ == "__main__":
    batch_classify_images()
