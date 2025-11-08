"""
Cloud Run Service - Wetland Classification
Trains Random Forest on AlphaEarth embeddings and classifies wetlands
"""

import os
import json
import tempfile
import numpy as np
import pandas as pd
import rasterio
from flask import Flask, request, jsonify
from google.cloud import storage
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

app = Flask(__name__)

# Configuration
BUCKET_NAME = os.environ.get('BUCKET_NAME', 'wetmaps')
RANDOM_SEED = 42
CLASS_NAMES = ['Marsh', 'Shallow Open Water', 'Swamp', 'Fen']

# Initialize GCS client
storage_client = storage.Client()
bucket = storage_client.bucket(BUCKET_NAME)

def download_from_gcs(blob_path, local_path):
    """Download file from Cloud Storage"""
    print(f"Downloading gs://{BUCKET_NAME}/{blob_path}...")
    blob = bucket.blob(blob_path)
    blob.download_to_filename(local_path)
    print(f"✓ Downloaded to {local_path}")

def upload_to_gcs(local_path, blob_path):
    """Upload file to Cloud Storage"""
    print(f"Uploading {local_path}...")
    blob = bucket.blob(blob_path)
    blob.upload_from_filename(local_path)
    print(f"✓ Uploaded to gs://{BUCKET_NAME}/{blob_path}")

def load_training_data(csv_path):
    """Load training samples from CSV"""
    print(f"Loading training data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Extract feature columns (AlphaEarth bands A00-A15)
    feature_cols = [f'A{i:02d}' for i in range(16)]
    
    # Check which columns exist
    available_cols = [col for col in feature_cols if col in df.columns]
    print(f"Available feature columns: {len(available_cols)}")
    
    X = df[available_cols].values
    y = df['classIndex'].values
    
    print(f"✓ Loaded {len(X)} samples with {X.shape[1]} features")
    print(f"Class distribution: {dict(zip(CLASS_NAMES, np.bincount(y)))}")
    
    return X, y

def train_random_forest(X, y):
    """Train Random Forest classifier with 70/15/15 split"""
    print("\n=== TRAINING RANDOM FOREST ===")
    
    # Split: 70% train, 30% temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_SEED, stratify=y
    )
    
    # Split temp into 15% val, 15% test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=RANDOM_SEED, stratify=y_temp
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    
    # Train Random Forest (30 trees like GEE script)
    clf = RandomForestClassifier(
        n_estimators=30,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        max_depth=20,
        min_samples_split=5,
        verbose=1
    )
    
    print("\nTraining model (30 trees)...")
    clf.fit(X_train, y_train)
    print("✓ Training complete!")
    
    # Validation metrics
    print("\nEvaluating on validation set...")
    y_val_pred = clf.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    
    # Test metrics
    print("Evaluating on test set...")
    y_test_pred = clf.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    conf_matrix = confusion_matrix(y_test, y_test_pred)
    class_report = classification_report(y_test, y_test_pred, target_names=CLASS_NAMES)
    
    print(f"\n{'='*50}")
    print(f"VALIDATION ACCURACY: {val_accuracy*100:.1f}%")
    print(f"TEST ACCURACY: {test_accuracy*100:.1f}%")
    print(f"{'='*50}")
    print(f"\nConfusion Matrix:")
    print(f"{'':>20} Predicted")
    print(f"{'':>20} {'M':>6} {'SW':>6} {'Sw':>6} {'Fen':>6}")
    for i, row in enumerate(conf_matrix):
        print(f"Actual {CLASS_NAMES[i]:>15} {row[0]:>6} {row[1]:>6} {row[2]:>6} {row[3]:>6}")
    print(f"\nClassification Report:")
    print(class_report)
    
    # Calculate per-class metrics
    precision = []
    recall = []
    for i in range(len(CLASS_NAMES)):
        tp = conf_matrix[i, i]
        fp = conf_matrix[:, i].sum() - tp
        fn = conf_matrix[i, :].sum() - tp
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        precision.append(prec)
        recall.append(rec)
        
        print(f"{CLASS_NAMES[i]:>20}: Precision={prec*100:.1f}%, Recall={rec*100:.1f}%")
    
    results = {
        'val_accuracy': float(val_accuracy),
        'test_accuracy': float(test_accuracy),
        'confusion_matrix': conf_matrix.tolist(),
        'precision': precision,
        'recall': recall,
        'classification_report': class_report
    }
    
    return clf, results

def classify_geotiff(model, input_tif_path, output_tif_path):
    """Classify AlphaEarth GeoTIFF and save result"""
    print(f"\n=== CLASSIFYING GEOTIFF ===")
    print(f"Opening {input_tif_path}...")
    
    with rasterio.open(input_tif_path) as src:
        # Read all bands
        img = src.read()  # Shape: (bands, height, width)
        profile = src.profile
        
        print(f"Image shape: {img.shape}")
        print(f"Bands: {src.count}, Height: {src.height}, Width: {src.width}")
        
        # Reshape for prediction: (height*width, bands)
        height, width = img.shape[1], img.shape[2]
        img_flat = img.reshape(img.shape[0], -1).T  # (pixels, bands)
        
        # Handle NaN/invalid values
        valid_mask = ~np.isnan(img_flat).any(axis=1)
        
        print(f"Valid pixels: {valid_mask.sum():,} / {len(valid_mask):,}")
        
        # Predict on valid pixels
        print("Running classification...")
        predictions = np.full(len(img_flat), 255, dtype=np.uint8)  # 255 = nodata
        predictions[valid_mask] = model.predict(img_flat[valid_mask])
        
        # Reshape back to 2D
        classified = predictions.reshape(height, width)
        
        # Update profile for output
        profile.update(
            dtype=rasterio.uint8,
            count=1,
            compress='lzw',
            nodata=255
        )
        
        # Write output
        print(f"Writing classification to {output_tif_path}...")
        with rasterio.open(output_tif_path, 'w', **profile) as dst:
            dst.write(classified, 1)
        
        print(f"✓ Classification complete!")
        
        # Calculate class distribution (exclude nodata)
        valid_classified = classified[classified != 255]
        class_dist = {CLASS_NAMES[i]: int(np.sum(valid_classified == i)) 
                      for i in range(len(CLASS_NAMES))}
        
        print("\nClass Distribution:")
        for class_name, count in class_dist.items():
            print(f"  {class_name}: {count:,} pixels")
        
        return class_dist

@app.route('/', methods=['GET'])
def home():
    """Home endpoint"""
    return jsonify({
        'service': 'Wetland Classifier',
        'version': '1.0',
        'endpoints': ['/health', '/train', '/classify']
    }), 200

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'}), 200

@app.route('/train', methods=['POST'])
def train_endpoint():
    """
    Train Random Forest model on samples from Cloud Storage
    
    Request body:
    {
        "training_csv": "training_data/calgary_samples.csv",
        "model_output": "models/wetland_rf_model.joblib"
    }
    """
    try:
        data = request.get_json() or {}
        training_csv = data.get('training_csv', 'training_data/calgary_samples.csv')
        model_output = data.get('model_output', 'models/wetland_rf_model.joblib')
        
        print(f"\n{'='*60}")
        print(f"TRAINING REQUEST RECEIVED")
        print(f"{'='*60}")
        print(f"Training CSV: gs://{BUCKET_NAME}/{training_csv}")
        print(f"Model output: gs://{BUCKET_NAME}/{model_output}")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Download training data
            local_csv = os.path.join(tmpdir, 'training_data.csv')
            download_from_gcs(training_csv, local_csv)
            
            # Load data
            X, y = load_training_data(local_csv)
            
            # Train model
            model, results = train_random_forest(X, y)
            
            # Save model
            local_model = os.path.join(tmpdir, 'model.joblib')
            print(f"\nSaving model to {local_model}...")
            joblib.dump(model, local_model)
            upload_to_gcs(local_model, model_output)
            
            # Save results
            results_json = json.dumps(results, indent=2)
            results_path = model_output.replace('.joblib', '_results.json')
            local_results = os.path.join(tmpdir, 'results.json')
            with open(local_results, 'w') as f:
                f.write(results_json)
            upload_to_gcs(local_results, results_path)
            
            print(f"\n{'='*60}")
            print(f"TRAINING COMPLETE!")
            print(f"{'='*60}")
            
            return jsonify({
                'status': 'success',
                'model_path': f'gs://{BUCKET_NAME}/{model_output}',
                'results_path': f'gs://{BUCKET_NAME}/{results_path}',
                'results': {
                    'val_accuracy': results['val_accuracy'],
                    'test_accuracy': results['test_accuracy'],
                    'confusion_matrix': results['confusion_matrix']
                }
            }), 200
            
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"ERROR: {str(e)}")
        print(f"{'='*60}")
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/classify', methods=['POST'])
def classify_endpoint():
    """
    Classify AlphaEarth GeoTIFF using trained model
    
    Request body:
    {
        "model_path": "models/wetland_rf_model.joblib",
        "input_tif": "inference_data/calgary_alphaearth.tif",
        "output_tif": "results/calgary_classified.tif"
    }
    """
    try:
        data = request.get_json() or {}
        model_path = data.get('model_path', 'models/wetland_rf_model.joblib')
        input_tif = data.get('input_tif', 'inference_data/calgary_alphaearth.tif')
        output_tif = data.get('output_tif', 'results/calgary_classified.tif')
        
        print(f"\n{'='*60}")
        print(f"CLASSIFICATION REQUEST RECEIVED")
        print(f"{'='*60}")
        print(f"Model: gs://{BUCKET_NAME}/{model_path}")
        print(f"Input: gs://{BUCKET_NAME}/{input_tif}")
        print(f"Output: gs://{BUCKET_NAME}/{output_tif}")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Download model
            local_model = os.path.join(tmpdir, 'model.joblib')
            download_from_gcs(model_path, local_model)
            print("Loading model...")
            model = joblib.load(local_model)
            print(f"✓ Model loaded (trained on {model.n_features_in_} features)")
            
            # Download input GeoTIFF
            local_input = os.path.join(tmpdir, 'input.tif')
            download_from_gcs(input_tif, local_input)
            
            # Classify
            local_output = os.path.join(tmpdir, 'classified.tif')
            class_dist = classify_geotiff(model, local_input, local_output)
            
            # Upload result
            upload_to_gcs(local_output, output_tif)
            
            print(f"\n{'='*60}")
            print(f"CLASSIFICATION COMPLETE!")
            print(f"{'='*60}")
            
            return jsonify({
                'status': 'success',
                'output_path': f'gs://{BUCKET_NAME}/{output_tif}',
                'class_distribution': class_dist
            }), 200
            
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"ERROR: {str(e)}")
        print(f"{'='*60}")
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=True)