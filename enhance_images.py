#!/usr/bin/env python3
"""
ASL Example Image Enhancement Script
Automatically enhances example images for better visibility in Learning Mode
"""

import cv2
import numpy as np
from pathlib import Path
import os
import argparse


def enhance_image(image):
    """
    Apply enhancement techniques to make hand gestures clearer
    
    Args:
        image: Input image (BGR)
    
    Returns:
        Enhanced image
    """
    # 1. Apply CLAHE for better contrast
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l = clahe.apply(l)
    
    lab = cv2.merge([l, a, b])
    image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # 2. Slight sharpening
    kernel_sharpening = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])
    image = cv2.filter2D(image, -1, kernel_sharpening)
    
    # 3. Denoise (subtle)
    image = cv2.fastNlMeansDenoisingColored(image, None, 6, 6, 7, 21)
    
    return image


def add_border_and_label(image, letter):
    """
    Add white border and letter label
    
    Args:
        image: Input image
        letter: Letter/number being shown
    
    Returns:
        Image with border and label
    """
    # Add white border
    border_size = 15
    image = cv2.copyMakeBorder(
        image,
        border_size, border_size, border_size, border_size,
        cv2.BORDER_CONSTANT,
        value=[255, 255, 255]
    )
    
    # Add letter label at top-left
    font = cv2.FONT_HERSHEY_BOLD
    font_scale = 1.2
    thickness = 3
    color = (0, 120, 200)  # Blue
    
    text = f"{letter}"
    
    # Add text with shadow
    cv2.putText(image, text, (22, 47), font, font_scale, (0, 0, 0), thickness+1)  # Shadow
    cv2.putText(image, text, (20, 45), font, font_scale, color, thickness)  # Text
    
    return image


def check_image_quality(image):
    """
    Check basic image quality metrics
    
    Returns:
        (score, issues): Quality score 0-100, list of issues
    """
    issues = []
    score = 100
    
    # Check brightness
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    
    if mean_brightness < 70:
        issues.append(f"Too dark (brightness: {mean_brightness:.0f})")
        score -= 30
    elif mean_brightness > 190:
        issues.append(f"Too bright (brightness: {mean_brightness:.0f})")
        score -= 20
    
    # Check blur (Laplacian variance)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var < 80:
        issues.append(f"Blurry (sharpness: {laplacian_var:.0f})")
        score -= 40
    
    # Check contrast
    std = np.std(gray)
    if std < 30:
        issues.append(f"Low contrast (std: {std:.0f})")
        score -= 20
    
    return max(0, score), issues


def process_dataset(input_dir, output_dir, top_n=5, min_quality=50):
    """
    Process dataset and select best examples
    
    Args:
        input_dir: Input dataset directory
        output_dir: Output directory
        top_n: Number of best examples to keep per letter
        min_quality: Minimum quality score (0-100)
    """
    print(f"📂 Input: {input_dir}")
    print(f"📁 Output: {output_dir}")
    print(f"🔝 Selecting top {top_n} images per letter")
    print(f"⚡ Minimum quality score: {min_quality}")
    print()
    
    os.makedirs(output_dir, exist_ok=True)
    
    total_processed = 0
    total_selected = 0
    
    # Process each class
    for class_dir in sorted(Path(input_dir).iterdir()):
        if not class_dir.is_dir():
            continue
        
        if 'test' in class_dir.name.lower():
            continue
        
        letter = class_dir.name
        print(f"🔤 Processing: {letter}")
        
        # Get all images
        images = list(class_dir.glob('*.jpeg')) + list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
        
        if not images:
            print(f"   ⚠️  No images found")
            continue
        
        # Score all images
        image_scores = []
        for img_path in images:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            score, issues = check_image_quality(img)
            image_scores.append((img_path, score, issues))
            total_processed += 1
        
        # Sort by quality score (highest first)
        image_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Select top N that meet minimum quality
        selected = [item for item in image_scores if item[1] >= min_quality][:top_n]
        
        if not selected:
            print(f"   ⚠️  No images met quality threshold")
            continue
        
        # Create output directory
        output_class_dir = Path(output_dir) / letter
        output_class_dir.mkdir(parents=True, exist_ok=True)
        
        # Process selected images
        for i, (img_path, score, issues) in enumerate(selected):
            # Load
            image = cv2.imread(str(img_path))
            
            # Enhance
            enhanced = enhance_image(image)
            
            # Add border and label
            enhanced = add_border_and_label(enhanced, letter)
            
            # Save
            output_filename = f"{letter}_example_{i+1}.jpg"
            output_path = output_class_dir / output_filename
            cv2.imwrite(str(output_path), enhanced, [cv2.IMWRITE_JPEG_QUALITY, 92])
            
            status = "✅" if score >= 80 else "⚠️ " if score >= 60 else "❌"
            print(f"   {status} {output_filename} (quality: {score}/100)")
            
            if issues and score < 80:
                for issue in issues:
                    print(f"       - {issue}")
            
            total_selected += 1
        
        print()
    
    print(f"✅ Complete!")
    print(f"   Processed: {total_processed} images")
    print(f"   Selected: {total_selected} images")
    print(f"   Output: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Enhance ASL example images for Learning Mode'
    )
    parser.add_argument(
        '--input',
        default='datasets/asl_dataset',
        help='Input dataset directory'
    )
    parser.add_argument(
        '--output',
        default='datasets/asl_dataset_enhanced',
        help='Output directory for enhanced images'
    )
    parser.add_argument(
        '--top-n',
        type=int,
        default=5,
        help='Number of best examples to keep per letter (default: 5)'
    )
    parser.add_argument(
        '--min-quality',
        type=int,
        default=50,
        help='Minimum quality score 0-100 (default: 50)'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("ASL Example Image Enhancement")
    print("=" * 60)
    print()
    
    process_dataset(
        args.input,
        args.output,
        top_n=args.top_n,
        min_quality=args.min_quality
    )


if __name__ == "__main__":
    main()
