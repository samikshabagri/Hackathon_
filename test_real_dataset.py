#!/usr/bin/env python3
"""
Real Dataset WAF Testing with Charts and Logging - FIXED VERSION
Uses actual CSIC 2010 dataset for comprehensive validation with visualizations
"""

import pandas as pd
import numpy as np
from ml_waf import ml_waf_decision
import logging
from sklearn.model_selection import train_test_split
import urllib.parse
import re
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
import json
import sys

# Create a custom logger
logger = logging.getLogger('real_dataset_test')
logger.setLevel(logging.INFO)

# Create handlers
file_handler = logging.FileHandler('real_dataset_test_fixed.log', mode='w', encoding='utf-8')
console_handler = logging.StreamHandler(sys.stdout)

# Create formatters and add it to handlers
log_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(log_format)
console_handler.setFormatter(log_format)

# Add handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

def create_real_dataset_charts(results, attack_stats, accuracy, total):
    """Create comprehensive charts for real dataset testing"""
    
    # Create output directory for charts
    os.makedirs('charts', exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # 1. Overall Performance Dashboard
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('🔬 Real Dataset WAF Performance Analysis', fontsize=16, fontweight='bold')
    
    # Pie chart for overall accuracy
    labels = ['Correct', 'Incorrect']
    sizes = [accuracy, 100-accuracy]
    colors = ['#2ecc71', '#e74c3c']
    ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax1.set_title('Overall Accuracy on Real Data', fontweight='bold')
    
    # Bar chart for performance by type
    attack_types = list(attack_stats.keys())
    accuracies = []
    for attack_type in attack_types:
        stats = attack_stats[attack_type]
        type_accuracy = stats["correct"] / stats["total"] * 100
        accuracies.append(type_accuracy)
    
    bars = ax2.bar(attack_types, accuracies, color=['#3498db', '#e74c3c'])
    ax2.set_title('Accuracy by Request Type', fontweight='bold')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_ylim(0, 105)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Confidence distribution histogram
    confidences = [r['confidence'] for r in results]
    decisions = [r['decision'] for r in results]
    
    # Separate confidences by decision
    allow_confidences = [conf for conf, dec in zip(confidences, decisions) if dec == 'ALLOW']
    block_confidences = [conf for conf, dec in zip(confidences, decisions) if dec == 'BLOCK']
    
    ax3.hist(allow_confidences, bins=20, alpha=0.7, label='ALLOW', color='#2ecc71')
    ax3.hist(block_confidences, bins=20, alpha=0.7, label='BLOCK', color='#e74c3c')
    ax3.set_title('Confidence Score Distribution', fontweight='bold')
    ax3.set_xlabel('Confidence Score')
    ax3.set_ylabel('Frequency')
    ax3.legend()
    
    # Summary table
    ax4.axis('off')
    summary_data = [
        ['Total Requests', f'{total:,}'],
        ['Correct Decisions', f'{int(accuracy*total/100):,}'],
        ['Incorrect Decisions', f'{int((100-accuracy)*total/100):,}'],
        ['Overall Accuracy', f'{accuracy:.2f}%'],
        ['', ''],
        ['Request Type', 'Accuracy'],
    ]
    
    for attack_type in attack_types:
        stats = attack_stats[attack_type]
        type_accuracy = stats["correct"] / stats["total"] * 100
        summary_data.append([attack_type, f'{type_accuracy:.2f}%'])
    
    table = ax4.table(cellText=summary_data, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style the table
    for i in range(len(summary_data)):
        for j in range(len(summary_data[0])):
            cell = table[(i, j)]
            if i == 0:  # Header row
                cell.set_facecolor('#34495e')
                cell.set_text_props(weight='bold', color='white')
    
    plt.tight_layout()
    plt.savefig('charts/real_dataset_performance_dashboard_fixed.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Detailed Results Heatmap
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Create heatmap data
    heatmap_data = []
    for result in results[:50]:  # Show first 50 results
        heatmap_data.append([
            result['confidence'],
            1 if result['decision'] == 'BLOCK' else 0,
            1 if result['correct'] else 0
        ])
    
    heatmap_df = pd.DataFrame(heatmap_data, 
                             columns=['Confidence', 'Decision (BLOCK=1)', 'Correct (1=Yes)'])
    
    sns.heatmap(heatmap_df.T, annot=False, cmap='RdYlGn', ax=ax, cbar_kws={'label': 'Value'})
    ax.set_title('Real Dataset Test Results Heatmap (First 50 Requests)', fontweight='bold')
    ax.set_xlabel('Request Index')
    ax.set_ylabel('Metrics')
    
    plt.tight_layout()
    plt.savefig('charts/real_dataset_heatmap_fixed.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Error Analysis Chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Error types analysis
    errors = [r for r in results if not r["correct"]]
    false_positives = [e for e in errors if e["expected"] == "ALLOW" and e["decision"] == "BLOCK"]
    false_negatives = [e for e in errors if e["expected"] == "BLOCK" and e["decision"] == "ALLOW"]
    
    error_types = ['False Positives', 'False Negatives']
    error_counts = [len(false_positives), len(false_negatives)]
    
    bars = ax1.bar(error_types, error_counts, color=['#f39c12', '#e74c3c'])
    ax1.set_title('Error Type Distribution', fontweight='bold')
    ax1.set_ylabel('Count')
    
    # Add value labels
    for bar, count in zip(bars, error_counts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 10,
                f'{count:,}', ha='center', va='bottom', fontweight='bold')
    
    # Confidence vs Accuracy scatter
    correct_confidences = [r['confidence'] for r in results if r['correct']]
    incorrect_confidences = [r['confidence'] for r in results if not r['correct']]
    
    ax2.scatter(range(len(correct_confidences)), correct_confidences, 
                alpha=0.6, label='Correct', color='#2ecc71', s=20)
    ax2.scatter(range(len(incorrect_confidences)), incorrect_confidences, 
                alpha=0.6, label='Incorrect', color='#e74c3c', s=20)
    ax2.set_title('Confidence vs Decision Correctness', fontweight='bold')
    ax2.set_xlabel('Request Index')
    ax2.set_ylabel('Confidence Score')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('charts/real_dataset_error_analysis_fixed.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Performance Trends
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Calculate rolling accuracy
    window_size = 100
    rolling_accuracy = []
    for i in range(window_size, len(results), window_size):
        window_results = results[i-window_size:i]
        window_correct = sum(1 for r in window_results if r['correct'])
        rolling_accuracy.append(window_correct / len(window_results) * 100)
    
    ax.plot(range(window_size, len(results), window_size), rolling_accuracy, 
            linewidth=2, color='#3498db', marker='o', markersize=4)
    ax.axhline(y=accuracy, color='#e74c3c', linestyle='--', label=f'Overall Accuracy: {accuracy:.2f}%')
    ax.set_title('Rolling Accuracy (Window Size: 100)', fontweight='bold')
    ax.set_xlabel('Request Index')
    ax.set_ylabel('Accuracy (%)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('charts/real_dataset_performance_trends_fixed.png', dpi=300, bbox_inches='tight')
    plt.close()

def extract_test_features(df):
    """Extract features for testing from real dataset"""
    logger.info("Extracting features from real dataset...")
    
    features = df.copy()
    
    # URL decode
    features['url_decoded'] = features['URL'].apply(lambda x: urllib.parse.unquote_plus(str(x)))
    features['body_decoded'] = features['content'].apply(lambda x: urllib.parse.unquote_plus(str(x)))
    
    return features

def test_waf_on_real_data():
    """Test WAF on real CSIC 2010 dataset with comprehensive logging and charts"""
    logger.info("=" * 80)
    logger.info("STARTING REAL DATASET WAF TESTING - FIXED VERSION")
    logger.info("=" * 80)
    
    # Load real dataset
    logger.info("Loading CSIC 2010 dataset...")
    df = pd.read_csv('csic_database.csv')
    
    # Check target column
    if 'classification' in df.columns:
        target_col = 'classification'
    elif 'label' in df.columns:
        target_col = 'label'
    elif 'class' in df.columns:
        target_col = 'class'
    else:
        logger.error("No target column found!")
        return False
    
    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Target distribution: {df[target_col].value_counts()}")
    
    # Extract features
    df = extract_test_features(df)
    
    # Split data (same as training: 80% train, 20% test)
    logger.info("Splitting data for testing...")
    test_df = df.sample(frac=0.2, random_state=42)  # 20% for testing
    logger.info(f"Test dataset size: {len(test_df)}")
    
    # Test WAF on real data
    results = []
    correct = 0
    total = len(test_df)
    
    logger.info("Testing WAF on real dataset...")
    
    for idx, row in test_df.iterrows():
        if idx % 1000 == 0:
            logger.info(f"Processed {idx}/{total} requests...")
        
        try:
            # Get request data
            url = str(row['URL'])
            method = str(row['Method']) if 'Method' in row else 'GET'
            body = str(row['content']) if 'content' in row else ""
            expected = "BLOCK" if row[target_col] == 1 else "ALLOW"
            
            # Test WAF
            result = ml_waf_decision(url, method, body=body)
            decision = result["decision"]
            confidence = result["probability"]
            
            is_correct = decision == expected
            
            if is_correct:
                correct += 1
            
            # Log detailed results (but limit to avoid huge log files)
            if idx % 100 == 0:  # Log every 100th request
                logger.info(f"Request {idx}: {method} {url[:50]}... | Expected: {expected} | Got: {decision} | Confidence: {confidence:.3f} | {'✅' if is_correct else '❌'}")
            
            results.append({
                "url": url[:100] + "..." if len(url) > 100 else url,
                "method": method,
                "expected": expected,
                "decision": decision,
                "confidence": confidence,
                "correct": is_correct,
                "true_label": row[target_col]
            })
            
        except Exception as e:
            logger.error(f"Error testing request {idx}: {e}")
            results.append({
                "url": str(row['URL'])[:100] + "..." if len(str(row['URL'])) > 100 else str(row['URL']),
                "method": str(row.get('Method', 'GET')),
                "expected": "BLOCK" if row[target_col] == 1 else "ALLOW",
                "decision": "ERROR",
                "confidence": 0.0,
                "correct": False,
                "true_label": row[target_col]
            })
    
    # Calculate statistics
    accuracy = correct / total * 100
    
    # Calculate by attack type
    attack_stats = {}
    for result in results:
        true_label = result["true_label"]
        attack_type = "Attack" if true_label == 1 else "Normal"
        
        if attack_type not in attack_stats:
            attack_stats[attack_type] = {"correct": 0, "total": 0}
        
        attack_stats[attack_type]["total"] += 1
        if result["correct"]:
            attack_stats[attack_type]["correct"] += 1
    
    # Generate charts
    logger.info("Generating comprehensive charts...")
    create_real_dataset_charts(results, attack_stats, accuracy, total)
    
    # Save detailed results to JSON
    results_data = {
        "test_date": datetime.now().isoformat(),
        "total_requests": total,
        "correct_decisions": correct,
        "accuracy": accuracy,
        "attack_stats": attack_stats,
        "sample_results": results[:20],  # Save first 20 results
        "error_examples": [r for r in results if not r["correct"]][:10]  # Save first 10 errors
    }
    
    with open('real_dataset_results_fixed.json', 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)
    
    # Print results
    print("\n" + "=" * 80)
    print("🔬 REAL DATASET WAF TESTING RESULTS - FIXED VERSION")
    print("=" * 80)
    print(f"📊 Overall Results:")
    print(f"   Total Requests: {total:,}")
    print(f"   Correct Decisions: {correct:,}")
    print(f"   Accuracy: {accuracy:.2f}%")
    print(f"   Error Rate: {100-accuracy:.2f}%")
    
    print(f"\n📋 Results by Type:")
    for attack_type, stats in attack_stats.items():
        type_accuracy = stats["correct"] / stats["total"] * 100
        print(f"   {attack_type}: {stats['correct']:,}/{stats['total']:,} ({type_accuracy:.2f}%)")
    
    # Show some examples
    print(f"\n📝 Sample Results (first 10):")
    for i, result in enumerate(results[:10]):
        status = "✅" if result["correct"] else "❌"
        print(f"   {i+1}. {status} {result['method']} {result['url']}")
        print(f"      Expected: {result['expected']}, Got: {result['decision']} ({result['confidence']:.3f})")
    
    # Show error examples
    errors = [r for r in results if not r["correct"]]
    if errors:
        print(f"\n❌ Error Examples (first 5):")
        for i, error in enumerate(errors[:5]):
            print(f"   {i+1}. {error['method']} {error['url']}")
            print(f"      Expected: {error['expected']}, Got: {error['decision']} ({error['confidence']:.3f})")
    
    print("\n" + "=" * 80)
    print("🎯 FINAL ASSESSMENT:")
    
    if accuracy >= 95:
        print("✅ EXCELLENT! WAF performs very well on real data!")
        print("✅ Ready for production deployment")
    elif accuracy >= 90:
        print("✅ GOOD! WAF performs well on real data")
        print("⚠️  Minor improvements needed")
    elif accuracy >= 80:
        print("⚠️  ACCEPTABLE! WAF needs some improvements")
        print("🔧 Consider tuning thresholds or adding patterns")
    else:
        print("❌ POOR! WAF needs significant improvements")
        print("🔧 Review feature engineering and decision logic")
    
    print(f"\n📈 Real Dataset Accuracy: {accuracy:.2f}%")
    print(f"🎯 This is the TRUE performance on real-world data!")
    print(f"📊 Charts saved to 'charts/' directory (with '_fixed' suffix)")
    print(f"📝 Detailed results saved to 'real_dataset_results_fixed.json'")
    print(f"📋 Log file: 'real_dataset_test_fixed.log'")
    
    # Log final summary
    logger.info("=" * 80)
    logger.info("REAL DATASET TESTING COMPLETED - FIXED VERSION")
    logger.info(f"Final Accuracy: {accuracy:.2f}%")
    logger.info(f"Total Requests: {total:,}")
    logger.info(f"Correct Decisions: {correct:,}")
    logger.info("=" * 80)
    
    # Force flush the log file
    file_handler.flush()
    
    return accuracy

if __name__ == "__main__":
    test_waf_on_real_data()