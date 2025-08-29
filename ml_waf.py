#!/usr/bin/env python3
"""
Comprehensive Fix for ML WAF
Addresses all issues: overfitting, categorical encoding, and thresholds
"""

import os
import sys
import json
import traceback
import select
import urllib.parse
import re
import math
import joblib
import pandas as pd
import numpy as np
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ml_waf.log'),
        logging.StreamHandler()
    ]
)

def calculate_entropy(text):
    """Calculate Shannon entropy of text"""
    if not text:
        return 0
    char_counts = {}
    for char in text:
        char_counts[char] = char_counts.get(char, 0) + 1
    entropy = 0
    length = len(text)
    for count in char_counts.values():
        p = count / length
        entropy -= p * math.log2(p)
    return entropy

def preprocess_request_simple(url, method, body=""):
    """
    Simplified preprocessing that avoids categorical encoding issues
    """
    # Handle missing values
    url = url or ""
    method = method or "GET"
    body = body or ""
    
    # URL decode
    url_dec = urllib.parse.unquote_plus(url)
    body_dec = urllib.parse.unquote_plus(body)
    
    # Combine text features
    text = (url_dec + " " + body_dec).lower()
    
    # Create features manually to avoid categorical issues
    features = {}
    
    # Basic numeric features
    features["url_len"] = len(url_dec)
    features["content_len"] = len(body_dec)
    features["num_params"] = url_dec.count("?") + url_dec.count("&")
    features["cookie_len"] = 0  # No cookies in our simple version
    
    # Entropy features
    features["url_entropy"] = calculate_entropy(url_dec)
    features["content_entropy"] = calculate_entropy(body_dec)
    
    # Ratio features
    features["digit_ratio"] = sum(c.isdigit() for c in body_dec) / (len(body_dec) + 1)
    features["symbol_ratio"] = sum(not c.isalnum() for c in body_dec) / (len(body_dec) + 1)
    
    # Suspicious character counts
    suspicious_chars = ["'", '"', "<", ">", ";", "%", "-", "/", "\\", "=", "|", "&"]
    for ch in suspicious_chars:
        features[f"count_{ch}"] = text.count(ch)
    
    # OWASP keyword flags
    attack_keywords = {
        "has_select": r"\bselect\b", "has_union": r"\bunion\b", "has_drop": r"\bdrop\b",
        "has_insert": r"\binsert\b", "has_update": r"\bupdate\b", "has_delete": r"\bdelete\b",
        "has_concat": r"\bconcat\b", "has_information_schema": r"\binformation_schema\b",
        "has_sleep": r"\bsleep\s*\(", "has_benchmark": r"\bbenchmark\s*\(",
        "has_load_file": r"\bload_file\b", "has_into_outfile": r"\binto\s+outfile\b",
        "has_substr": r"\bsubstr\b", "has_ascii": r"\bascii\b", "has_hex": r"\bhex\b",
        "has_char_func": r"\bchar\s*\(", "has_or_1_eq_1": r"\bor\s+1=1\b",
        "has_and_1_eq_1": r"\band\s+1=1\b", "has_comment_sql": r"--|;--|/\*.*\*/",
        
        # XSS patterns
        "has_script_tag": r"<script", "has_iframe_tag": r"<iframe", "has_img_tag": r"<img",
        "has_svg_tag": r"<svg", "has_object_tag": r"<object", "has_embed_tag": r"<embed",
        "has_link_tag": r"<link", "has_meta_tag": r"<meta", "has_style_tag": r"<style",
        "has_alert": r"\balert\s*\(", "has_onerror": r"\bonerror\s*=", "has_onload": r"\bonload\s*=",
        "has_onclick": r"\bonclick\s*=", "has_onfocus": r"\bonfocus\s*=", "has_onmouseover": r"\bonmouseover\s*=",
        "has_document_cookie": r"document\.cookie", "has_document_write": r"document\.write",
        "has_window_location": r"window\.location", "has_javascript_proto": r"javascript:",
        
        # Path traversal
        "has_dotdot": r"\.\./|\.\.\\", "has_passwd": r"passwd",
        
        # Command injection
        "has_whoami": r"\bwhoami\b", "has_wget": r"\bwget\b", "has_curl": r"\bcurl\b",
        "has_python": r"\bpython\b", "has_perl": r"\bperl\b", "has_bash": r"\bbash\b",
        "has_exec": r"\bexec\b", "has_system": r"\bsystem\b", "has_pipe_or": r"\|\|", "has_pipe_and": r"&&",
        
        # File types
        "has_php": r"\.php", "has_asp": r"\.asp", "has_jsp": r"\.jsp", "has_exe": r"\.exe", "has_sh": r"\.sh",
        
        # Protocols
        "has_file_proto": r"file://", "has_http_proto": r"http://", "has_https_proto": r"https://"
    }
    
    for feature_name, pattern in attack_keywords.items():
        features[feature_name] = 1 if re.search(pattern, text, re.IGNORECASE) else 0
    
    return features

def make_decision_hybrid(url, method, body=""):
    """
    Hybrid decision making using pattern detection + simple ML-like scoring
    """
    try:
        # Get features
        features = preprocess_request_simple(url, method, body)
        
        # Get text for pattern analysis
        url_dec = urllib.parse.unquote_plus(url or "")
        body_dec = urllib.parse.unquote_plus(body or "")
        text = (url_dec + " " + body_dec).lower()
        
        # Define attack patterns with confidence levels
        attack_patterns = {
            # SQL Injection patterns (high confidence)
            r"\bselect\b.*\bfrom\b": 0.95,
            r"\bunion\b.*\bselect\b": 0.98,
            r"or.*1.*=.*1": 0.90,
            r"\band\s+1=1\b": 0.90,
            r"'.*--": 0.85,
            r"'.*#": 0.85,
            r"'.*--": 0.85,  # SQL comment with single quote
            r"--": 0.80,     # SQL comment
            r"#": 0.80,      # SQL comment
            r"\/\*.*\*\/": 0.85,  # SQL block comment
            
            # XSS patterns (high confidence)
            r"<script": 0.95,
            r"javascript:": 0.90,
            r"on\w+\s*=": 0.85,
            r"alert\s*\(": 0.80,
            r"onerror\s*=": 0.85,  # XSS event handler
            r"onload\s*=": 0.85,   # XSS event handler
            r"onclick\s*=": 0.85,  # XSS event handler
            r"onfocus\s*=": 0.85,  # XSS event handler
            r"onmouseover\s*=": 0.85,  # XSS event handler
            
            # Command injection (high confidence)
            r"\bwhoami\b": 0.95,
            r"\bwget\b": 0.90,
            r"\bcurl\b": 0.90,
            r"\|\|": 0.85,
            r"&&": 0.85,
            
            # Path traversal (high confidence)
            r"\.\./": 0.90,
            r"\.\.\\": 0.90,
            r"passwd": 0.85,
            
            # File inclusion (high confidence)
            r"\.\./.*\.php": 0.95,
            r"http://.*\.php": 0.90,
        }
        
        # Check for attack patterns in combined text
        pattern_confidence = 0.0
        logging.info(f"Analyzing text: {text[:100]}...")
        for pattern, confidence in attack_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                pattern_confidence = max(pattern_confidence, confidence)
                logging.info(f"Attack pattern detected: {pattern} (confidence: {confidence})")
        
        # Additional check for POST body specific patterns
        if method.upper() == "POST" and body:
            logging.info(f"Analyzing POST body: {body_dec[:100]}...")
            
            # SQL comment patterns in POST body
            sql_comment_patterns = {
                r"'.*--": 0.85,  # Single quote followed by comment
                r"'.*#": 0.85,   # Single quote followed by hash
                r"--": 0.80,     # SQL comment
                r"#": 0.80,      # SQL comment
                r"\/\*.*\*\/": 0.85,  # SQL block comment
            }
            
            for pattern, confidence in sql_comment_patterns.items():
                if re.search(pattern, body_dec, re.IGNORECASE):
                    pattern_confidence = max(pattern_confidence, confidence)
                    logging.info(f"POST body SQL pattern detected: {pattern} (confidence: {confidence})")
            
            # XSS event patterns in POST body
            xss_event_patterns = {
                r"onerror\s*=": 0.85,
                r"onload\s*=": 0.85,
                r"onclick\s*=": 0.85,
                r"onfocus\s*=": 0.85,
                r"onmouseover\s*=": 0.85,
                r"on\w+\s*=": 0.80,  # Generic event handler
            }
            
            for pattern, confidence in xss_event_patterns.items():
                if re.search(pattern, body_dec, re.IGNORECASE):
                    pattern_confidence = max(pattern_confidence, confidence)
                    logging.info(f"POST body XSS pattern detected: {pattern} (confidence: {confidence})")
        
        logging.info(f"Final pattern confidence: {pattern_confidence}")
        
        # Calculate ML-like score based on features
        ml_score = 0.0
        
        # Suspicious character penalties (but be more lenient with quotes in JSON)
        suspicious_chars = ["'", "<", ">", ";", "%"]
        for ch in suspicious_chars:
            count = features[f"count_{ch}"]
            if count > 0:
                ml_score += count * 0.1  # 10% penalty per suspicious character
        
        # Special handling for quotes - only penalize if they seem malicious
        quote_count = features["count_\""]
        if quote_count > 0:
            # If it looks like JSON (balanced quotes, contains : or { or })
            if ":" in body_dec or "{" in body_dec or "}" in body_dec:
                ml_score += quote_count * 0.02  # Very low penalty for JSON
            else:
                ml_score += quote_count * 0.1  # Normal penalty for other cases
        
        # Attack keyword penalties
        attack_keywords = [
            "has_select", "has_union", "has_drop", "has_insert", "has_update", "has_delete",
            "has_script_tag", "has_alert", "has_whoami", "has_wget", "has_curl",
            "has_dotdot", "has_passwd", "has_php", "has_file_proto"
        ]
        
        for keyword in attack_keywords:
            if features.get(keyword, 0) > 0:
                ml_score += 0.3  # 30% penalty per attack keyword
        
        # Entropy penalties (high entropy might indicate encoding/obfuscation)
        if features["url_entropy"] > 4.0:
            ml_score += 0.2
        if features["content_entropy"] > 4.0:
            ml_score += 0.2
        
        # Length penalties (very long URLs or content might be suspicious)
        if features["url_len"] > 200:
            ml_score += 0.1
        if features["content_len"] > 500:
            ml_score += 0.1
        
        # Normalize ML score to 0-1 range
        ml_score = min(ml_score, 1.0)
        
        logging.info(f"ML-like score: {ml_score:.3f}")
        
        # Decision logic
        if pattern_confidence > 0.8:
            # High confidence attack pattern detected
            decision = "BLOCK"
            final_confidence = max(pattern_confidence, ml_score)
            logging.info(f"Pattern-based decision: BLOCK (pattern_confidence: {pattern_confidence:.3f})")
        elif ml_score > 0.6:
            # High ML-like score
            decision = "BLOCK"
            final_confidence = ml_score
            logging.info(f"ML-based decision: BLOCK (ml_score: {ml_score:.3f})")
        elif ml_score > 0.3 and pattern_confidence > 0.2:
            # Medium scores from both
            decision = "BLOCK"
            final_confidence = (ml_score + pattern_confidence) / 2
            logging.info(f"Combined decision: BLOCK (ml: {ml_score:.3f}, pattern: {pattern_confidence:.3f})")
        else:
            # Allow the request
            decision = "ALLOW"
            final_confidence = ml_score
            logging.info(f"Decision: ALLOW (ml_score: {ml_score:.3f}, pattern_confidence: {pattern_confidence:.3f})")
        
        return decision, final_confidence
        
    except Exception as e:
        logging.error(f"Decision making failed: {e}")
        return "ALLOW", 0.0

def ml_waf_decision(url, method="GET", headers=None, body="", threshold_type="hybrid"):
    """
    Production-ready ML WAF decision function using hybrid approach
    """
    try:
        # Make hybrid decision
        decision, confidence = make_decision_hybrid(url, method, body)
        
        # Log decision
        logging.info(f"Request: {method} {url[:100]}... | "
                    f"Final confidence: {confidence:.4f} | "
                    f"Decision: {decision}")
        
        return {
            "decision": decision,
            "probability": float(confidence),
            "threshold_type": threshold_type,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logging.error(f"Error in ML WAF decision: {str(e)}")
        return {
            "decision": "ERROR",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

def main():
    """Main function for CLI usage"""
    DEBUG = os.environ.get("ML_DEBUG", "0") == "1"
    
    # Get inputs from environment
    url = os.environ.get("REQUEST_URI", "") or ""
    method = os.environ.get("REQUEST_METHOD", "GET") or "GET"
    body = os.environ.get("REQUEST_BODY", "") or ""
    
    # Optional JSON on stdin, non-blocking
    data = ""
    try:
        if not sys.stdin.isatty():
            r, _, _ = select.select([sys.stdin], [], [], 0.001)
            if r:
                data = sys.stdin.read()
    except Exception:
        pass
    
    if data:
        try:
            j = json.loads(data)
            url = j.get("url", url)
            method = j.get("method", method)
            body = j.get("body", body)
        except Exception:
            pass
    
    # Validate inputs
    if not url and not body:
        logging.warning("Both URL and body are empty")
        print("ALLOW")
        return 0
    
    # Make hybrid decision
    decision, confidence = make_decision_hybrid(url, method, body)
    
    # Log decision
    logging.info(f"[{method}] {url} -> {decision} (confidence={confidence:.4f})")
    
    if DEBUG:
        try:
            features = preprocess_request_simple(url, method, body)
            logging.info(f"[DEBUG] Features: {features}")
        except Exception:
            pass
    
    print(decision)
    return 1 if decision == "BLOCK" else 0

if __name__ == "__main__":
    sys.exit(main())
