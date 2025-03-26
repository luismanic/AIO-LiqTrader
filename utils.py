import logging
import cv2
import numpy as np
import requests
from datetime import datetime

logger = logging.getLogger("AIO-LiqTrader")

def get_live_btc_price():
    """Fetch current BTC price from an API"""
    try:
        response = requests.get('https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd')
        data = response.json()
        price = data['bitcoin']['usd']
        return price
    except Exception as e:
        logger.error(f"Error fetching BTC price: {e}")
        return None

def draw_debug_image(screenshot, analysis_results, output_path):
    """Create a visualization of detection results for debugging"""
    if not analysis_results:
        return False
    
    # Make a copy to avoid modifying the original
    debug_img = screenshot.copy()
    
    # Draw detected profile lines
    for line in analysis_results.get('profile_lines', []):
        y = line['y_position']
        color = (255, 255, 0) if line['color'] == 'yellow' else (255, 255, 255)
        cv2.line(debug_img, (0, y), (debug_img.shape[1], y), color, 2)
        
        # Add price label
        cv2.putText(debug_img, f"${line['price']:.2f}", (10, y-5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # Draw limit order blocks
    for order in analysis_results.get('limit_orders', []):
        y = order['y_position']
        width = order['length']
        color = (0, 255, 0) if order['type'] == 'buy' else (0, 0, 255)
        cv2.rectangle(debug_img, (0, y-5), (width, y+5), color, -1)
    
    # Draw liquidation zones
    for zone in analysis_results.get('liquidation_zones', []):
        y = zone['y_position']
        cv2.line(debug_img, (debug_img.shape[1]-100, y), 
                 (debug_img.shape[1], y), (255, 0, 255), 3)
    
    # Save the debug image
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = f"{output_path}/debug_{timestamp}.png"
        cv2.imwrite(filepath, cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR))
        return True
    except Exception as e:
        logger.error(f"Error saving debug image: {e}")
        return False

def filter_noise(binary_image, min_area=100):
    """Filter out small noise from binary images"""
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered = np.zeros_like(binary_image)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_area:
            cv2.drawContours(filtered, [contour], -1, 255, -1)
    
    return filtered

def calculate_average_length(bars):
    """Calculate the average length of horizontal bars"""
    if not bars:
        return 0
    
    total_length = sum(bar['length'] for bar in bars)
    return total_length / len(bars)