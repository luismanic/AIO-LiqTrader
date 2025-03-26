import os
import time
import logging
import numpy as np
import mss
from PIL import Image
from datetime import datetime
from screeninfo import get_monitors
import pyautogui

logger = logging.getLogger("AIO-LiqTrader")

class ScreenshotManager:
    def __init__(self, config):
        self.config = config
        self.monitor_info = None
        
        # Create screenshots directory if it doesn't exist
        if self.config.save_screenshots and not os.path.exists(self.config.screenshots_dir):
            os.makedirs(self.config.screenshots_dir)
    
    def detect_monitors(self):
        """Detect and identify the target monitor"""
        monitors = get_monitors()
        logger.info(f"Detected {len(monitors)} monitors")
        
        for i, monitor in enumerate(monitors):
            logger.info(f"Monitor {i+1}: {monitor.name if hasattr(monitor, 'name') else 'Unknown'} - "
                       f"{monitor.width}x{monitor.height} at position ({monitor.x}, {monitor.y})")
            
            # Try to match the target monitor
            if hasattr(monitor, 'name') and self.config.target_monitor in monitor.name:
                self.monitor_info = monitor
                logger.info(f"Target monitor found: {monitor.name}")
                return
        
        # If named monitor not found, default to the primary monitor
        for monitor in monitors:
            if monitor.is_primary:
                self.monitor_info = monitor
                logger.info(f"Using primary monitor as target: {monitor.width}x{monitor.height}")
                return
            
        # If still no monitor found, use the first one
        if monitors:
            self.monitor_info = monitors[0]
            logger.info(f"Using first available monitor as target: {monitors[0].width}x{monitors[0].height}")
    
    def find_target_window(self):
        """Find the target Chrome window with CoinAnk BTC chart"""
        # This simplified version just looks for any window with the target keywords in the title
        try:
            # Get a list of all window titles
            titles = pyautogui.getAllTitles()
            
            # Filter for windows matching our target keywords
            matching_titles = [title for title in titles if any(keyword in title for keyword in self.config.target_window_keywords)]
            
            if not matching_titles:
                logger.warning("No matching Chrome windows found")
                return None
            
            logger.info(f"Found matching window: '{matching_titles[0]}'")
            return matching_titles[0]  # Return the first matching title
            
        except Exception as e:
            logger.error(f"Error finding target window: {e}", exc_info=True)
            return None
    
    def capture_screenshot(self):
        """Capture a screenshot of the target window or the entire screen"""
        try:
            # Try to find the target window
            target_window = self.find_target_window()
            
            if target_window:
                # If we found a target window, try to focus on it
                # Note: pyautogui can't reliably activate windows like win32gui can
                # So we'll just take a full screen screenshot
                logger.info(f"Taking full screen screenshot (target window: {target_window})")
            
            # Take full screen screenshot
            with mss.mss() as sct:
                if self.monitor_info:
                    # Use the detected monitor
                    monitor = {
                        "left": self.monitor_info.x,
                        "top": self.monitor_info.y,
                        "width": self.monitor_info.width,
                        "height": self.monitor_info.height
                    }
                else:
                    # Use the primary monitor
                    monitor = sct.monitors[1]  # monitor 1 is the primary monitor in mss
                
                screenshot = sct.grab(monitor)
                
                # Convert to PIL Image
                img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
                
                if self.config.save_screenshots:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filepath = os.path.join(self.config.screenshots_dir, f"screenshot_{timestamp}.png")
                    img.save(filepath)
                    logger.info(f"Screenshot saved to {filepath}")
                
                # Convert to numpy array for processing
                return np.array(img)
        
        except Exception as e:
            logger.error(f"Error capturing screenshot: {e}", exc_info=True)
            return None