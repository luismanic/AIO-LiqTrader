# AIO-LiqTrader: AI-Driven BTC Order Book Liquidity Trading Strategy
# Author: [Your Name]
# Date: March 26, 2025

import logging
import time
import os
from config import Config
from screenshot import ScreenshotManager
from analysis import AnalysisEngine
from trading import TradingEngine
from visualization import create_visualization_system

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("aio_liqtrader.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("AIO-LiqTrader")

class AIOLiqTrader:
    def __init__(self):
        self.config = Config()
        self.screenshot_manager = ScreenshotManager(self.config)
        self.analysis_engine = AnalysisEngine(self.config)
        self.trading_engine = TradingEngine(self.config)
        self.running = False
        self.last_screenshot_time = 0
        
        # Initialize visualization system
        self.viz_system = None
        try:
            self.viz_system = create_visualization_system(self.config)
            logger.info("Visualization system initialized")
        except Exception as e:
            logger.error(f"Failed to initialize visualization system: {e}", exc_info=True)
        
        logger.info("AIO-LiqTrader initialized")
    
    def start(self):
        """Start the trading system"""
        self.running = True
        
        # Detect monitors
        self.screenshot_manager.detect_monitors()
        
        # Main loop
        while self.running:
            try:
                current_time = time.time()
                # Take screenshot at specified interval
                if current_time - self.last_screenshot_time >= self.config.screenshot_interval:
                    self.process_cycle()
                    self.last_screenshot_time = current_time
                
                # Small sleep to prevent CPU overuse
                time.sleep(1)
            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received, stopping...")
                self.running = False
            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)
                time.sleep(5)  # Wait before retrying
    
    def process_cycle(self):
        """Process a complete analysis cycle"""
        logger.info("Starting analysis cycle")
        
        # Capture screenshot
        screenshot = self.screenshot_manager.capture_screenshot()
        if screenshot is None:
            logger.error("Failed to capture screenshot, skipping cycle")
            return
        
        # Get screenshot path for visualization
        latest_screenshot_path = None
        if hasattr(self.config, 'screenshots_dir') and hasattr(self.config, 'save_screenshots') and self.config.save_screenshots:
            screenshot_files = [f for f in os.listdir(self.config.screenshots_dir) if f.endswith('.png')]
            if screenshot_files:
                latest_screenshot_path = os.path.join(self.config.screenshots_dir, sorted(screenshot_files)[-1])
        
        # Analyze screenshot
        analysis_results = self.analysis_engine.analyze(screenshot)
        if not analysis_results:
            logger.warning("Analysis produced no results, skipping trading decisions")
            return
        
        # Generate visualizations
        if self.viz_system and analysis_results.get('valid', False):
            try:
                viz_paths = self.viz_system.create_visualization(analysis_results, latest_screenshot_path)
                if viz_paths:
                    image_path, html_path = viz_paths
                    logger.info(f"Generated visualization image: {image_path}")
                    logger.info(f"Generated interactive HTML report: {html_path}")
            except Exception as e:
                logger.error(f"Failed to generate visualizations: {e}", exc_info=True)
        
        # Make trading decisions
        trade_signal = self.trading_engine.evaluate(analysis_results)
        if trade_signal:
            logger.info(f"Trade signal generated: {trade_signal}")
        
        logger.info("Analysis cycle completed")

# If running as main script
if __name__ == "__main__":
    logger.info("Starting AIO-LiqTrader")
    trader = AIOLiqTrader()
    trader.start()