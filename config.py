import os
import json
import logging
import configparser

logger = logging.getLogger("AIO-LiqTrader")

class Config:
    def __init__(self, config_file='config.ini'):
        self.config = configparser.ConfigParser()
        
        # Default configuration
        self.screenshot_interval = 60  # seconds
        self.target_monitor = "DELL P2417H"
        self.target_window_keywords = ["CoinAnk", "BTC"]
        self.price_proximity_threshold = 250  # USD
        self.limit_order_proximity = 350  # USD
        self.liquidation_proximity = 300  # USD
        self.volume_analysis_bars = 20
        self.volume_surge_threshold = 1.5
        self.high_confidence_threshold = 85
        self.medium_confidence_threshold = 65
        self.save_screenshots = True
        self.screenshots_dir = "screenshots"
        self.debug_mode = False
        self.api_key = ""
        self.api_secret = ""
        self.discord_webhook = ""
        
        # Try to load configuration file
        if os.path.exists(config_file):
            try:
                self.config.read(config_file)
                self.load_from_config()
                logger.info(f"Configuration loaded from {config_file}")
            except Exception as e:
                logger.error(f"Error loading configuration: {e}")
        else:
            # Create default configuration file
            self.save_config(config_file)
            logger.info(f"Default configuration created at {config_file}")
    
    def load_from_config(self):
        """Load settings from config file"""
        if 'Settings' in self.config:
            settings = self.config['Settings']
            self.screenshot_interval = settings.getint('screenshot_interval', self.screenshot_interval)
            self.target_monitor = settings.get('target_monitor', self.target_monitor)
            self.target_window_keywords = json.loads(settings.get('target_window_keywords', json.dumps(self.target_window_keywords)))
            self.price_proximity_threshold = settings.getint('price_proximity_threshold', self.price_proximity_threshold)
            self.limit_order_proximity = settings.getint('limit_order_proximity', self.limit_order_proximity)
            self.liquidation_proximity = settings.getint('liquidation_proximity', self.liquidation_proximity)
            self.volume_analysis_bars = settings.getint('volume_analysis_bars', self.volume_analysis_bars)
            self.volume_surge_threshold = settings.getfloat('volume_surge_threshold', self.volume_surge_threshold)
            self.high_confidence_threshold = settings.getint('high_confidence_threshold', self.high_confidence_threshold)
            self.medium_confidence_threshold = settings.getint('medium_confidence_threshold', self.medium_confidence_threshold)
            self.save_screenshots = settings.getboolean('save_screenshots', self.save_screenshots)
            self.screenshots_dir = settings.get('screenshots_dir', self.screenshots_dir)
            self.debug_mode = settings.getboolean('debug_mode', self.debug_mode)
        
        if 'API' in self.config:
            api = self.config['API']
            self.api_key = api.get('api_key', self.api_key)
            self.api_secret = api.get('api_secret', self.api_secret)
            self.discord_webhook = api.get('discord_webhook', self.discord_webhook)
    
    def save_config(self, config_file):
        """Save current configuration to file"""
        if 'Settings' not in self.config:
            self.config.add_section('Settings')
        
        settings = self.config['Settings']
        settings['screenshot_interval'] = str(self.screenshot_interval)
        settings['target_monitor'] = self.target_monitor
        settings['target_window_keywords'] = json.dumps(self.target_window_keywords)
        settings['price_proximity_threshold'] = str(self.price_proximity_threshold)
        settings['limit_order_proximity'] = str(self.limit_order_proximity)
        settings['liquidation_proximity'] = str(self.liquidation_proximity)
        settings['volume_analysis_bars'] = str(self.volume_analysis_bars)
        settings['volume_surge_threshold'] = str(self.volume_surge_threshold)
        settings['high_confidence_threshold'] = str(self.high_confidence_threshold)
        settings['medium_confidence_threshold'] = str(self.medium_confidence_threshold)
        settings['save_screenshots'] = str(self.save_screenshots)
        settings['screenshots_dir'] = self.screenshots_dir
        settings['debug_mode'] = str(self.debug_mode)
        
        if 'API' not in self.config:
            self.config.add_section('API')
        
        api = self.config['API']
        api['api_key'] = self.api_key
        api['api_secret'] = self.api_secret
        api['discord_webhook'] = self.discord_webhook
        
        with open(config_file, 'w') as f:
            self.config.write(f)