import logging
import requests
import json
import utils
from datetime import datetime

logger = logging.getLogger("AIO-LiqTrader")

class TradingEngine:
    def __init__(self, config):
        self.config = config
        self.last_signal_time = 0
    
    def evaluate(self, analysis_results):
        """Evaluate analysis results and generate trade signals"""
        if not analysis_results.get('valid', False):
            return None
        
        # Extract components
        profile_lines = analysis_results.get('profile_lines', [])
        limit_orders = analysis_results.get('limit_orders', [])
        liquidation_zones = analysis_results.get('liquidation_zones', [])
        volume_analysis = analysis_results.get('volume_analysis', {})
        
        # Calculate confidence score
        confidence_score = self.calculate_confidence(
            profile_lines, limit_orders, liquidation_zones, volume_analysis
        )
        
        # Determine trade direction (long/short)
        trade_direction = self.determine_direction(profile_lines, limit_orders)
        
        # Generate signal if confidence exceeds threshold
        if confidence_score >= self.config.high_confidence_threshold:
            signal = {
                'timestamp': datetime.now().isoformat(),
                'confidence': confidence_score,
                'direction': trade_direction,
                'action': 'EXECUTE',
                'details': {
                    'profile_lines': len(profile_lines),
                    'limit_orders': len(limit_orders),
                    'liquidation_zones': len(liquidation_zones),
                    'volume_trend': volume_analysis.get('trend', 'neutral')
                }
            }
            
            # Send alert notification if configured
            self.send_alert(signal)
            
            return signal
        
        elif confidence_score >= self.config.medium_confidence_threshold:
            signal = {
                'timestamp': datetime.now().isoformat(),
                'confidence': confidence_score,
                'direction': trade_direction,
                'action': 'ALERT',
                'details': {
                    'profile_lines': len(profile_lines),
                    'limit_orders': len(limit_orders),
                    'liquidation_zones': len(liquidation_zones),
                    'volume_trend': volume_analysis.get('trend', 'neutral')
                }
            }
            
            return signal
        
        return None
    
    def calculate_confidence(self, profile_lines, limit_orders, liquidation_zones, volume_analysis):
        """Calculate a confidence score based on all signals"""
        try:
            # Base confidence starts at required level for a valid profile line
            base_confidence = 70  # Starting with 70 if core criteria are met
            
            # Initial check - must have profile lines and limit orders to be valid
            if not profile_lines or not limit_orders:
                return 0
            
            # Step 1: Assess order profile lines quality
            profile_line_count = len(profile_lines)
            
            # More profile lines slightly increase confidence
            profile_score = min(5 * profile_line_count, 10)  # Max +10 for profile lines
            
            # Step 2: Assess limit order clusters quality
            limit_order_count = len(limit_orders)
            
            # Calculate limit order score based on density
            if limit_order_count >= 10:
                limit_order_score = 10  # Max +10 points
            elif limit_order_count >= 5:
                limit_order_score = 5
            else:
                limit_order_score = 0
            
            # Step 3: Assess liquidation zones (supporting signal)
            liquidation_zone_count = len(liquidation_zones)
            
            if liquidation_zone_count >= 2:
                liquidation_score = 10  # Max +10 points for multiple liquidation zones
            elif liquidation_zone_count >= 1:
                liquidation_score = 5   # +5 points for a single liquidation zone
            else:
                liquidation_score = 0
            
            # Step 4: Assess volume behavior (supporting signal)
            volume_score = 0
            if volume_analysis.get('valid', False):
                # Trend decreasing is a strong signal
                if volume_analysis.get('trend_decreasing', False):
                    volume_score += 5
                
                # Opposite surge is an even stronger signal
                if volume_analysis.get('opposite_surge', False):
                    volume_score += 10
            
            # Maximum volume score is capped at +15
            volume_score = min(volume_score, 15)
            
            # Calculate total confidence score
            confidence_score = base_confidence + profile_score + limit_order_score + liquidation_score + volume_score
            
            # Ensure the score is between 0 and 100
            confidence_score = min(max(confidence_score, 0), 100)
            
            logger.info(f"Confidence score calculation: {base_confidence} (base) + " +
                       f"{profile_score} (profile) + {limit_order_score} (orders) + " +
                       f"{liquidation_score} (liq) + {volume_score} (volume) = {confidence_score}")
            
            return confidence_score
            
        except Exception as e:
            logger.error(f"Error calculating confidence score: {e}")
            return 0
    
    def determine_direction(self, profile_lines, limit_orders):
        """Determine trade direction (long/short) based on analysis"""
        try:
            if not profile_lines or not limit_orders:
                return "NONE"
            
            # Get current BTC price
            current_price = utils.get_live_btc_price()
            if current_price is None:
                logger.warning("Failed to get live BTC price, can't determine direction")
                return "NONE"
            
            # Find the profile line closest to current price
            closest_line = min(profile_lines, key=lambda x: abs(x['price'] - current_price))
            
            # Determine direction based on price difference
            price_diff = closest_line['price'] - current_price
            
            # Look at the limit orders around this line to confirm direction
            buy_orders = [order for order in limit_orders if order['type'] == 'buy']
            sell_orders = [order for order in limit_orders if order['type'] == 'sell']
            
            # Check buy/sell order strength
            buy_strength = sum(order['length'] for order in buy_orders) if buy_orders else 0
            sell_strength = sum(order['length'] for order in sell_orders) if sell_orders else 0
            
            if price_diff < 0:
                # Current price is above the line, suggesting a short
                # But confirm with order strength
                if sell_strength > buy_strength:
                    logger.info(f"Short signal: price at ${current_price:.2f} is above " +
                               f"profile line at ${closest_line['price']:.2f} and " +
                               f"sell orders ({sell_strength}) > buy orders ({buy_strength})")
                    return "SHORT"
            else:
                # Current price is below the line, suggesting a long
                # But confirm with order strength
                if buy_strength > sell_strength:
                    logger.info(f"Long signal: price at ${current_price:.2f} is below " +
                               f"profile line at ${closest_line['price']:.2f} and " +
                               f"buy orders ({buy_strength}) > sell orders ({sell_strength})")
                    return "LONG"
            
            # If order strengths contradict the price difference or are similar,
            # use the price difference as the deciding factor
            if abs(price_diff) > 50:  # If difference is significant
                if price_diff < 0:
                    logger.info(f"Short signal based on price difference: ${price_diff:.2f}")
                    return "SHORT"
                else:
                    logger.info(f"Long signal based on price difference: ${price_diff:.2f}")
                    return "LONG"
            
            # If we reach here, the signal is unclear
            logger.info("No clear direction signal determined")
            return "NONE"
            
        except Exception as e:
            logger.error(f"Error determining trade direction: {e}")
            return "NONE"
    
    def send_alert(self, signal):
        """Send alert via Discord webhook if configured"""
        if not self.config.discord_webhook:
            return
        
        try:
            webhook_url = self.config.discord_webhook
            data = {
                "content": f"🚨 **AIO-LiqTrader Alert** 🚨",
                "embeds": [{
                    "title": f"{signal['direction']} Signal - Confidence: {signal['confidence']}%",
                    "description": f"Action: {signal['action']}",
                    "color": 3447003 if signal['direction'] == 'LONG' else 15158332,
                    "fields": [
                        {"name": "Profile Lines", "value": signal['details']['profile_lines'], "inline": True},
                        {"name": "Limit Orders", "value": signal['details']['limit_orders'], "inline": True},
                        {"name": "Liquidation Zones", "value": signal['details']['liquidation_zones'], "inline": True},
                        {"name": "Volume Trend", "value": signal['details']['volume_trend'], "inline": True},
                    ],
                    "timestamp": signal['timestamp']
                }]
            }
            
            headers = {'Content-Type': 'application/json'}
            response = requests.post(webhook_url, data=json.dumps(data), headers=headers)
            response.raise_for_status()
            
            logger.info(f"Alert sent to Discord webhook")
            
        except Exception as e:
            logger.error(f"Failed to send Discord alert: {e}")