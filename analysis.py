import logging
import cv2
import numpy as np
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
import requests
import utils
import os

logger = logging.getLogger("AIO-LiqTrader")

class AnalysisEngine:
    def __init__(self, config):
        self.config = config
        # Store the latest pixel to price ratio for reuse across analysis steps
        self.pixel_to_price_ratio = None
    
    def analyze(self, screenshot):
        """Main analysis process that runs all detection steps"""
        try:
            # Process the screenshot into regions of interest
            processed_img = self.preprocess_image(screenshot)
            if processed_img is None:
                return None
            
            # Calculate the pixel-to-price ratio using Y-axis values
            self.pixel_to_price_ratio = self.calculate_pixel_price_ratio(processed_img)
            if self.pixel_to_price_ratio is None:
                logger.error("Failed to calculate pixel-to-price ratio")
                return None
            
            # 1. Detect order profile lines and check price proximity
            profile_lines = self.detect_order_profile_lines(processed_img)
            if not profile_lines:
                logger.info("No significant order profile lines detected")
                return {"valid": False, "reason": "No order profile lines"}
            
            # 2. Detect limit order clusters
            limit_orders = self.detect_limit_order_clusters(processed_img, profile_lines)
            
            # 3. Detect liquidation zones
            liquidation_zones = self.detect_liquidation_zones(processed_img, profile_lines)
            
            # 4. Analyze volume behavior
            volume_analysis = self.analyze_volume_behavior(processed_img)
            
            # Combine results
            results = {
                "valid": True,
                "profile_lines": profile_lines,
                "limit_orders": limit_orders,
                "liquidation_zones": liquidation_zones,
                "volume_analysis": volume_analysis,
                "pixel_to_price_ratio": self.pixel_to_price_ratio
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error in analysis: {e}", exc_info=True)
            return None
    
    def preprocess_image(self, screenshot):
        """Prepare the image for analysis by segmenting into regions of interest"""
        try:
            # Convert image to HSV color space for better color filtering
            hsv_img = cv2.cvtColor(screenshot, cv2.COLOR_RGB2HSV)
            
            # Define regions of interest based on typical chart layout
            height, width = screenshot.shape[:2]
            
            # Store regions for downstream analysis
            result = {
                'original': screenshot,
                'hsv': hsv_img,
                'regions': {
                    'y_axis_left': screenshot[:, 0:int(width * 0.1)],  # Left 10% for Y-axis price labels
                    'y_axis_right': screenshot[:, int(width * 0.9):],  # Right 10% for liquidation heatmap
                    'chart_main': screenshot[:, int(width * 0.1):int(width * 0.9)],  # Middle 80% for chart
                    'volume_bars': screenshot[int(height * 0.8):, :],  # Bottom 20% for volume bars
                }
            }
            
            return result
        
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}", exc_info=True)
            return None
    
    def calculate_pixel_price_ratio(self, processed_img):
        """Calculate the ratio of pixels to price units using Y-axis labels"""
        try:
            # Extract the Y-axis region
            y_axis = processed_img['regions']['y_axis_left']
            
            # Use OCR to detect price values along the Y-axis
            # This is a simplified version - in practice, you'd need more robust OCR
            # with preprocessing to enhance text detection
            gray = cv2.cvtColor(y_axis, cv2.COLOR_RGB2GRAY)
            
            # Apply thresholding to improve OCR
            _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
            
            # Use pytesseract to extract text
            text = pytesseract.image_to_string(thresh, config='--psm 6')
            
            # Parse and extract price values with their y-positions
            price_positions = []
            for line in text.splitlines():
                try:
                    # Look for numbers that might be prices
                    if any(c.isdigit() for c in line):
                        # Clean the text and convert to a number
                        price_text = ''.join(c for c in line if c.isdigit() or c == '.' or c == ',')
                        price_text = price_text.replace(',', '')
                        price = float(price_text)
                        
                        # Estimate the y-position of this price
                        # This is complex and would need refinement in a real implementation
                        y_pos = text.splitlines().index(line) * 20  # Simplified estimate
                        
                        price_positions.append((price, y_pos))
                except:
                    continue
            
            # Need at least 2 prices to calculate ratio
            if len(price_positions) < 2:
                logger.warning("Not enough price points detected for pixel-to-price calculation")
                # Fallback to a hardcoded ratio (needs calibration)
                return 5.0  # 5 USD per pixel
            
            # Sort by y-position (smaller y values are higher on the screen)
            price_positions.sort(key=lambda x: x[1])
            
            # Calculate ratio from the first and last detected prices
            first_price, first_y = price_positions[0]
            last_price, last_y = price_positions[-1]
            
            pixel_distance = last_y - first_y
            price_difference = first_price - last_price  # Prices decrease as y increases
            
            if price_difference <= 0:
                logger.warning("Invalid price difference calculated")
                return 5.0  # Fallback to default
            
            ratio = price_difference / pixel_distance  # USD per pixel
            logger.info(f"Calculated pixel-to-price ratio: {ratio:.2f} USD/pixel")
            
            return ratio
            
        except Exception as e:
            logger.error(f"Error calculating pixel to price ratio: {e}", exc_info=True)
            return None
    
    def detect_order_profile_lines(self, processed_img):
        """Detect significant horizontal white or yellow order profile lines with enhanced filtering"""
        try:
            # Extract the main chart area for line detection
            chart_img = processed_img['regions']['chart_main']
            hsv_img = cv2.cvtColor(chart_img, cv2.COLOR_RGB2HSV)
            height, width = chart_img.shape[:2]
            
            # If debug mode is enabled, save intermediate images
            if hasattr(self.config, 'debug_mode') and self.config.debug_mode:
                debug_path = os.path.join(self.config.screenshots_dir, 'debug')
                os.makedirs(debug_path, exist_ok=True)
                
                # Save the chart region for inspection
                cv2.imwrite(os.path.join(debug_path, 'chart_region.png'), chart_img)
            
            # Use a much wider yellow range to detect any yellowish colors
            lower_yellow = np.array([20, 30, 100])  # More inclusive lower bounds
            upper_yellow = np.array([70, 255, 255]) # Wider hue range
            yellow_mask = cv2.inRange(hsv_img, lower_yellow, upper_yellow)
            
            # Use a more inclusive red range
            lower_red1 = np.array([0, 50, 50])
            upper_red1 = np.array([15, 255, 255])
            lower_red2 = np.array([165, 50, 50])
            upper_red2 = np.array([180, 255, 255])
            red_mask1 = cv2.inRange(hsv_img, lower_red1, upper_red1)
            red_mask2 = cv2.inRange(hsv_img, lower_red2, upper_red2)
            red_mask = cv2.bitwise_or(red_mask1, red_mask2)
            
            # Include orange range as many trading platforms use orange
            lower_orange = np.array([15, 50, 100])
            upper_orange = np.array([25, 255, 255])
            orange_mask = cv2.inRange(hsv_img, lower_orange, upper_orange)
            
            # White color range (keep this for potential white lines)
            lower_white = np.array([0, 0, 200])
            upper_white = np.array([180, 30, 255])
            white_mask = cv2.inRange(hsv_img, lower_white, upper_white)
            
            # Include all potential line colors
            line_mask = cv2.bitwise_or(cv2.bitwise_or(cv2.bitwise_or(yellow_mask, red_mask), white_mask), orange_mask)
            
            # Save color masks if in debug mode
            if hasattr(self.config, 'debug_mode') and self.config.debug_mode:
                debug_path = os.path.join(self.config.screenshots_dir, 'debug')
                os.makedirs(debug_path, exist_ok=True)
                
                # Save original chart
                cv2.imwrite(os.path.join(debug_path, 'chart_region.png'), chart_img)
                
                # Save color masks
                cv2.imwrite(os.path.join(debug_path, 'yellow_mask.png'), yellow_mask)
                cv2.imwrite(os.path.join(debug_path, 'red_mask.png'), red_mask)
                cv2.imwrite(os.path.join(debug_path, 'white_mask.png'), white_mask)
                cv2.imwrite(os.path.join(debug_path, 'orange_mask.png'), orange_mask)
                
                # Save combined mask
                cv2.imwrite(os.path.join(debug_path, 'line_mask.png'), line_mask)
            
            # Apply more gentle morphological operations to preserve thin lines
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
            line_mask = cv2.morphologyEx(line_mask, cv2.MORPH_CLOSE, kernel)
            
            # Filter small noise but be less aggressive
            min_area = max(width // 20, 50)  # Minimum of 50 pixels or 1/20th of width
            line_mask = utils.filter_noise(line_mask, min_area=min_area)
            
            # Save processed mask if in debug mode
            if hasattr(self.config, 'debug_mode') and self.config.debug_mode:
                cv2.imwrite(os.path.join(debug_path, 'processed_line_mask.png'), line_mask)
            
            # Detect horizontal lines using Hough transform with less strict parameters
            lines = cv2.HoughLinesP(
                line_mask, 
                rho=1,
                theta=np.pi/180,
                threshold=width//30,       # Even lower threshold
                minLineLength=width//10,   # Detect even shorter lines
                maxLineGap=100             # Allow even larger gaps
            )
            
            if lines is None:
                logger.info("No order profile lines detected")
                return []
            
            # Extract horizontal lines and determine their color
            detected_lines = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # Ensure the line is mostly horizontal
                if abs(y2 - y1) > 5:  # Allow a small slope
                    continue
                
                # Get average Y position
                y_avg = (y1 + y2) // 2
                
                # Determine color at this position
                color_samples = []
                for x in range(x1, x2, 10):  # Sample every 10 pixels
                    if 0 <= x < width and 0 <= y_avg < height:
                        color_samples.append(hsv_img[y_avg, x])
                
                if not color_samples:
                    continue
                    
                avg_color = np.mean(color_samples, axis=0)
                # Check if color is yellow, red, or white based on the specific chart colors
                if 20 <= avg_color[0] <= 70 and avg_color[1] >= 30:
                    color = "yellow"
                elif ((0 <= avg_color[0] <= 15) or (165 <= avg_color[0] <= 180)) and avg_color[1] >= 50:
                    color = "red"
                elif 15 <= avg_color[0] <= 25 and avg_color[1] >= 50:
                    color = "orange"
                else:
                    color = "white"
                
                # Calculate price at this Y position
                # Y position is relative to chart_main, so we need to adjust
                # to get position in the original image
                y_position = y_avg + int(height * 0.1)  # Assuming chart_main starts at 10% of height
                estimated_price = self.calculate_price_from_y_position(y_position, processed_img['original'].shape[0])
                
                # Measure length of the line
                length = x2 - x1
                
                # Add to detected lines (will filter later based on average)
                detected_lines.append({
                    'y_position': y_position,
                    'length': length,
                    'color': color,
                    'price': estimated_price,
                    'x1': x1,
                    'x2': x2,
                    'y1': y1,
                    'y2': y2
                })
            
            # If no lines detected, return empty list
            if not detected_lines:
                logger.info("No order profile lines detected after initial filter")
                return []
            
            # Calculate average line length
            total_length = sum(line['length'] for line in detected_lines)
            average_length = total_length / len(detected_lines)
            logger.info(f"Average order profile line length: {average_length:.2f} pixels")
            
            # Sort lines by length (descending)
            sorted_by_length = sorted(detected_lines, key=lambda x: x['length'], reverse=True)
            
            # Keep only the top 90% of lines by length that are above the average
            top_index = max(1, int(0.9 * len(sorted_by_length)))
            top_lines = sorted_by_length[:top_index]
            
            # Filter for only above-average lines
            above_average_lines = [line for line in top_lines if line['length'] > average_length]
            
            if hasattr(self.config, 'debug_mode') and self.config.debug_mode:
                # Create a visualization of the filtered lines
                debug_img = chart_img.copy()
                for line in above_average_lines:
                    color_bgr = (0, 255, 255)  # Yellow in BGR
                    if line['color'] == 'white':
                        color_bgr = (255, 255, 255)  # White in BGR
                    elif line['color'] == 'red':
                        color_bgr = (0, 0, 255)  # Red in BGR
                    elif line['color'] == 'orange':
                        color_bgr = (0, 165, 255)  # Orange in BGR
                    
                    cv2.line(debug_img, (line['x1'], line['y1']), (line['x2'], line['y2']), color_bgr, 2)
                    cv2.putText(debug_img, f"L: {line['length']}", (line['x2'] + 5, line['y1'] + 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 1)
                
                cv2.imwrite(os.path.join(debug_path, 'filtered_lines.png'), debug_img)
            
            logger.info(f"Filtered {len(above_average_lines)} significant order profile lines out of {len(detected_lines)} total")
            
            # Calculate line density score (for use in trading signals)
            if above_average_lines and average_length > 0:
                max_length = max(line['length'] for line in above_average_lines)
                min_length = min(line['length'] for line in above_average_lines)
                
                # Calculate score for each line based on its length relative to average
                for line in above_average_lines:
                    # Normalized score between 0 and 1
                    if max_length == min_length:
                        line['strength_score'] = 1.0  # Avoid division by zero
                    else:
                        line['strength_score'] = (line['length'] - min_length) / (max_length - min_length)
                    
                    # Categorize line strength
                    if line['strength_score'] >= 0.75:
                        line['strength'] = 'very_strong'
                    elif line['strength_score'] >= 0.5:
                        line['strength'] = 'strong'
                    elif line['strength_score'] >= 0.25:
                        line['strength'] = 'moderate'
                    else:
                        line['strength'] = 'weak'
                    
                    logger.info(f"{line['color'].capitalize()} line at ${line['price']:.2f} - length: {line['length']} px, " +
                               f"strength: {line['strength']} ({line['strength_score']:.2f})")
            
            # Sort lines by y-position
            above_average_lines.sort(key=lambda x: x['y_position'])
            
            # Get current BTC price
            current_btc_price = utils.get_live_btc_price()
            if current_btc_price is None:
                logger.warning("Failed to get live BTC price, using estimated price from chart")
                current_btc_price = self.estimate_current_price_from_chart(processed_img)
            
            # Filter lines based on proximity to current price with a generous threshold
            price_threshold = getattr(self.config, 'price_proximity_threshold', 5000)  # Default to 5000 if not in config
            qualified_lines = []
            
            for line in above_average_lines:
                price_diff = abs(line['price'] - current_btc_price)
                
                if price_diff <= price_threshold:
                    line['price_diff'] = price_diff
                    qualified_lines.append(line)
                    logger.info(f"Order profile line detected at ${line['price']:.2f} " +
                               f"(${price_diff:.2f} from current price)")
            
            return qualified_lines
                
        except Exception as e:
            logger.error(f"Error detecting order profile lines: {e}", exc_info=True)
            return []
    
    def calculate_price_from_y_position(self, y_position, total_height):
        """Convert a y-position in pixels to a price estimate"""
        if self.pixel_to_price_ratio is None:
            logger.warning("No pixel-to-price ratio available")
            return 0
        
        # On charts, lower y-position (higher on screen) = higher price
        # We use the midpoint of the chart as a reference
        center_y = total_height // 2
        
        # Get price difference from center
        pixel_diff = center_y - y_position
        price_diff = pixel_diff * self.pixel_to_price_ratio
        
        # We need a base price to add the difference to
        # This is where we'd ideally read from the chart or use API
        # For now, using a mock value as placeholder
        base_price = 90000  # This should be replaced with actual detected price
        estimated_price = base_price + price_diff
        
        return estimated_price
    
    def estimate_current_price_from_chart(self, processed_img):
        """Estimate the current BTC price from the chart"""
        # In a real implementation, this would use OCR to read the price
        # from the chart's right axis or price label
        # For now, returning a placeholder value
        return 90000  # Default to $90k if we can't determine it
    
    def detect_limit_order_clusters(self, processed_img, profile_lines):
        """Detect large limit order blocks near profile lines"""
        try:
            if not profile_lines:
                return []
            
            # Extract the main chart area for detection
            chart_img = processed_img['regions']['chart_main']
            hsv_img = cv2.cvtColor(chart_img, cv2.COLOR_RGB2HSV)
            height, width = chart_img.shape[:2]
            
            # Define color ranges for green and red blocks in HSV
            # Green color range (buy orders)
            lower_green = np.array([35, 100, 50])
            upper_green = np.array([85, 255, 255])
            green_mask = cv2.inRange(hsv_img, lower_green, upper_green)
            
            # Red color range (sell orders)
            lower_red1 = np.array([0, 100, 50])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([160, 100, 50])
            upper_red2 = np.array([180, 255, 255])
            
            red_mask1 = cv2.inRange(hsv_img, lower_red1, upper_red1)
            red_mask2 = cv2.inRange(hsv_img, lower_red2, upper_red2)
            red_mask = cv2.bitwise_or(red_mask1, red_mask2)
            
            # Apply morphological operations to enhance horizontal blocks
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
            green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
            red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
            
            # Filter small noise
            green_mask = utils.filter_noise(green_mask, min_area=100)
            red_mask = utils.filter_noise(red_mask, min_area=100)
            
            # Detect contours for green blocks (buy orders)
            green_contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Detect contours for red blocks (sell orders)
            red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Process all contours
            all_blocks = []
            
            # Process green contours (buy orders)
            for contour in green_contours:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Horizontal blocks should be wider than tall
                if w <= h:
                    continue
                    
                # Measure length and calculate y-position
                y_position = y + (h // 2) + int(height * 0.1)  # Adjust for chart_main position
                
                # Calculate price at this position
                price = self.calculate_price_from_y_position(y_position, processed_img['original'].shape[0])
                
                # Add to blocks list
                all_blocks.append({
                    'y_position': y_position,
                    'length': w,
                    'height': h,
                    'type': 'buy',
                    'price': price
                })
            
            # Process red contours (sell orders)
            for contour in red_contours:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Horizontal blocks should be wider than tall
                if w <= h:
                    continue
                    
                # Measure length and calculate y-position
                y_position = y + (h // 2) + int(height * 0.1)  # Adjust for chart_main position
                
                # Calculate price at this position
                price = self.calculate_price_from_y_position(y_position, processed_img['original'].shape[0])
                
                # Add to blocks list
                all_blocks.append({
                    'y_position': y_position,
                    'length': w,
                    'height': h,
                    'type': 'sell',
                    'price': price
                })
            
            # Calculate average block length to identify significant ones
            if all_blocks:
                avg_length = utils.calculate_average_length(all_blocks)
                significant_threshold = max(avg_length * 1.5, width * 0.05)  # At least 5% of chart width
                
                # Filter to keep only significant blocks
                significant_blocks = [block for block in all_blocks if block['length'] >= significant_threshold]
                
                # Log the counts
                logger.info(f"Detected {len(significant_blocks)} significant limit order blocks " +
                           f"out of {len(all_blocks)} total")
            else:
                significant_blocks = []
                logger.info("No limit order blocks detected")
            
            # Now check which blocks are near profile lines
            proximity_threshold = getattr(self.config, 'limit_order_proximity', 2500)  # Default to 2500 if not in config
            qualified_blocks = []
            
            for profile_line in profile_lines:
                blocks_near_line = []
                
                for block in significant_blocks:
                    price_diff = abs(block['price'] - profile_line['price'])
                    
                    if price_diff <= proximity_threshold:
                        block['price_diff'] = price_diff
                        blocks_near_line.append(block)
                
                # If we found at least 5 blocks near this line, it's a qualified cluster
                if len(blocks_near_line) >= 5:
                    logger.info(f"Found {len(blocks_near_line)} limit orders near profile line at ${profile_line['price']:.2f}")
                    qualified_blocks.extend(blocks_near_line)
            
            return qualified_blocks
            
        except Exception as e:
            logger.error(f"Error detecting limit order clusters: {e}", exc_info=True)
            return []
    
    def detect_liquidation_zones(self, processed_img, profile_lines):
        """Detect liquidation zones near profile lines"""
        try:
            if not profile_lines:
                return []
            
            # Extract the right-side liquidation heatmap area
            liq_img = processed_img['regions']['y_axis_right']
            gray = cv2.cvtColor(liq_img, cv2.COLOR_RGB2GRAY)
            height, width = liq_img.shape[:2]
            
            # Use edge detection to find horizontal bars
            edges = cv2.Canny(gray, 50, 150)
            
            # Apply morphological operations to enhance horizontal lines
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 1))
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
            
            # Detect horizontal lines using Hough transform
            lines = cv2.HoughLinesP(
                edges, 
                rho=1,
                theta=np.pi/180,
                threshold=20,
                minLineLength=width//4,
                maxLineGap=5
            )
            
            if lines is None:
                logger.info("No liquidation zones detected")
                return []
            
            # Extract horizontal lines
            liquidation_zones = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # Ensure the line is horizontal
                if abs(y2 - y1) > 3:
                    continue
                
                # Get average Y position
                y_avg = (y1 + y2) // 2
                
                # Convert to position in original image
                y_position = y_avg
                
                # Calculate price at this Y position
                estimated_price = self.calculate_price_from_y_position(y_position, processed_img['original'].shape[0])
                
                # Measure length of the bar
                length = x2 - x1
                
                # Add to detected liquidation zones
                liquidation_zones.append({
                    'y_position': y_position,
                    'length': length,
                    'price': estimated_price
                })
            
            # Calculate average length to identify significant zones
            if liquidation_zones:
                avg_length = utils.calculate_average_length(liquidation_zones)
                significant_threshold = max(avg_length * 1.5, width * 0.2)  # At least 20% of width
                
                # Filter to keep only significant zones
                significant_zones = [zone for zone in liquidation_zones if zone['length'] >= significant_threshold]
                
                logger.info(f"Detected {len(significant_zones)} significant liquidation zones " +
                          f"out of {len(liquidation_zones)} total")
            else:
                significant_zones = []
            
            # Check which zones are near profile lines
            proximity_threshold = getattr(self.config, 'liquidation_proximity', 2500)  # Default to 2500 if not in config
            qualified_zones = []
            
            for profile_line in profile_lines:
                for zone in significant_zones:
                    price_diff = abs(zone['price'] - profile_line['price'])
                    
                    if price_diff <= proximity_threshold:
                        zone['price_diff'] = price_diff
                        if zone not in qualified_zones:  # Avoid duplicates
                            qualified_zones.append(zone)
                            logger.info(f"Liquidation zone detected at ${zone['price']:.2f} " +
                                      f"(${price_diff:.2f} from profile line)")
            
            return qualified_zones
            
        except Exception as e:
            logger.error(f"Error detecting liquidation zones: {e}", exc_info=True)
            return []
    
    def analyze_volume_behavior(self, processed_img):
        """Analyze volume trends and potential reversal signals"""
        try:
            # Extract the volume bar region
            volume_img = processed_img['regions']['volume_bars']
            hsv_img = cv2.cvtColor(volume_img, cv2.COLOR_RGB2HSV)
            height, width = volume_img.shape[:2]
            
            # Define color ranges for green and red volume bars in HSV
            # Green bars (buying volume)
            lower_green = np.array([35, 50, 50])
            upper_green = np.array([85, 255, 255])
            green_mask = cv2.inRange(hsv_img, lower_green, upper_green)
            
            # Red bars (selling volume)
            lower_red1 = np.array([0, 50, 50])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([160, 50, 50])
            upper_red2 = np.array([180, 255, 255])
            
            red_mask1 = cv2.inRange(hsv_img, lower_red1, upper_red1)
            red_mask2 = cv2.inRange(hsv_img, lower_red2, upper_red2)
            red_mask = cv2.bitwise_or(red_mask1, red_mask2)
            
            # Detect volume bar contours
            green_contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Extract volume bars
            volume_bars = []
            
            # Process green contours (buy volume)
            for contour in green_contours:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Volume bars should be taller than wide
                if h <= w:
                    continue
                    
                # Calculate position and height
                x_position = x + (w // 2)
                height_pixels = h
                
                # Add to volume bars list
                volume_bars.append({
                    'x_position': x_position,
                    'height': height_pixels,
                    'type': 'buy',
                    'area': cv2.contourArea(contour)
                })
            
            # Process red contours (sell volume)
            for contour in red_contours:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Volume bars should be taller than wide
                if h <= w:
                    continue
                    
                # Calculate position and height
                x_position = x + (w // 2)
                height_pixels = h
                
                # Add to volume bars list
                volume_bars.append({
                    'x_position': x_position,
                    'height': height_pixels,
                    'type': 'sell',
                    'area': cv2.contourArea(contour)
                })
            
            # Sort bars by x-position (chronological order)
            volume_bars.sort(key=lambda x: x['x_position'])
            
            if len(volume_bars) < getattr(self.config, 'volume_analysis_bars', 20):
                logger.warning(f"Not enough volume bars detected for analysis " +
                             f"({len(volume_bars)} found, {getattr(self.config, 'volume_analysis_bars', 20)} needed)")
                return {'valid': False}
            
            # Keep the most recent bars
            recent_bars = volume_bars[-getattr(self.config, 'volume_analysis_bars', 20):]
            
            # Calculate average volume height
            avg_height = sum(bar['height'] for bar in recent_bars) / len(recent_bars)
            
            # Determine the current trend based on most recent bars
            recent_types = [bar['type'] for bar in recent_bars[-5:]]  # Last 5 bars
            if recent_types.count('buy') > recent_types.count('sell'):
                current_trend = 'uptrend'
            elif recent_types.count('sell') > recent_types.count('buy'):
                current_trend = 'downtrend'
            else:
                current_trend = 'neutral'
            
            # Analyze trends based on the whitepaper's two scenarios
            
            # Scenario 1: Check for decreasing trend volume
            trend_decreasing = False
            if current_trend in ['uptrend', 'downtrend']:
                # Get recent bars matching the trend
                trend_bars = [bar for bar in recent_bars[-5:] if 
                             (current_trend == 'uptrend' and bar['type'] == 'buy') or
                             (current_trend == 'downtrend' and bar['type'] == 'sell')]
                
                if len(trend_bars) >= 3:
                    # Check if heights are decreasing
                    heights = [bar['height'] for bar in trend_bars]
                    if all(heights[i] >= heights[i+1] for i in range(len(heights)-1)):
                        trend_decreasing = True
                        logger.info(f"Detected decreasing {current_trend} volume")
            
            # Scenario 2: Check for opposite volume surge
            opposite_surge = False
            last_bar = recent_bars[-1]
            if ((current_trend == 'downtrend' and last_bar['type'] == 'buy') or
                (current_trend == 'uptrend' and last_bar['type'] == 'sell')):
                
                # Check if the last bar exceeds the surge threshold
                surge_threshold = getattr(self.config, 'volume_surge_threshold', 1.5)
                if last_bar['height'] >= avg_height * surge_threshold:
                    opposite_surge = True
                    logger.info(f"Detected opposite volume surge: {last_bar['type']} bar " +
                               f"in {current_trend}")
            
            # Compile results
            volume_analysis = {
                'valid': True,
                'bars_analyzed': len(recent_bars),
                'current_trend': current_trend,
                'trend_decreasing': trend_decreasing,
                'opposite_surge': opposite_surge,
                'confidence_boost': trend_decreasing or opposite_surge
            }
            
            return volume_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing volume behavior: {e}", exc_info=True)
            return {'valid': False}
            
            # Keep the most recent bars
            recent_bars = volume_bars[-getattr(self.config, 'volume_analysis_bars', 20):]
            
            # Calculate average volume height
            avg_height = sum(bar['height'] for bar in recent_bars) / len(recent_bars)
            
            # Determine the current trend based on most recent bars
            recent_types = [bar['type'] for bar in recent_bars[-5:]]  # Last 5 bars
            if recent_types.count('buy') > recent_types.count('sell'):
                current_trend = 'uptrend'
            elif recent_types.count('sell') > recent_types.count('buy'):
                current_trend = 'downtrend'
            else:
                current_trend = 'neutral'
            
            # Analyze trends based on the whitepaper's two scenarios
            
            # Scenario 1: Check for decreasing trend volume
            trend_decreasing = False
            if current_trend in ['uptrend', 'downtrend']:
                # Get recent bars matching the trend
                trend_bars = [bar for bar in recent_bars[-5:] if 
                             (current_trend == 'uptrend' and bar['type'] == 'buy') or
                             (current_trend == 'downtrend' and bar['type'] == 'sell')]
                
                if len(trend_bars) >= 3:
                    # Check if heights are decreasing
                    heights = [bar['height'] for bar in trend_bars]
                    if all(heights[i] >= heights[i+1] for i in range(len(heights)-1)):
                        trend_decreasing = True
                        logger.info(f"Detected decreasing {current_trend} volume")