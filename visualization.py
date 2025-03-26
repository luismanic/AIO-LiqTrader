import os
import json
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import numpy as np
from datetime import datetime

class VisualizationEngine:
    """
    Visualization engine for AIO-LiqTrader's analysis results
    Provides clear visual representations of order profile lines, limit orders, and trading signals
    """
    
    def __init__(self, config):
        self.config = config
        self.visualization_dir = os.path.join('visualizations')
        os.makedirs(self.visualization_dir, exist_ok=True)
        
        # Color mappings for different strength levels
        self.strength_colors = {
            'very_strong': '#FF0000',  # Red
            'strong': '#FF6600',       # Orange
            'moderate': '#FFCC00',     # Yellow
            'weak': '#00CC00'          # Green
        }
        
        # Color mappings for different line types
        self.line_type_colors = {
            'yellow': '#FFD700',
            'white': '#FFFFFF',
            'red': '#FF0000',
            'orange': '#FFA500'
        }
    
    def create_visualization(self, analysis_results, screenshot_path=None):
        """
        Create comprehensive visualization of analysis results
        
        Args:
            analysis_results: Dictionary containing analysis results
            screenshot_path: Path to the screenshot being analyzed (optional)
        
        Returns:
            Path to the saved visualization file
        """
        if not analysis_results or not analysis_results.get('valid', False):
            return None
        
        # Extract key components from analysis
        profile_lines = analysis_results.get('profile_lines', [])
        limit_orders = analysis_results.get('limit_orders', [])
        liquidation_zones = analysis_results.get('liquidation_zones', [])
        pixel_to_price_ratio = analysis_results.get('pixel_to_price_ratio', 0)
        
        # Create timestamp for the visualization file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create the visualization figures
        fig = plt.figure(figsize=(15, 10))
        
        # Set up the grid for subplots
        gs = fig.add_gridspec(3, 3)
        
        # 1. Create main price chart with order profile lines
        ax_main = fig.add_subplot(gs[0:2, 0:2])
        self._create_price_chart(ax_main, profile_lines)
        
        # 2. Create strength distribution chart
        ax_strength = fig.add_subplot(gs[0, 2])
        self._create_strength_distribution(ax_strength, profile_lines)
        
        # 3. Create order concentration chart
        ax_orders = fig.add_subplot(gs[1, 2])
        self._create_order_concentration(ax_orders, limit_orders)
        
        # 4. Create liquidation zones chart
        ax_liq = fig.add_subplot(gs[2, 0])
        self._create_liquidation_zones(ax_liq, liquidation_zones)
        
        # 5. Create color-type distribution chart
        ax_colors = fig.add_subplot(gs[2, 1])
        self._create_color_distribution(ax_colors, profile_lines)
        
        # 6. Create summary statistics
        ax_stats = fig.add_subplot(gs[2, 2])
        self._create_summary_statistics(ax_stats, analysis_results)
        
        # Set overall title
        plt.suptitle(f'AIO-LiqTrader Analysis Results - {timestamp}', fontsize=16)
        plt.tight_layout()
        
        # Save the visualization
        output_path = os.path.join(self.visualization_dir, f'analysis_{timestamp}.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Create a separate HTML report with interactive elements
        html_path = self._create_html_report(analysis_results, timestamp, screenshot_path)
        
        return output_path, html_path
    
    def _create_price_chart(self, ax, profile_lines):
        """Create the main price chart with order profile lines"""
        # Sort lines by price
        sorted_lines = sorted(profile_lines, key=lambda x: x['price'])
        
        if not sorted_lines:
            ax.text(0.5, 0.5, 'No profile lines detected', ha='center', va='center')
            ax.set_title('Price Chart - Order Profile Lines')
            return
        
        # Extract prices and strengths
        prices = [line['price'] for line in sorted_lines]
        strengths = [line.get('strength_score', 0) for line in sorted_lines]
        types = [line['color'] for line in sorted_lines]
        
        # Determine min and max prices with padding
        price_range = max(prices) - min(prices)
        min_price = min(prices) - price_range * 0.05
        max_price = max(prices) + price_range * 0.05
        
        # Draw horizontal lines for each price level with color based on strength
        for i, line in enumerate(sorted_lines):
            strength_category = line.get('strength', 'weak')
            color = self.strength_colors.get(strength_category, '#00CC00')
            line_width = 1 + line.get('strength_score', 0) * 4  # Scale line width by strength
            
            # Draw the horizontal line
            ax.axhline(y=line['price'], xmin=0.05, xmax=0.95, 
                      linewidth=line_width, color=color, 
                      alpha=0.7, label=strength_category if i == 0 else "")
            
            # Add small price label on the right
            if line.get('strength_score', 0) > 0.5:  # Only label stronger lines to avoid clutter
                ax.text(1.01, line['price'], f"${line['price']:.0f}", 
                       va='center', ha='left', fontsize=8, color=color)
        
        # Set y-axis limits
        ax.set_ylim(min_price, max_price)
        
        # Remove x-axis ticks as they're not relevant here
        ax.set_xticks([])
        ax.set_xlim(0, 1)
        
        # Set title and labels
        ax.set_title('Price Chart - Order Profile Lines')
        ax.set_ylabel('Price (USD)')
        
        # Create legend for strength categories
        handles = [mpatches.Patch(color=color, label=cat.capitalize()) 
                 for cat, color in self.strength_colors.items()]
        ax.legend(handles=handles, loc='upper right', title='Line Strength')
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.3)
    
    def _create_strength_distribution(self, ax, profile_lines):
        """Create chart showing distribution of line strengths"""
        if not profile_lines:
            ax.text(0.5, 0.5, 'No profile lines detected', ha='center', va='center')
            ax.set_title('Line Strength Distribution')
            return
        
        # Count lines in each strength category
        strength_counts = {
            'very_strong': 0,
            'strong': 0,
            'moderate': 0,
            'weak': 0
        }
        
        for line in profile_lines:
            strength = line.get('strength', 'weak')
            if strength in strength_counts:
                strength_counts[strength] += 1
        
        # Create bar chart
        categories = list(strength_counts.keys())
        counts = list(strength_counts.values())
        colors = [self.strength_colors[cat] for cat in categories]
        
        ax.bar(categories, counts, color=colors)
        
        # Add count labels on top of bars
        for i, count in enumerate(counts):
            ax.text(i, count + 0.5, str(count), ha='center')
        
        # Set title and labels
        ax.set_title('Line Strength Distribution')
        ax.set_ylabel('Number of Lines')
        
        # Rotate x-axis labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    def _create_order_concentration(self, ax, limit_orders):
        """Create chart showing concentration of limit orders"""
        if not limit_orders:
            ax.text(0.5, 0.5, 'No limit orders detected', ha='center', va='center')
            ax.set_title('Limit Order Concentration')
            return
        
        # Group orders by type (buy/sell)
        buy_orders = [order for order in limit_orders if order.get('type') == 'buy']
        sell_orders = [order for order in limit_orders if order.get('type') == 'sell']
        
        # Create a horizontal bar chart
        categories = ['Buy Orders', 'Sell Orders']
        counts = [len(buy_orders), len(sell_orders)]
        colors = ['green', 'red']
        
        ax.barh(categories, counts, color=colors)
        
        # Add count labels inside bars
        for i, count in enumerate(counts):
            ax.text(count / 2, i, str(count), va='center', ha='center', 
                   color='white', fontweight='bold')
        
        # Set title and labels
        ax.set_title('Limit Order Concentration')
        ax.set_xlabel('Number of Orders')
    
    def _create_liquidation_zones(self, ax, liquidation_zones):
        """Create chart showing liquidation zones"""
        if not liquidation_zones:
            ax.text(0.5, 0.5, 'No liquidation zones detected', ha='center', va='center')
            ax.set_title('Liquidation Zones')
            return
        
        # Sort zones by price
        sorted_zones = sorted(liquidation_zones, key=lambda x: x['price'])
        
        # Extract prices and lengths (use length as a proxy for significance)
        prices = [zone['price'] for zone in sorted_zones]
        lengths = [zone['length'] for zone in sorted_zones]
        
        # Normalize lengths for visualization
        normalized_lengths = [length / max(lengths) for length in lengths]
        
        # Determine min and max prices with padding
        price_range = max(prices) - min(prices)
        min_price = min(prices) - price_range * 0.05
        max_price = max(prices) + price_range * 0.05
        
        # Create horizontal bars for each liquidation zone
        for i, zone in enumerate(sorted_zones):
            width = 0.4 + (normalized_lengths[i] * 0.5)  # Scale width by normalized length
            ax.barh(zone['price'], width, height=price_range/50, color='purple', alpha=0.7)
            
            # Add price label
            ax.text(width + 0.02, zone['price'], f"${zone['price']:.0f}", 
                   va='center', fontsize=8)
        
        # Set y-axis limits
        ax.set_ylim(min_price, max_price)
        
        # Set title and labels
        ax.set_title('Liquidation Zones')
        ax.set_ylabel('Price (USD)')
        ax.set_xlim(0, 1)
        ax.set_xticks([])
    
    def _create_color_distribution(self, ax, profile_lines):
        """Create chart showing distribution of line colors"""
        if not profile_lines:
            ax.text(0.5, 0.5, 'No profile lines detected', ha='center', va='center')
            ax.set_title('Line Color Distribution')
            return
        
        # Count lines by color
        color_counts = {
            'yellow': 0,
            'white': 0,
            'red': 0,
            'orange': 0
        }
        
        for line in profile_lines:
            color = line.get('color', 'white')
            if color in color_counts:
                color_counts[color] += 1
        
        # Create pie chart
        labels = list(color_counts.keys())
        sizes = list(color_counts.values())
        colors = [self.line_type_colors[label] for label in labels]
        
        # Filter out zero values
        filtered_labels = []
        filtered_sizes = []
        filtered_colors = []
        for i, size in enumerate(sizes):
            if size > 0:
                filtered_labels.append(labels[i])
                filtered_sizes.append(sizes[i])
                filtered_colors.append(colors[i])
        
        if not filtered_sizes:
            ax.text(0.5, 0.5, 'No color data available', ha='center', va='center')
            ax.set_title('Line Color Distribution')
            return
        
        # Create the pie chart with adjusted colors for visibility
        ax.pie(filtered_sizes, labels=filtered_labels, colors=filtered_colors, 
              autopct='%1.1f%%', startangle=90, 
              textprops={'color': 'black', 'fontweight': 'bold'})
        
        # Set title
        ax.set_title('Line Color Distribution')
    
    def _create_summary_statistics(self, ax, analysis_results):
        """Create text summary of analysis statistics"""
        # Turn off axis
        ax.axis('off')
        
        # Extract key statistics
        profile_lines = analysis_results.get('profile_lines', [])
        limit_orders = analysis_results.get('limit_orders', [])
        liquidation_zones = analysis_results.get('liquidation_zones', [])
        pixel_to_price = analysis_results.get('pixel_to_price_ratio', 0)
        
        # Count strength categories
        strength_counts = {
            'very_strong': 0,
            'strong': 0,
            'moderate': 0,
            'weak': 0
        }
        
        for line in profile_lines:
            strength = line.get('strength', 'weak')
            if strength in strength_counts:
                strength_counts[strength] += 1
        
        # Count order types
        buy_orders = len([o for o in limit_orders if o.get('type') == 'buy'])
        sell_orders = len([o for o in limit_orders if o.get('type') == 'sell'])
        
        # Calculate average price
        if profile_lines:
            avg_price = sum(line['price'] for line in profile_lines) / len(profile_lines)
        else:
            avg_price = 0
        
        # Create summary text
        summary = [
            "ANALYSIS SUMMARY",
            "----------------",
            f"Total Order Profile Lines: {len(profile_lines)}",
            f"Very Strong Lines: {strength_counts['very_strong']}",
            f"Strong Lines: {strength_counts['strong']}",
            f"Moderate Lines: {strength_counts['moderate']}",
            f"Weak Lines: {strength_counts['weak']}",
            "",
            f"Total Limit Orders: {len(limit_orders)}",
            f"Buy Orders: {buy_orders}",
            f"Sell Orders: {sell_orders}",
            f"Buy/Sell Ratio: {buy_orders/sell_orders:.2f}" if sell_orders > 0 else "Buy/Sell Ratio: ∞",
            "",
            f"Liquidation Zones: {len(liquidation_zones)}",
            f"Pixel-to-Price Ratio: {pixel_to_price:.2f} USD/px",
            f"Average Price Level: ${avg_price:.2f}"
        ]
        
        # Add the summary text to the plot
        ax.text(0, 1, '\n'.join(summary), va='top', ha='left', fontsize=10)
    
    def _create_html_report(self, analysis_results, timestamp, screenshot_path=None):
        """Create an interactive HTML report with all analysis results"""
        profile_lines = analysis_results.get('profile_lines', [])
        limit_orders = analysis_results.get('limit_orders', [])
        liquidation_zones = analysis_results.get('liquidation_zones', [])
        
        # Create HTML file path
        html_path = os.path.join(self.visualization_dir, f'analysis_{timestamp}.html')
        
        # Start building HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>AIO-LiqTrader Analysis Report - {timestamp}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                .header {{ background-color: #333; color: white; padding: 10px; text-align: center; }}
                .section {{ margin-bottom: 20px; padding: 15px; border: 1px solid #ddd; }}
                .section-title {{ background-color: #f0f0f0; padding: 5px; margin-bottom: 10px; }}
                .summary-stats {{ display: flex; flex-wrap: wrap; }}
                .stat-box {{ flex: 1; min-width: 200px; margin: 5px; padding:
                 10px; border: 1px solid #ddd; text-align: center; }}
                .stat-value {{ font-size: 24px; font-weight: bold; margin-bottom: 5px; }}
                .stat-label {{ font-size: 14px; color: #666; }}
                .chart {{ width: 100%; height: 400px; margin-top: 20px; }}
                table {{ width: 100%; border-collapse: collapse; }}
                th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
                tr:hover {{ background-color: #f5f5f5; }}
                .very_strong {{ color: #FF0000; font-weight: bold; }}
                .strong {{ color: #FF6600; font-weight: bold; }}
                .moderate {{ color: #FFCC00; }}
                .weak {{ color: #00CC00; }}
                .buy {{ color: green; }}
                .sell {{ color: red; }}
                .screenshot {{ max-width: 100%; margin-top: 20px; border: 1px solid #ddd; }}
            </style>
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>AIO-LiqTrader Analysis Report</h1>
                    <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
                
                <div class="section">
                    <div class="section-title">
                        <h2>Summary Statistics</h2>
                    </div>
                    
                    <div class="summary-stats">
                        <div class="stat-box">
                            <div class="stat-value">{len(profile_lines)}</div>
                            <div class="stat-label">Order Profile Lines</div>
                        </div>
                        <div class="stat-box">
                            <div class="stat-value">{len(limit_orders)}</div>
                            <div class="stat-label">Limit Orders</div>
                        </div>
                        <div class="stat-box">
                            <div class="stat-value">{len(liquidation_zones)}</div>
                            <div class="stat-label">Liquidation Zones</div>
                        </div>
                        <div class="stat-box">
                            <div class="stat-value">{analysis_results.get('pixel_to_price_ratio', 0):.2f}</div>
                            <div class="stat-label">USD/pixel</div>
                        </div>
                    </div>
                    
                    <div class="chart">
                        <canvas id="strengthDistribution"></canvas>
                    </div>
                </div>
        """
        
        # Add Order Profile Lines section with table
        html_content += """
                <div class="section">
                    <div class="section-title">
                        <h2>Order Profile Lines</h2>
                    </div>
                    
                    <table id="profileLinesTable">
                        <tr>
                            <th>Price (USD)</th>
                            <th>Color</th>
                            <th>Length (px)</th>
                            <th>Strength</th>
                            <th>Score</th>
                        </tr>
        """
        
        # Add rows for each profile line
        for line in sorted(profile_lines, key=lambda x: x.get('strength_score', 0), reverse=True):
            strength = line.get('strength', 'weak')
            html_content += f"""
                        <tr>
                            <td>${line['price']:.2f}</td>
                            <td>{line['color'].capitalize()}</td>
                            <td>{line['length']}</td>
                            <td class="{strength}">{strength.replace('_', ' ').capitalize()}</td>
                            <td>{line.get('strength_score', 0):.2f}</td>
                        </tr>
            """
        
        html_content += """
                    </table>
                    
                    <div class="chart">
                        <canvas id="priceLevelsChart"></canvas>
                    </div>
                </div>
        """
        
        # Add Limit Orders section
        html_content += """
                <div class="section">
                    <div class="section-title">
                        <h2>Limit Orders Analysis</h2>
                    </div>
                    
                    <div class="chart">
                        <canvas id="orderTypesChart"></canvas>
                    </div>
                </div>
        """
        
        # Add Liquidation Zones section
        html_content += """
                <div class="section">
                    <div class="section-title">
                        <h2>Liquidation Zones</h2>
                    </div>
                    
                    <table id="liquidationTable">
                        <tr>
                            <th>Price (USD)</th>
                            <th>Length (px)</th>
                        </tr>
        """
        
        # Add rows for each liquidation zone
        for zone in sorted(liquidation_zones, key=lambda x: x['price']):
            html_content += f"""
                        <tr>
                            <td>${zone['price']:.2f}</td>
                            <td>{zone['length']}</td>
                        </tr>
            """
        
        html_content += """
                    </table>
                </div>
        """
        
        # Add screenshot if available
        if screenshot_path and os.path.exists(screenshot_path):
            html_content += f"""
                <div class="section">
                    <div class="section-title">
                        <h2>Analysis Screenshot</h2>
                    </div>
                    <img class="screenshot" src="{os.path.relpath(screenshot_path, os.path.dirname(html_path))}" alt="Analysis Screenshot">
                </div>
            """
        
        # Add JavaScript for charts
        html_content += """
                <script>
                    // Strength Distribution Chart
                    const strengthCtx = document.getElementById('strengthDistribution').getContext('2d');
                    new Chart(strengthCtx, {
                        type: 'bar',
                        data: {
                            labels: ['Very Strong', 'Strong', 'Moderate', 'Weak'],
                            datasets: [{
                                label: 'Number of Lines',
                                data: [
        """
        
        # Count strength categories
        strength_counts = {
            'very_strong': 0,
            'strong': 0,
            'moderate': 0,
            'weak': 0
        }
        
        for line in profile_lines:
            strength = line.get('strength', 'weak')
            if strength in strength_counts:
                strength_counts[strength] += 1
        
        html_content += f"{strength_counts['very_strong']}, {strength_counts['strong']}, {strength_counts['moderate']}, {strength_counts['weak']}"
        
        html_content += """
                                ],
                                backgroundColor: [
                                    '#FF0000',
                                    '#FF6600',
                                    '#FFCC00',
                                    '#00CC00'
                                ]
                            }]
                        },
                        options: {
                            responsive: true,
                            plugins: {
                                title: {
                                    display: true,
                                    text: 'Line Strength Distribution'
                                },
                            }
                        }
                    });

                    // Price Levels Chart
                    const priceCtx = document.getElementById('priceLevelsChart').getContext('2d');
                    new Chart(priceCtx, {
                        type: 'scatter',
                        data: {
                            datasets: [
        """
        
        # Create datasets for each strength category
        for strength, color in [('very_strong', '#FF0000'), ('strong', '#FF6600'), 
                              ('moderate', '#FFCC00'), ('weak', '#00CC00')]:
            html_content += f"""
                                {{
                                    label: '{strength.replace('_', ' ').capitalize()} Lines',
                                    data: [
            """
            
            # Add data points for this strength category
            points = []
            for line in profile_lines:
                if line.get('strength') == strength:
                    # Use price as y-coordinate and strength score * 10 as point size
                    point_size = max(5, int(line.get('strength_score', 0) * 20))
                    points.append(f"{{x: {line['length']}, y: {line['price']}, r: {point_size}}}")
            
            html_content += ", ".join(points)
            
            html_content += f"""
                                    ],
                                    backgroundColor: '{color}',
                                    pointRadius: 5,
                                }},
            """
        
        html_content += """
                            ]
                        },
                        options: {
                            responsive: true,
                            plugins: {
                                title: {
                                    display: true,
                                    text: 'Order Profile Lines by Price and Length'
                                },
                                tooltip: {
                                    callbacks: {
                                        label: function(context) {
                                            return `Price: $${context.parsed.y.toFixed(2)}, Length: ${context.parsed.x}px`;
                                        }
                                    }
                                }
                            },
                            scales: {
                                x: {
                                    title: {
                                        display: true,
                                        text: 'Line Length (pixels)'
                                    }
                                },
                                y: {
                                    title: {
                                        display: true,
                                        text: 'Price (USD)'
                                    }
                                }
                            }
                        }
                    });

                    // Order Types Chart
                    const orderCtx = document.getElementById('orderTypesChart').getContext('2d');
        """
        
        # Count buy and sell orders
        buy_orders = len([o for o in limit_orders if o.get('type') == 'buy'])
        sell_orders = len([o for o in limit_orders if o.get('type') == 'sell'])
        
        html_content += f"""
                    new Chart(orderCtx, {{
                        type: 'pie',
                        data: {{
                            labels: ['Buy Orders', 'Sell Orders'],
                            datasets: [{{
                                data: [{buy_orders}, {sell_orders}],
                                backgroundColor: ['green', 'red']
                            }}]
                        }},
                        options: {{
                            responsive: true,
                            plugins: {{
                                title: {{
                                    display: true,
                                    text: 'Limit Order Distribution'
                                }}
                            }}
                        }}
                    }});
                </script>
            </div>
        </body>
        </html>
        """
        
        # Write the HTML content to file
        with open(html_path, 'w') as file:
            file.write(html_content)
        
        return html_path


# Integration with main application
def create_visualization_system(config):
    """Factory function to create and return a visualization engine instance"""
    return VisualizationEngine(config)