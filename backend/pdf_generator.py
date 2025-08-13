"""
AI Agriculture Advisor - PDF Report Generator
This module generates comprehensive PDF reports with crop advice and predictions.
"""

from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.pdfgen import canvas
import os
from datetime import datetime
import io

class CropReportGenerator:
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
    
    def _setup_custom_styles(self):
        """Setup custom paragraph styles for the report"""
        # Title style
        self.title_style = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.darkgreen
        )
        
        # Section header style
        self.section_style = ParagraphStyle(
            'CustomSection',
            parent=self.styles['Heading2'],
            fontSize=16,
            spaceAfter=12,
            spaceBefore=20,
            textColor=colors.darkblue
        )
        
        # Subsection style
        self.subsection_style = ParagraphStyle(
            'CustomSubsection',
            parent=self.styles['Heading3'],
            fontSize=14,
            spaceAfter=8,
            spaceBefore=12,
            textColor=colors.darkred
        )
        
        # Normal text style
        self.normal_style = ParagraphStyle(
            'CustomNormal',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceAfter=6,
            alignment=TA_LEFT
        )
        
        # Highlight style
        self.highlight_style = ParagraphStyle(
            'CustomHighlight',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceAfter=6,
            alignment=TA_LEFT,
            textColor=colors.darkgreen,
            leftIndent=20
        )
    
    def generate_crop_report(self, crop_data, weather_data, crop_advice, price_predictions, output_path):
        """
        Generate a comprehensive PDF report for crop advice
        
        Args:
            crop_data: Dictionary containing crop information
            weather_data: Dictionary containing weather information
            crop_advice: Dictionary containing crop-specific advice
            price_predictions: Dictionary containing price predictions
            output_path: Path where to save the PDF
        """
        try:
            # Create PDF document
            doc = SimpleDocTemplate(output_path, pagesize=A4)
            story = []
            
            # Add title page
            story.extend(self._create_title_page(crop_data))
            
            # Add table of contents
            story.extend(self._create_table_of_contents())
            
            # Add crop information section
            story.extend(self._create_crop_info_section(crop_data))
            
            # Add weather information section
            story.extend(self._create_weather_section(weather_data))
            
            # Add crop advice section
            story.extend(self._create_advice_section(crop_advice))
            
            # Add price predictions section
            story.extend(self._create_price_section(price_predictions))
            
            # Add recommendations summary
            story.extend(self._create_summary_section(crop_advice, price_predictions))
            
            # Build PDF
            doc.build(story)
            
            print(f"PDF report generated successfully: {output_path}")
            return True
            
        except Exception as e:
            print(f"Error generating PDF report: {str(e)}")
            return False
    
    def _create_title_page(self, crop_data):
        """Create the title page of the report"""
        elements = []
        
        # Main title
        title = Paragraph("AI Agriculture Advisor Report", self.title_style)
        elements.append(title)
        elements.append(Spacer(1, 0.5*inch))
        
        # Subtitle
        subtitle = Paragraph(f"Crop: {crop_data.get('crop_type', 'N/A')}", self.section_style)
        elements.append(subtitle)
        elements.append(Spacer(1, 0.3*inch))
        
        # Location
        location = Paragraph(f"Location: {crop_data.get('location', 'N/A')}", self.normal_style)
        elements.append(location)
        elements.append(Spacer(1, 0.2*inch))
        
        # Date
        current_date = datetime.now().strftime("%B %d, %Y")
        date_text = Paragraph(f"Report Generated: {current_date}", self.normal_style)
        elements.append(date_text)
        elements.append(Spacer(1, 0.5*inch))
        
        # Description
        description = Paragraph(
            "This report provides comprehensive agricultural advice including weather analysis, "
            "crop-specific recommendations, and market price predictions to help optimize "
            "your farming operations.",
            self.normal_style
        )
        elements.append(description)
        elements.append(Spacer(1, 0.5*inch))
        
        return elements
    
    def _create_table_of_contents(self):
        """Create table of contents"""
        elements = []
        
        toc_title = Paragraph("Table of Contents", self.section_style)
        elements.append(toc_title)
        elements.append(Spacer(1, 0.2*inch))
        
        toc_items = [
            "1. Crop Information",
            "2. Weather Analysis",
            "3. Crop-Specific Advice",
            "4. Price Predictions",
            "5. Recommendations Summary"
        ]
        
        for item in toc_items:
            toc_item = Paragraph(item, self.normal_style)
            elements.append(toc_item)
            elements.append(Spacer(1, 0.1*inch))
        
        elements.append(Spacer(1, 0.3*inch))
        return elements
    
    def _create_crop_info_section(self, crop_data):
        """Create crop information section"""
        elements = []
        
        # Section header
        section_title = Paragraph("1. Crop Information", self.section_style)
        elements.append(section_title)
        
        # Crop details table
        crop_info = [
            ["Crop Type", crop_data.get('crop_type', 'N/A')],
            ["Location", crop_data.get('location', 'N/A')],
            ["State", crop_data.get('state', 'N/A')],
            ["Growing Season", crop_data.get('season', 'N/A')]
        ]
        
        crop_table = Table(crop_info, colWidths=[2*inch, 3*inch])
        crop_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        elements.append(crop_table)
        elements.append(Spacer(1, 0.3*inch))
        
        return elements
    
    def _create_weather_section(self, weather_data):
        """Create weather information section"""
        elements = []
        
        # Section header
        section_title = Paragraph("2. Weather Analysis", self.section_style)
        elements.append(section_title)
        
        # Current weather table
        weather_info = [
            ["Parameter", "Value", "Status"],
            ["Temperature", f"{weather_data.get('temperature', 'N/A')}°C", 
             self._get_temperature_status(weather_data.get('temperature', 25))],
            ["Humidity", f"{weather_data.get('humidity', 'N/A')}%", 
             self._get_humidity_status(weather_data.get('humidity', 60))],
            ["Pressure", f"{weather_data.get('pressure', 'N/A')} hPa", "Normal"],
            ["Wind Speed", f"{weather_data.get('wind_speed', 'N/A')} m/s", "Normal"],
            ["Conditions", weather_data.get('description', 'N/A'), "Current"]
        ]
        
        weather_table = Table(weather_info, colWidths=[1.5*inch, 1.5*inch, 2*inch])
        weather_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        elements.append(weather_table)
        elements.append(Spacer(1, 0.3*inch))
        
        # Weather description
        if 'demo_mode' in weather_data:
            weather_note = Paragraph(
                "Note: Weather data is in demo mode. For real-time data, please provide OpenWeatherMap API key.",
                self.highlight_style
            )
            elements.append(weather_note)
            elements.append(Spacer(1, 0.2*inch))
        
        return elements
    
    def _create_advice_section(self, crop_advice):
        """Create crop advice section"""
        elements = []
        
        # Section header
        section_title = Paragraph("3. Crop-Specific Advice", self.section_style)
        elements.append(section_title)
        
        if 'error' in crop_advice:
            error_msg = Paragraph(f"Error: {crop_advice['error']}", self.highlight_style)
            elements.append(error_msg)
            elements.append(Spacer(1, 0.2*inch))
            return elements
        
        # Current conditions
        conditions_title = Paragraph("Current Growing Conditions", self.subsection_style)
        elements.append(conditions_title)
        
        conditions_data = [
            ["Condition", "Value", "Status"],
            ["Temperature", crop_advice.get('current_conditions', {}).get('temperature', 'N/A'), ""],
            ["Humidity", crop_advice.get('current_conditions', {}).get('humidity', 'N/A'), ""],
            ["Rainfall", crop_advice.get('current_conditions', {}).get('rainfall', 'N/A'), ""]
        ]
        
        conditions_table = Table(conditions_data, colWidths=[1.5*inch, 1.5*inch, 2*inch])
        conditions_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgreen),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightyellow),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        elements.append(conditions_table)
        elements.append(Spacer(1, 0.3*inch))
        
        # Recommendations
        recommendations_title = Paragraph("Recommendations", self.subsection_style)
        elements.append(recommendations_title)
        
        recommendations = crop_advice.get('recommendations', {})
        for key, value in recommendations.items():
            key_formatted = key.replace('_', ' ').title()
            rec_text = Paragraph(f"<b>{key_formatted}:</b> {value}", self.normal_style)
            elements.append(rec_text)
            elements.append(Spacer(1, 0.1*inch))
        
        elements.append(Spacer(1, 0.2*inch))
        
        # Risk assessment
        risk_title = Paragraph("Risk Assessment", self.subsection_style)
        elements.append(risk_title)
        
        risk_assessment = crop_advice.get('risk_assessment', {})
        risk_level = risk_assessment.get('risk_level', 'Unknown')
        risk_color = self._get_risk_color(risk_level)
        
        risk_level_text = Paragraph(f"Risk Level: <b>{risk_level}</b>", self.normal_style)
        elements.append(risk_level_text)
        elements.append(Spacer(1, 0.1*inch))
        
        risks = risk_assessment.get('risks', [])
        for risk in risks:
            risk_text = Paragraph(f"• {risk}", self.normal_style)
            elements.append(risk_text)
            elements.append(Spacer(1, 0.05*inch))
        
        elements.append(Spacer(1, 0.2*inch))
        
        # Next actions
        actions_title = Paragraph("Recommended Actions", self.subsection_style)
        elements.append(actions_title)
        
        actions = crop_advice.get('next_actions', [])
        for action in actions:
            action_text = Paragraph(f"• {action}", self.normal_style)
            elements.append(action_text)
            elements.append(Spacer(1, 0.05*inch))
        
        elements.append(Spacer(1, 0.3*inch))
        
        return elements
    
    def _create_price_section(self, price_predictions):
        """Create price predictions section"""
        elements = []
        
        # Section header
        section_title = Paragraph("4. Price Predictions", self.section_style)
        elements.append(section_title)
        
        if not price_predictions or 'error' in price_predictions:
            error_msg = Paragraph("Price prediction data not available.", self.highlight_style)
            elements.append(error_msg)
            elements.append(Spacer(1, 0.2*inch))
            return elements
        
        # Current price
        current_price = price_predictions.get('current_price', 'N/A')
        current_price_text = Paragraph(f"Current Market Price: <b>₹{current_price}</b>", self.normal_style)
        elements.append(current_price_text)
        elements.append(Spacer(1, 0.2*inch))
        
        # Price trend
        trend_title = Paragraph("Price Trend Forecast (Next 3 Months)", self.subsection_style)
        elements.append(trend_title)
        
        trend_data = price_predictions.get('trend', {})
        if trend_data and 'dates' in trend_data and 'predictions' in trend_data:
            dates = trend_data['dates']
            predictions = trend_data['predictions']
            
            trend_table_data = [["Month", "Predicted Price (₹)"]]
            for date, pred in zip(dates, predictions):
                month = datetime.strptime(date, '%Y-%m-%d').strftime('%B %Y')
                trend_table_data.append([month, f"{pred:.2f}"])
            
            trend_table = Table(trend_table_data, colWidths=[2.5*inch, 2.5*inch])
            trend_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 11),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            elements.append(trend_table)
            elements.append(Spacer(1, 0.3*inch))
        
        # Market insights
        insights_title = Paragraph("Market Insights", self.subsection_style)
        elements.append(insights_title)
        
        insights = [
            "• Monitor local market conditions regularly",
            "• Consider storage options if prices are expected to rise",
            "• Plan harvest timing based on price predictions",
            "• Stay informed about government policies affecting crop prices"
        ]
        
        for insight in insights:
            insight_text = Paragraph(insight, self.normal_style)
            elements.append(insight_text)
            elements.append(Spacer(1, 0.05*inch))
        
        elements.append(Spacer(1, 0.3*inch))
        
        return elements
    
    def _create_summary_section(self, crop_advice, price_predictions):
        """Create summary and final recommendations section"""
        elements = []
        
        # Section header
        section_title = Paragraph("5. Recommendations Summary", self.section_style)
        elements.append(section_title)
        
        # Executive summary
        summary_title = Paragraph("Executive Summary", self.subsection_style)
        elements.append(summary_title)
        
        summary_text = Paragraph(
            "Based on the analysis of current weather conditions, crop requirements, and market trends, "
            "this report provides actionable recommendations to optimize your farming operations. "
            "Follow the specific advice for your crop type and monitor conditions regularly.",
            self.normal_style
        )
        elements.append(summary_text)
        elements.append(Spacer(1, 0.2*inch))
        
        # Key recommendations
        key_recs_title = Paragraph("Key Recommendations", self.subsection_style)
        elements.append(key_recs_title)
        
        key_recommendations = [
            "1. Follow the irrigation schedule based on current weather conditions",
            "2. Apply fertilizers at the recommended timing for optimal crop growth",
            "3. Monitor for signs of stress or disease based on risk assessment",
            "4. Plan harvest and marketing based on price predictions",
            "5. Maintain regular crop monitoring and adjust practices as needed"
        ]
        
        for rec in key_recommendations:
            rec_text = Paragraph(rec, self.normal_style)
            elements.append(rec_text)
            elements.append(Spacer(1, 0.1*inch))
        
        elements.append(Spacer(1, 0.3*inch))
        
        # Contact information
        contact_title = Paragraph("Additional Support", self.subsection_style)
        elements.append(contact_title)
        
        contact_info = [
            "For additional agricultural advice, consider consulting:",
            "• Local agricultural extension officers",
            "• Agricultural universities and research institutions",
            "• Experienced local farmers",
            "• Agricultural consultants"
        ]
        
        for info in contact_info:
            info_text = Paragraph(info, self.normal_style)
            elements.append(info_text)
            elements.append(Spacer(1, 0.05*inch))
        
        elements.append(Spacer(1, 0.3*inch))
        
        return elements
    
    def _get_temperature_status(self, temp):
        """Get temperature status for display"""
        if temp is None or temp == 'N/A':
            return "Unknown"
        try:
            temp_val = float(temp)
            if temp_val < 15:
                return "Cold"
            elif temp_val > 35:
                return "Hot"
            else:
                return "Optimal"
        except:
            return "Unknown"
    
    def _get_humidity_status(self, humidity):
        """Get humidity status for display"""
        if humidity is None or humidity == 'N/A':
            return "Unknown"
        try:
            hum_val = float(humidity)
            if hum_val < 40:
                return "Dry"
            elif hum_val > 80:
                return "Humid"
            else:
                return "Optimal"
        except:
            return "Unknown"
    
    def _get_risk_color(self, risk_level):
        """Get color for risk level display"""
        risk_colors = {
            'Low': colors.green,
            'Medium': colors.orange,
            'High': colors.red
        }
        return risk_colors.get(risk_level, colors.black)

# Example usage
if __name__ == "__main__":
    # Test PDF generation with sample data
    generator = CropReportGenerator()
    
    sample_crop_data = {
        'crop_type': 'Rice',
        'location': 'Punjab',
        'state': 'Punjab',
        'season': 'Kharif'
    }
    
    sample_weather_data = {
        'temperature': 28.5,
        'humidity': 65,
        'pressure': 1013,
        'wind_speed': 5.2,
        'description': 'scattered clouds',
        'demo_mode': True
    }
    
    sample_crop_advice = {
        'crop_type': 'Rice',
        'current_conditions': {
            'temperature': '28.5°C (Optimal)',
            'humidity': '65% (Optimal)',
            'rainfall': '0mm'
        },
        'recommendations': {
            'water_management': 'Maintain normal irrigation schedule',
            'fertilizer_application': 'Before transplanting and at panicle initiation',
            'irrigation_schedule': 'Daily during early growth, every 2-3 days later',
            'general_care': 'Conditions are favorable for crop growth'
        },
        'risk_assessment': {
            'risk_level': 'Low',
            'risks': ['No significant risks identified']
        },
        'next_actions': [
            'Regular monitoring of crop health',
            'Prepare for next fertilizer application'
        ]
    }
    
    sample_price_predictions = {
        'current_price': 2500,
        'trend': {
            'dates': ['2024-01-15', '2024-02-15', '2024-03-15'],
            'predictions': [2550, 2600, 2650]
        }
    }
    
    # Generate sample report
    output_path = "../data/sample_crop_report.pdf"
    success = generator.generate_crop_report(
        sample_crop_data,
        sample_weather_data,
        sample_crop_advice,
        sample_price_predictions,
        output_path
    )
    
    if success:
        print(f"Sample PDF report generated: {output_path}")
    else:
        print("Failed to generate sample PDF report") 